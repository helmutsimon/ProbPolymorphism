# coding=utf-8


import numpy as np
import pandas as pd
import gzip, pickle
from pymc3 import *
import theano.tensor as tt
from pymc3.distributions.continuous import Normal, HalfNormal, get_tau_sd, Flat
from scipy import stats
import statsmodels.tsa.api as smt


def correct_missing_data(data_table, method, sex):
    """Outlier bins those with no variants or adjacent. This code treats
    such outliers as missing data by two possible methods: simply deleting the bins
    (method='delete') or using last observation carried forward (method='LOCF').
    Neither method is ideal, but there are a number of genomic locations where no
    SNPs are annotated."""
    variants_profiled = data_table.iloc[:, np.arange(5, 17)]
    variant_counts = variants_profiled.sum(axis=1)
    filtr1 = variant_counts > 0
    if sex is None:
        filtr2 = data_table['seqbin'] == 1
    else:
        filtr2 = data_table['seqbin_' + sex] == 1
    if sex is None:
        std_col = 'stdrate'
    else:
        std_col = 'stdrate_' + sex
    # Also filter out sites adjacent to a bad site
    filtr = filtr1 & np.pad(filtr1[: -1], (1, 0), 'constant', constant_values=1)\
                & np.pad(filtr1[1:], (0,1), 'constant', constant_values=1)\
                & filtr2
    if method == 'LOCF':
        for rowix, value in enumerate(filtr):
            if value == 0:
                if rowix > 0:
                    data_table.iloc[rowix, np.arange(5, 17)] =\
                                    data_table.iloc[rowix - 1, np.arange(5, 17)]
                    data_table.loc[rowix, std_col] = data_table.loc[rowix - 1, std_col]
                else:
                    data_table.drop(rowix, inplace=True)
    elif method == 'delete':
        data_table = data_table[filtr]
    else:
        assert False, 'Invalid method'
    print('Number of rows altered (no variants or adjacent) = ', len(filtr) - sum(filtr))
    return data_table

def evaluate_ARMA_models(y, AR_lim, MA_lim):
    orders, aics = list(), list()
    for i in range(1, AR_lim + 1):
        for j in range(0, MA_lim +1):
            try:
                mdlt = smt.ARMA(y, order=(i, j)).fit(method='mle', trend='nc', disp=-1)
                orders.append((i, j))
                aics.append(mdlt.aic)
            except ValueError:
                print(i, j, ' not convergent.')
            except np.linalg.LinAlgError as e:
                if "SVD did not converge" in str(e):
                    print(i, j, ' not convergent.')
                else:
                    raise
    assert len(orders) == len(aics), 'Error: Different lengths.'
    print('Number of convergent models = ', len(orders))
    aics, orders = zip(*sorted(zip(aics, orders)))
    for a, b in zip(aics[:5], orders[:5]):
        print(a, b)
    return orders


def run_MCMC_ARMApq(x, y, draws, model):
    """Derive slope and intercept for ARMA(p, q) model with known p nd q.
    We initially fit a model to the residuals using statsmodels.tsa.api.ARMA.
    Details of this model (as produced by ARMA_select_models.py) are provided as a parameter
    to the present function to allow derivation of reasonably accurate prior distributions for phi and theta.
    If these priors are too broad, the MCMC will not converge in a reasonable time."""

    p = model['order'][0]
    q = model['order'][1]
    phi_means = model['tab']['params'].to_numpy()[:p]
    phi_sd = model['tab']['bse'].to_numpy()[:p]
    print(phi_means, phi_sd)
    theta_means = model['tab']['params'].to_numpy()[-q:]
    theta_sd = model['tab']['bse'].to_numpy()[-q:]

    # NaN values can occur in std err (see e.g. stackoverflow.com/questions/35675693 & 210228.
    # We therefore conservatively replace any NaNs by 0.1.
    phi_sd = np.nan_to_num(phi_sd) + np.isnan(phi_sd) * 0.1
    theta_sd = np.nan_to_num(theta_sd) + np.isnan(theta_sd) * 0.1
    m = p + q
    with Model() as model9:
        alpha = Normal('alpha', mu=0, sd=10)
        beta = Normal('beta', mu=0, sd=10)
        sd = HalfNormal('sd', sd=10)
        if p == 1:
            phi = Normal('phi', mu=phi_means[0], sd=phi_sd[0])
        else:
            phi = Normal('phi', mu=phi_means, sd=phi_sd, shape=p)
        if q ==1:
            theta = Normal('theta', mu=theta_means[0], sd=theta_sd[0])
        else:
            theta = Normal('theta', mu=theta_means, sd=theta_sd, shape=q)
        y = tt.as_tensor(y)
        x = tt.as_tensor(x)
        y_r = y[m:]
        x_r = x[m:]
        resids = y - beta * x - alpha

        if p == 1:
            u = phi * resids[p - 1: -1]
        else:
            u = tt.add(*[phi[i] * resids[p - (i + 1):-(i + 1)] for i in range(p)])
        eps = resids[p:] - u
        if q == 1:
            v = theta * eps[q - 1: -1]
        else:
            v = tt.add(*[theta[i] * eps[q - (i + 1):-(i + 1)] for i in range(q)])
        mu = alpha + beta * x_r + u[q:] + v
        data = Normal('y_r', mu=mu, sd=sd, observed=y_r)

    with model9:
        if q == 1:
            step = Metropolis([phi])
        else:
            step = Metropolis([phi, theta])
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=step, progressbar=False)

    print(summary(trace, varnames=['alpha', 'beta', 'sd', 'phi', 'theta']))
    #plt.show(forestplot(trace, varnames=['alpha', 'beta', 'sd', 'phi', 'theta']))
    #traceplot(trace, varnames=['alpha', 'beta', 'sd', 'phi', 'theta'])
    return trace


def run_MCMC_ARMA_multi(x, y, draws, models):
    """Derive slope and intercept for ARMA model across multiple chromosomes. The slope and intercept are held constant
    across the chromosomes, while the ARMA model for residuals can vary across chromosomes. The details of
    these models are created by ARMS_select_models.py.
    If q=0, i.e. the model is pure AR(p) a separate function should be used."""
    num_chroms = len(models)

    with Model() as model9:
        alpha = Normal('alpha', mu=0, sd=10)
        beta = Normal('beta', mu=0, sd=10)
        sd = HalfNormal('sd', sd=10)
        steps = list()
        var_names1 = list()

        for c in range(num_chroms):
            p = models[c]['order'][0]
            q = models[c]['order'][1]
            m = p + q
            phi_means = models[c]['tab']['params'].to_numpy()[:p]
            phi_sd = models[c]['tab']['bse'].to_numpy()[:p]
            theta_means = models[c]['tab']['params'].to_numpy()[-q:]
            theta_sd = models[c]['tab']['bse'].to_numpy()[-q:]
            # NaN values can occur in std err (see e.g. stackoverflow.com/questions/35675693 & 210228.
            # We therefore conservatively replace any NaNs by 0.1.
            phi_sd = np.nan_to_num(phi_sd) + np.isnan(phi_sd) * 0.1
            theta_sd = np.nan_to_num(theta_sd) + np.isnan(theta_sd) * 0.1

            if p == 1:
                phi = Normal('phi_%i' % c, mu=phi_means[0], sd=phi_sd[0])
            else:
                phi = Normal('phi_%i' % c, mu=phi_means, sd=phi_sd, shape=p)
            if q == 1:
                theta = Normal('theta_%i' % c, mu=theta_means[0], sd=theta_sd[0])
            else:
                theta = Normal('theta_%i' % c, mu=theta_means, sd=theta_sd, shape=q)
            y[c] = tt.as_tensor(y[c])
            x[c] = tt.as_tensor(x[c])
            y_r = y[c][m:]
            x_r = x[c][m:]
            resids = y[c] - beta * x[c] - alpha
            if p == 1:
                u = phi * resids[p - 1: -1]
            else:
                u = tt.add(*[phi[i] * resids[p - (i + 1):-(i + 1)] for i in range(p)])
            eps = resids[p:] - u
            if q == 1:
                v = theta * eps[q - 1: -1]
            else:
                v = tt.add(*[theta[i] * eps[q - (i + 1):-(i + 1)] for i in range(q)])
            mu = alpha + beta * x_r + u[q:] + v

            data = Normal('y_r_%i' % c, mu=mu, sd=sd, observed=y_r)
            step = Metropolis([phi, theta])  # See pymc3 #1304.
            steps.append(step)
            var_names1.append('phi_%i' % c)
            var_names1.append('theta_%i' % c)

    with model9:
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=steps, progressbar=False)

    print(summary(trace, varnames=['alpha', 'beta'] + var_names1))
    #plt.show(forestplot(trace, varnames=['alpha', 'beta']))
    #traceplot(trace, varnames=['alpha', 'beta'])
    return trace


def run_MCMC_ARp(x, y, draws, p, resmdl):
    phi_means = resmdl.params[:p]
    phi_sd = resmdl.bse[:p]

    with Model() as model8:
        alpha = Normal('alpha', mu=0, sd=10)
        beta = Normal('beta', mu=0, sd=10)
        sd = HalfNormal('sd', sd=10)
        phi = Normal('phi', mu=phi_means, sd=phi_sd, shape=p)
        y = tt.as_tensor(y)
        x = tt.as_tensor(x)
        y_r = y[p:]
        x_r = x[p:]
        resids = y - beta * x - alpha

        u = tt.add(*[phi[i] * resids[p - (i + 1):-(i + 1)] for i in range(p)])
        mu = alpha + beta * x_r + u
        data = Normal('y_r', mu=mu, sd=sd, observed=y_r)

    with model8:
        if p == 1:
            step = None
        else:
            step = Metropolis([phi])
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=step, progressbar=False)

    print(summary(trace, varnames=['alpha', 'beta', 'sd', 'phi']))
    #plt.show(forestplot(trace, varnames=['alpha', 'beta', 'sd', 'phi']))
    #traceplot(trace, varnames=['alpha', 'beta', 'sd', 'phi'])
    return trace

def save_model_details(mdl, fname=None):
    params = mdl.params
    bse = mdl.bse
    frame = {'params': params, 'bse': bse}
    tab = pd.DataFrame(frame)
    best_mdl_details = dict({'order': (len(mdl.arparams), len(mdl.maparams)), 'tab': tab})
    if fname is not None:
        with gzip.open(fname, 'wb') as outfile:
            pickle.dump(best_mdl_details, outfile)
    return best_mdl_details
