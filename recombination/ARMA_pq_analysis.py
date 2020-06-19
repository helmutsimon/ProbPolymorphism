# coding=utf-8


""" ARMA_pq_analysis.py uses MCMC so solve a linear model of SNP rates against recombination rates
    using ARMA(p, q) residuals. It analyses each mutation direction for a single chromosome.
    It uses data from merge_male_and_female_recombination_rates.py or
    sample_ensembl_for_recombination.py and selects an optimal model for each mutation direction.

    Arguments are job no, input file name (MCMC) draws and sex. Sample run statement (on Mac) is:

    Example run statement:
    nohup python3 /Users/helmutsimon/repos/MutationProbVariance/recombination/ARMA_pq_analysis.py ARMA001
      std_rates_with_variants_sex-averaged_ch1.csv  > arma001.txt &"""


import numpy as np
import pandas as pd
import os
import sys
from time import time
import pymc3
from pymc3 import *
import theano
import scipy
from sklearn.linear_model import LinearRegression
import sklearn
import warnings
import statsmodels
import statsmodels.tsa.api as smt
import click
from scitrack import CachingLogger, get_file_hexdigest

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import recombination

warnings.simplefilter("ignore", FutureWarning)

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('csv_name')
@click.argument('draws', type=int, default=10000)
@click.option('-s', '--sex', default=None)
@click.option('-d', '--dir', default='Recombination_data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, csv_name, draws, sex, dir):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + job_no + ".log"    # change
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__, label="Imported module ")
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__, label="Imported module ")
    LOGGER.log_message('Name = ' + theano.__name__ + ', version = ' + theano.__version__, label="Imported module ")
    LOGGER.log_message('Name = ' + pymc3.__name__ + ', version = ' + pymc3.__version__, label="Imported module ")
    LOGGER.log_message('Name = ' + scipy.__name__ + ', version = ' + scipy.__version__, label="Imported module ")
    LOGGER.log_message('Name = ' + statsmodels.__name__ + ', version = ' + statsmodels.__version__,
                       label="Imported module ")
    LOGGER.log_message('Name = ' + sklearn.__name__ + ', version = ' + sklearn.__version__, label="Imported module ")
    pd.set_option('display.max_columns', None)
    infile = open(dir + '/' + csv_name, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    data_table = pd.read_csv(dir + '/' + csv_name, sep=',', index_col=0)
    data_table = recombination.correct_missing_data(data_table, 'LOCF', sex)
    if sex is None:
        std_col = 'stdrate'
    else:
        std_col = 'stdrate_' + sex
    std_rates = data_table[std_col].values
    result_rows = list()
    columns = ['mtype', 'mrate', 'p', 'q', 'alpha', 'alpha25', 'alpha975', 'beta', 'beta25', 'beta975',
               'pval', 'r2', 'variance', 'variance25', 'variance975', 'mutprop', 'mutprop25', 'mutprop975', 'mutperco']
    for mtype in data_table.columns[5:17]:
        variant_counts = data_table.loc[:, mtype]
        allele = mtype[0]
        base_counts = data_table.loc[:, allele]
        base_counts = base_counts.values.flatten()
        finitemask = np.isfinite(base_counts)
        var_rates = variant_counts / base_counts
        var_rates = var_rates.values.flatten()
        var_rates = var_rates[finitemask]
        print('\nAvge. mutation rate for ', mtype, ' = ', np.mean(var_rates))
        xvals = std_rates.reshape(-1, 1)
        lmodel = LinearRegression()
        lmodel.fit(xvals, var_rates)
        residuals = var_rates - lmodel.predict(xvals)
        sys.stdout.flush()
        print('Slope, intercept, mean of residuals = ',
          '%.8f' % lmodel.coef_[0], '%.8f' % lmodel.intercept_, '%.12f' % np.mean(residuals))
        orders = recombination.evaluate_ARMA_models(residuals, 10, 4)
        best_order = orders[0]
        best_mdl = smt.ARMA(residuals, order=best_order).fit(method='mle', trend='nc', disp=0)
        print(best_mdl.summary())
        mdl_details = recombination.save_model_details(best_mdl)
        p = mdl_details['order'][0]
        q = mdl_details['order'][1]
        if q == 0:
            trace = recombination.run_MCMC_ARp(std_rates, var_rates, draws, p, best_mdl)
        else:
            trace = recombination.run_MCMC_ARMApq(std_rates, var_rates, draws, mdl_details)
        neg_slope = sum(t['beta'] <= 0 for t in trace)
        print('Probability slope <=0 = ', neg_slope / draws)
        ss_tot = np.var(var_rates)
        obs_mean = np.mean(var_rates)
        ss_reg_vars = [np.var(obs_mean - t['alpha'] - t['beta'] * std_rates) for t in trace]
        ss_res_vars = [np.var(var_rates - t['alpha'] - t['beta'] * std_rates) for t in trace]
        r2_vars = 1 - (ss_res_vars / ss_tot)
        vmean = np.mean(ss_tot - ss_res_vars)
        vlow = np.percentile(ss_tot - ss_res_vars, 2.5)
        vhigh = np.percentile(ss_tot - ss_res_vars, 97.5)
        print('Variance in SNP rate due to recombination = ', vmean)
        print('CIs = ', vlow, vhigh)
        print('Proportion of var. in SNP rate explained by recombination (R2) = ', np.mean(r2_vars))
        print('Standard deviation of R2 = ', np.std(r2_vars))
        print('95% interval for R2 = ', np.percentile(r2_vars, 2.5), np.percentile(r2_vars, 97.5))
        av_mrate = np.mean(var_rates)
        print('Average SNP rate = ', av_mrate)
        mutation_rate = 1.29e-8
        intercept_mean = np.mean([t['alpha'] for t in trace])
        intercept_CI_low = np.percentile([t['alpha'] for t in trace], 2.5)
        intercept_CI_high = np.percentile([t['alpha'] for t in trace], 97.5)
        rfunc = lambda x: (av_mrate - x) / av_mrate
        print('Proportion muts due to recomb = ', rfunc(intercept_mean),
              'CIs = ', rfunc(intercept_CI_low), rfunc(intercept_CI_high))
        recomb_rate = .0116 / (100 * 1e4)
        mutsper = (rfunc(intercept_mean) * mutation_rate) / recomb_rate
        print('Mutations per CO = ', mutsper)
        sys.stdout.flush()
        s = summary(trace,  varnames=['alpha', 'beta'])
        result_row = [mtype, np.mean(var_rates), p, q,
                      s.loc['alpha', 'mean'],  s.loc['alpha', 'hpd_2.5'], s.loc['alpha', 'hpd_97.5'],
                      s.loc['beta', 'mean'],   s.loc['beta', 'hpd_2.5'],  s.loc['beta', 'hpd_97.5'],
                      neg_slope / draws, np.mean(r2_vars), vmean, vlow, vhigh,
                      rfunc(intercept_mean), rfunc(intercept_CI_low), rfunc(intercept_CI_high), mutsper]
        result_rows.append(result_row)
    results_table = pd.DataFrame(result_rows, columns=columns)
    results_table.set_index('mtype', inplace=True)
    if sex is None:
        outfile_name = dir + '/ARMApq_results_' + job_no + '.csv'
    else:
        outfile_name = dir + '/ARMApq_results_' + sex + '_' + job_no + '.csv'
    results_table.to_csv(outfile_name)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

