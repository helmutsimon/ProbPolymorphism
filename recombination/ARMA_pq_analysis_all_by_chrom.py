# coding=utf-8


""" ARMA_pq_analysis_all_by_chrom.py uses MCMC so solve a linear model of SNP rates against
    recombination rates using ARMA(p, q) residuals. It aggregates variants across mutation
    direction, and processes a list of chromosomes provided. using data on preferred ARMA models
    produced by ARMA_select_models.py. It uses data from merge_male_and_female_recombination_rates.py
    or sample_ensembl_for_recombination.py and details of preferred models from ARMA_select_models.py.

    Arguments are job no, chromosomes to process, number of (MCMC) draws, sex, suffix for model files, data directory.
    Sample run statement (on Mac) is:

    Example run statement is:
    nohup python3 /Users/helmutsimon/repos/ProbPolymorphism/recombination/ARMA_pq_analysis_all_by_chrom.py
    ARMAabctest -x _020  > ARMAabctest.txt &"""


import numpy as np
import pandas as pd
import os
import sys
from time import time
import pymc3
from pymc3 import *
import gzip, pickle
import theano
import warnings
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
@click.option('-c', '--chroms', default=None)
@click.option('-r', '--draws', type=int, default=50000)
@click.option('-s', '--sex', default='sexav')
@click.option('-x', '--suffixes' , default=None)
@click.option('-d', '--folder', default='Recombination_data', type=click.Path(),
              help='Directory name for data and log files. Defaults is Recombination_data')
def main(job_no, chroms, draws, sex, suffixes, folder):
    start_time = time()
    if not os.path.exists(folder):
        os.makedirs(folder)
    LOGGER.log_file_path = folder + "/" + str(os.path.basename(__file__)) + job_no + ".log"    # change
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

    #Mutation rates per chromosome are calculated from data at Jonsson et al.  Parental influence on human
    # germline de novomutations in 1,548 trios from Iceland. ('Estimate mutation rates from Jonsson data.ipynb)
    mrates = [1.1045541764661985e-08, 1.2481509352581898e-08, 1.254443516411994e-08, 1.2609734521720365e-08,
              1.216379148788216e-08, 1.2228991967962778e-08, 1.2298304077726808e-08, 1.3325693328599174e-08,
              1.0711369887343474e-08, 1.238059175011868e-08, 1.2241940318060874e-08, 1.2117457093135447e-08,
              1.0174746106096945e-08, 1.0146311894484388e-08, 1.0516600482736078e-08, 1.2597261162425896e-08,
              1.1681529656302903e-08, 1.1855256275211491e-08, 1.214570124735936e-08, 1.1756514975959873e-08,
              8.965863348091259e-09, 9.024242643357694e-09]
    result_rows = list()
    columns = ['snvdens', 'p', 'q', 'alpha', 'alpha25', 'alpha975', 'beta', 'beta25', 'beta975', 'slopem',
               'slopem25', 'slopem75', 'pval', 'r2', 'variance', 'variance25', 'variance975', 'mutprop', 'mutprop25',
               'mutprop975', 'mutperco', 'mutperco25', 'mutperco975']
    if chroms is None:
        chroms = np.arange(1, 23).astype(str).tolist()
    else:
        chroms = chroms.split(',')
    for chromplace, chrom in enumerate(chroms):
        print(chromplace, 'Chromosome ', chrom)
    if suffixes is None:
        suffixes = len(chroms) * [""]
    else:
        suffixes = suffixes.split(',')
    results = list()
    for i, chrom in enumerate(chroms):
        csv_filename = folder + '/recomb_table_all_sexes_ch' + chrom + '.csv'
        infile = open(csv_filename, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        data_table = pd.read_csv(csv_filename, sep=',', index_col=0)
        data_table = recombination.correct_missing_data(data_table, 'LOCF', sex)
        variants_profiled = data_table.iloc[:, np.arange(5, 17)]
        variant_counts = variants_profiled.sum(axis=1)
        var_rates = variant_counts / 10000
        std_col = 'stdrate_' + sex
        std_rates = data_table[std_col].values
        print('Avge. & var. of mutation rate ', np.mean(var_rates), np.var(var_rates))
        suffix = suffixes[i]
        file_name = folder + '/ARMA_model_ch' + str(chrom) + suffix + '.pklz'
        infile = open(file_name, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        with gzip.open(file_name, 'rb') as model_details:
            model_details = pickle.load(model_details)
        trace = recombination.run_MCMC_ARMApq(std_rates, var_rates, draws, model_details)
        p = model_details['order'][0]
        q = model_details['order'][1]
        neg_slope = sum(t['beta'] <= 0 for t in trace)
        print('Chrom', chrom, 'variates with slope <=0 ', neg_slope)
        print('Chrom', chrom, 'probability slope <=0 = ', neg_slope / draws)
        ss_tot = np.var(var_rates)
        ss_res_vars = [np.var(var_rates - t['alpha'] - t['beta'] * std_rates) for t in trace]
        r2_vars = 1 - (ss_res_vars / ss_tot)
        variance_variates = ss_tot - ss_res_vars
        vmean = np.mean(variance_variates)
        vlow = np.percentile(variance_variates, 2.5)
        vhigh = np.percentile(variance_variates, 97.5)
        chr_results = pd.DataFrame(variance_variates, columns=['vars'])
        chr_results['chr'] = chrom
        results.append(chr_results)
        snv_dens = np.mean(var_rates)
        intercept_mean = np.mean([t['alpha'] for t in trace])
        intercept_CI_low = np.percentile([t['alpha'] for t in trace], 2.5)
        intercept_CI_high = np.percentile([t['alpha'] for t in trace], 97.5)
        rfunc = lambda x: (snv_dens - x) / snv_dens
        mfunc = lambda x: x * mutation_rate / (snv_dens * 0.0116)
        print('Proportion muts due to recomb = ', rfunc(intercept_mean),
              'CIs = ', rfunc(intercept_CI_high), rfunc(intercept_CI_low))
        recomb_rate = np.mean(std_rates) * 0.0116 / (100 * 1e4)
        mutation_rate = mrates[i]
        mutsper = (rfunc(intercept_mean) * mutation_rate) / recomb_rate
        print('Mutations per CO = ', mutsper)
        sys.stdout.flush()
        s = summary(trace,  varnames=['alpha', 'beta'])
        result_row = [np.mean(var_rates), p, q,
                      s.loc['alpha', 'mean'],  s.loc['alpha', 'hpd_2.5'], s.loc['alpha', 'hpd_97.5'],
                      s.loc['beta', 'mean'],   s.loc['beta', 'hpd_2.5'],  s.loc['beta', 'hpd_97.5'],
                      mfunc(s.loc['beta', 'mean']), mfunc(s.loc['beta', 'hpd_2.5']), mfunc(s.loc['beta', 'hpd_97.5']),
                      neg_slope / draws, np.mean(r2_vars),
                      vmean, vlow, vhigh, rfunc(intercept_mean), rfunc(intercept_CI_low), rfunc(intercept_CI_high),
                      mutsper, (rfunc(intercept_CI_high) * mutation_rate) / recomb_rate,
                      (rfunc(intercept_CI_low) * mutation_rate) / recomb_rate]
        result_rows.append(result_row)
    results_table = pd.DataFrame(result_rows, columns=columns)
    outfile_name = folder + '/ARMApq_results_' + job_no + '.csv'
    results_table.to_csv(outfile_name)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    results = pd.concat(results)
    outfile_name = folder + '/ARMApq_variates_' + sex + '_'+ job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(results, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

