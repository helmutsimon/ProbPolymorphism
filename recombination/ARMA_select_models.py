# coding=utf-8


""" ARMA_select_models.py selects ARMA models to best fit residuals for linear regression of SNV
    density on recombination rate for selected chromosomes. It uses data from
    merge_male_and_female_recombination_rates.py or sample_ensembl_for_recombination.py.

    Example runstatement is:
    nohup python3 /Users/helmutsimon/repos/MutationProbVariance/recombination/ARMA_select_models.py ARMAsm_001
      1,3,5,7 -d Recombination_data  > ARMAsm_001.txt &"""


import numpy as np
import pandas as pd
import os
import sys
from time import time
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
@click.argument('sex')
@click.option('-c', '--chroms', default=None)
@click.option('-r', '--rank', type=int, default=0, help='Rank of model to accept (ordered by AIC.)')
@click.option('-d', '--dir', default='Recombination_data', type=click.Path(),
              help='Directory name for data and log files. Defaults is Recombination_data')
def main(job_no, sex, chroms, rank, dir):
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
    LOGGER.log_message('Name = ' + scipy.__name__ + ', version = ' + scipy.__version__, label="Imported module ")
    LOGGER.log_message('Name = ' + statsmodels.__name__ + ', version = ' + statsmodels.__version__,
                       label="Imported module ")
    LOGGER.log_message('Name = ' + sklearn.__name__ + ', version = ' + sklearn.__version__, label="Imported module ")
    pd.set_option('display.max_columns', None)
    if chroms is None:
        chroms = np.arange(1, 23).astype(str).tolist()
    else:
        chroms = chroms.split(',')
    if rank:
        LOGGER.log_message("%1d" % rank, label="Rank of model to select (best=0).".ljust(30))
    for chrom in chroms:
        csv_name = dir + '/recomb_table_all_sexes_ch' + chrom + '.csv'
        infile = open(csv_name, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        data_table = pd.read_csv(csv_name, sep=',', index_col=0)
        data_table = recombination.correct_missing_data(data_table, 'LOCF', sex)
        std_col = 'stdrate_' + sex
        std_rates = data_table[std_col].values
        variants_profiled = data_table.iloc[:, np.arange(5, 17)]
        variant_counts = variants_profiled.sum(axis=1)
        var_rates = variant_counts / 10000
        print('\n\nChromosome number   = ' + chrom)
        print('Avge. mutation rate = ', np.mean(var_rates))
        xvals = std_rates.reshape(-1, 1)
        lmodel = LinearRegression()
        lmodel.fit(xvals, var_rates)
        residuals = var_rates - lmodel.predict(xvals)
        sys.stdout.flush()
        print('Slope, intercept, mean of residuals = ',
          '%.8f' % lmodel.coef_[0], '%.8f' % lmodel.intercept_, '%.12f' % np.mean(residuals))
        orders = recombination.evaluate_ARMA_models(residuals, 10, 4)
        best_order = orders[rank]
        best_mdl = smt.ARMA(residuals, order=best_order).fit(method='mle', trend='nc', disp=0)
        print(best_mdl.summary())
        outfile_name = dir + '/ARMA_model_ch' + chrom + '_' +  job_no + '.pklz'
        recombination.save_model_details(best_mdl, outfile_name)
        outfile = open(outfile_name, 'r')
        LOGGER.output_file(outfile.name)
        outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

