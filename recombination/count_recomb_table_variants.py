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

    #Mutation rates per chromosome are calculated from data at Jonsson et al.  Parental influence on human
    # germline de novomutations in 1,548 trios from Iceland. ('Estimate mutation rates from Jonsson data.ipynb)
    result_rows = list()
    chroms = np.arange(1, 23).astype(str).tolist()
    for i, chrom in enumerate(chroms):
        csv_filename = folder + '/recomb_table_all_sexes_ch' + chrom + '.csv'
        infile = open(csv_filename, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        data_table = pd.read_csv(csv_filename, sep=',', index_col=0)
        data_table = recombination.correct_missing_data(data_table, 'LOCF', sex)
        variants_profiled = data_table.iloc[:, np.arange(5, 17)]
        variant_counts = variants_profiled.sum(axis=1)
        total_variants =  variant_counts.sum()
        result_rows.append([i, total_variants])
    results_table = pd.DataFrame(result_rows)
    outfile_name = folder + '/recomb_var_counts' + job_no + '.csv'
    results_table.to_csv(outfile_name)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

