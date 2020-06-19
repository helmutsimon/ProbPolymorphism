# coding=utf-8

"""
Mutation_analysis_by_chrom.py analyses variance due to context aggregated over all mutation
directions for 1-mers, 3-mers, 5-mers and 7-mers by chromosome. Results are calculated for 2 cases:
marginalised over central allele and not. Input files are produced by bayes_analysis.py.

Parameters are job number and suffix to identify input files.

Sample run statement is:
nohup python3 ~/repos/ProbPolymorphism/context/mutation_analysis_by_chrom.py mabc001 b > mabc001.txt&

"""


import numpy as np
import pandas as pd
import os
import sys
import gzip, pickle
from time import time
from itertools import product
import statsmodels
from statsmodels.stats.weightstats import DescrStatsW
import click
from scitrack import CachingLogger, get_file_hexdigest

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import probpoly_bayes

LOGGER = CachingLogger(create_dir=True)

@click.command()
@click.argument('job_no')
@click.argument('suffix')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, suffix, dir):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + statsmodels.__name__ + ', version = ' + statsmodels.__version__,
                       label="Imported module".ljust(30))

    result = pd.DataFrame(columns=['kmer', 'variance', 'Marginalise over central base?'])
    heads = ['kmer', 'chromosome', 'variance', 'Marginalise over central base?']
    i = 2
    for kmer_variable in [1, 2, 3]:
        for chrom in range(1, 23):
            filename = dir + '/var_dict_humanch' + str(chrom) + suffix + '.pklz'     #suffix = b
            infile = open(filename, 'r')
            LOGGER.input_file(infile.name)
            infile.close()
            with gzip.open(filename, 'rb') as var_data:
                var_data = pickle.load(var_data)
            var_counts = probpoly_bayes.unpack_var_contexts_to_dataframe(kmer_variable, var_data)

            filename = dir + '/context_dict_ccrfishch' + str(chrom) + '.pklz'
            if chrom == 1:
                filename = dir + '/context_dict_ccrfish001.pklz'
            infile = open(filename, 'r')
            LOGGER.input_file(infile.name)
            infile.close()
            with gzip.open(filename, 'rb') as context_data:
                context_data = pickle.load(context_data)
            context_counts = probpoly_bayes.unpack_all_contexts_to_dataframe(kmer_variable, context_data)

            #Reformat context counts by repeating columns to match snv_densities dataframe.
            extended_context_counts, cols = probpoly_bayes.reformat_context_counts(context_counts, var_counts)
            extended_context_counts.set_index(context_counts.index, inplace=True)
            snv_densities = var_counts / extended_context_counts

            #Calculate variance across mutation types, marginalising over the central base.
            context_ratios = context_counts.div(context_counts.sum(axis=1), axis=0)
            extended_context_ratios, cols = probpoly_bayes.reformat_context_counts(context_ratios, snv_densities)
            extended_context_ratios.set_index(context_ratios.index, inplace=True)
            con_weighted = (snv_densities * extended_context_ratios).sum(axis=1)
            u = DescrStatsW(con_weighted, weights=context_counts.sum(axis=1), ddof=0)
            print('Marginalised variance due to ' + str(2 * kmer_variable + 1) + ' -mers = ', u.var)
            row = np.array([2 * kmer_variable + 1, chrom, u.var, 'yes'])
            row = pd.Series(row, index=heads, name=i)
            result = result.append(row)
            i += 1

            #Calculate variance conditioned on kmer, not marginalising over the central base.
            #Firstly we reorganise the SNV densities table so that rows correspond to kmers
            # (including central base) and columns correspond to the derived base.
            contexts_generator = product('ACGT', repeat=2 * kmer_variable + 1)
            contexts = tuple(''.join(context) for context in contexts_generator)
            kmer_densities = np.zeros((len(contexts), 4))
            kmer_densities = pd.DataFrame(kmer_densities, index=contexts, columns=['C', 'T', 'A', 'G'])
            for context in snv_densities.index:
                for mut in snv_densities.columns:
                    ref = mut[0]
                    derived = mut[3]
                    kmer = context[0: kmer_variable] + ref + context[kmer_variable: 2 * kmer_variable]
                    kmer_densities.loc[kmer, derived] = snv_densities.loc[context, mut]

            #We also reorganise context counts into counts of kmers.
            kmer_counts = np.zeros((len(contexts)))
            kmer_counts = pd.Series(kmer_counts, index=contexts)
            for kmer in kmer_counts.index:
                context = kmer[0: kmer_variable] + kmer[kmer_variable + 1: 2 * kmer_variable + 1]
                ref = kmer[kmer_variable]
                kmer_counts[kmer] = context_counts.loc[context, ref]

            #Calculate the weighted variance over the full kmer.
            v = DescrStatsW(kmer_densities.sum(axis=1), weights=kmer_counts, ddof=0)
            print('Unmarginalised variance due to ' + str(2 * kmer_variable + 1) + ' -mers = ', v.var)
            row = np.array([2 * kmer_variable + 1, chrom, v.var, 'no'])
            row = pd.Series(row, index=heads, name=i)
            result = result.append(row)
            i += 1

    print(result)
    filename = dir + "/aggregated_results" + job_no + ".csv"
    result.to_csv(filename, sep=',')
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))

if __name__ == "__main__":
    main()
