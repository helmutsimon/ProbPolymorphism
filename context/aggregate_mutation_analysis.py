# coding=utf-8

""" aggregate_mutation_analysis.py analyses variance due to context aggregated over all mutation
    directions for 1-mers, 3-mers, 5-mers and 7-mers. Results are calculated for 2 cases:
    marginalised over central allele and not. Input files are produced by bayes_analysis.py.

    Parameters are job number and suffix to identify input files.

    Sample run statement is:
    nohup python3 ~/repos/ProbPolymorphism/context/aggregate_mutation_analysis.py ama001 ba100 > ama001.txt&"""


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

    #Find variance due to CpG
    filename = dir + '/var_counts_1' + suffix + '.pklz'
    infile = open(filename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    with gzip.open(filename, 'rb') as var_counts:
        var_counts = pickle.load(var_counts)
    filename = dir + '/context_counts_1' + suffix + '.pklz'
    infile = open(filename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    with gzip.open(filename, 'rb') as context_counts:
        context_counts = pickle.load(context_counts)

    cpg_contexts = context_counts.loc['CG', 'C'] + context_counts.loc['TG', 'C'] + context_counts.loc['AG', 'C'] + \
                   context_counts.loc['GG', 'C'] + \
                   context_counts.loc['CC', 'G'] + context_counts.loc['CT', 'G'] + context_counts.loc['CA', 'G'] + \
                   context_counts.loc['CG', 'G']
    CpG_ratio = cpg_contexts / context_counts.values.sum()
    non_cpg_contexts = context_counts.values.sum() - cpg_contexts
    print('Total CpG sites           : ', cpg_contexts)
    print('Total intronic sites      : ', context_counts.values.sum())
    print('Proportion CpG sites      : ', CpG_ratio)
    var_counts['C'] = var_counts['C->T'] + var_counts['C->A'] + var_counts['C->G']
    var_counts['G'] = var_counts['G->T'] + var_counts['G->A'] + var_counts['G->C']
    CpG_vars = var_counts.loc['CG', 'C'] + var_counts.loc['TG', 'C'] + var_counts.loc['AG', 'C'] + \
               var_counts.loc['GG', 'C'] + \
               var_counts.loc['CC', 'G'] + var_counts.loc['CT', 'G'] + var_counts.loc['CA', 'G'] + \
               var_counts.loc['CG', 'G']
    print('Total CpG variants        : ', CpG_vars)
    non_CpG_vars = var_counts.values.sum() - CpG_vars
    m1 = CpG_vars / cpg_contexts
    m0 = non_CpG_vars / non_cpg_contexts
    m_ave = var_counts.values.sum() / context_counts.values.sum()
    print('SNV density at CpG sites  : ', m1)
    print('SNV density at other sites: ', m0)
    print('Average SNV density       : ', m_ave)
    t1 = CpG_ratio * (m1 - m_ave) ** 2
    t2 = (1 - CpG_ratio) * (m0 - m_ave) ** 2
    print('Variance due to CpG sites : ', t1 + t2)
    LOGGER.log_message("%.2e" % (t1 + t2), label="Variance due to CpG".ljust(50))

    #Deal with the 1-mer case.
    var_counts['C'] = var_counts['C->T'] + var_counts['C->A'] + var_counts['C->G']
    var_counts['T'] = var_counts['T->C'] + var_counts['T->A'] + var_counts['T->G']
    var_counts['A'] = var_counts['A->T'] + var_counts['A->C'] + var_counts['A->G']
    var_counts['G'] = var_counts['G->T'] + var_counts['G->A'] + var_counts['G->C']
    variant_counts = var_counts.sum(axis=0)
    variant_counts = variant_counts[variant_counts.index.isin(['C', 'T', 'A', 'G'])]
    con_counts = context_counts.sum(axis=0)
    mut_rates = variant_counts / con_counts
    w = DescrStatsW(mut_rates, weights=con_counts, ddof=0)
    row = np.array([1, w.var, 'no'])
    row = pd.Series(row, index=result.columns, name=0)
    result = result.append(row)
    row = np.array([1, 0.0, 'yes'])
    row = pd.Series(row, index=result.columns, name=1)
    result = result.append(row)

    i = 2
    for kmer_variable in [1, 2, 3]:
        filename = dir + '/var_counts_' + str(kmer_variable) + suffix + '.pklz'
        infile = open(filename, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        with gzip.open(filename, 'rb') as var_counts:
            var_counts = pickle.load(var_counts)
        filename = dir + '/context_counts_' + str(kmer_variable) + suffix + '.pklz'
        infile = open(filename, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        with gzip.open(filename, 'rb') as context_counts:
            context_counts = pickle.load(context_counts)

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
        row = np.array([2 * kmer_variable + 1, u.var, 'yes'])
        row = pd.Series(row, index=result.columns, name=i)
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
        row = np.array([2 * kmer_variable + 1, v.var, 'no'])
        row = pd.Series(row, index=result.columns, name=i)
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
