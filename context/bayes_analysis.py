# coding=utf-8

""" bayes_analysis.py samples posterior distributions of variance due to context for various
    mutation types using Bayesian binomial model with a beta prior. Input files are variant and
    context count files produced by count_variants.py and count_contexts_with_re.py or their
    intergenic equivalents.

    Parameters are job number, variation dict filename, context dict filename and number of samples.
    Filenames include subdirectory, if any, but not suffix (pklz assumed).

    Sample run statement is:
    nohup python3 ~/repos/ProbPolymorphism/context/bayes_analysis.py ba010
    data/var_dict_humanch1c4 data/context_dict_humanch1c6 2e4 > ba010.txt &"""


import pickle
import os
import sys
import gzip
import numpy as np
from time import time
import pandas as pd
import click
from scitrack import CachingLogger, get_file_hexdigest

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import probpoly_bayes

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('var_filename', type=click.Path(exists=True))
@click.argument('context_filename', type=click.Path(exists=True))
@click.argument('draws', type=float)
@click.argument('prior', type=float, default=1.)
@click.option('-n', '--nocpg', is_flag=True, default=False, help='Condition out CpG variants.')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, var_filename, context_filename, draws, prior, nocpg, dir):
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
    draws = int(draws)

    infile_name = var_filename
    infile = open(infile_name, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    with gzip.open(infile_name, 'rb') as var_data:
        var_data = pickle.load(var_data)

    infile_name = context_filename
    infile = open(infile_name, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    with gzip.open(infile_name, 'rb') as context_data:
        context_data = pickle.load(context_data)

    for kmer_variable, split in zip([1, 2, 3], [1, 4, 64]):
    #for kmer_variable, split in zip([4], [1024]):
        contexts = probpoly_bayes.unpack_all_contexts_to_dataframe(kmer_variable, context_data)
        duration = time() - start_time
        print('Unpacked all_contexts for ', kmer_variable, '-mers at', "%.2f" % (duration / 60.), 'minutes.')
        sys.stdout.flush()
        variants = probpoly_bayes.unpack_var_contexts_to_dataframe(kmer_variable, var_data)
        duration = time() - start_time
        print('Unpacked var_contexts for ', kmer_variable, '-mers at', "%.2f" % (duration / 60.), 'minutes.')
        sys.stdout.flush()

        outfile_name, dataset = dir + '/var_counts_' + str(kmer_variable) + job_no + '.pklz', variants
        with gzip.open(outfile_name, 'wb') as outfile:
            pickle.dump(dataset, outfile)
        outfile = open(outfile_name, 'r')
        LOGGER.output_file(outfile.name)
        outfile.close()

        outfile_name, dataset = dir + '/context_counts_' + str(kmer_variable) + job_no + '.pklz', contexts
        with gzip.open(outfile_name, 'wb') as outfile:
            pickle.dump(dataset, outfile)
        outfile = open(outfile_name, 'r')
        LOGGER.output_file(outfile.name)
        outfile.close()

        ncols = 12
        contexts, columns = probpoly_bayes.reformat_context_counts(contexts, variants)
        w_var_samples = probpoly_bayes.calculate_variances(variants, contexts, split, prior, draws, ncols, columns)

        outfile_name = 'data/bayes_var_samples_' + job_no + '_k=' + str(kmer_variable) + '.pklz'
        with gzip.open(outfile_name, 'wb') as outfile:
            pickle.dump(w_var_samples, outfile)
        outfile = open(outfile_name, 'r')
        LOGGER.output_file(outfile.name)
        outfile.close()
        del w_var_samples

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))

if __name__ == "__main__":
    main()
