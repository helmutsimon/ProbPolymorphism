# coding=utf-8

""" merge_chromosome_data.py Merges the files for individual chromosomes produced by count_variants.py
    or count_contexts_with_re.py (or their intergenic equivalents) for a given set of chromosomes.
    Input files are variant and context count files produced by count_variants.py and count_contexts.py.

    Parameters are job number, root of input filename, list of input file suffixes separated by commas.

    Sample run statement is:
    nohup python3 ~/repos/Probpolymorphism/context/merge_chromosome_data.py mcd005
    var_dict_humanch 1b,3b,5b,7b,9b,11b,13b,15b,17b,19b,21b > mcd005.txt &"""


import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle, gzip
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('infile_root')
@click.argument('suffixes')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, infile_root, suffixes, dir):
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

    file_suffixes = suffixes.split(',')
    total_dict = defaultdict(int)
    counts = list()
    for c in file_suffixes:
        filename = dir + '/' + infile_root + c + '.pklz'
        infile = open(filename, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        with gzip.open(filename, 'rb') as chrdict:
            chrdict = pickle.load(chrdict)
        counts.append([c, sum(chrdict.values())])
        for k in chrdict.keys():
            total_dict[k] += chrdict[k]
    counts= pd.DataFrame.from_records(counts)
    outfile_name = dir + '/' + 'merged_context_data_' + job_no +  '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(total_dict, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    fname = dir + '/' + 'merge_counts_' + job_no +  '.csv'
    counts.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()


    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))

if __name__ == "__main__":
    main()
