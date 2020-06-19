# coding=utf-8

""" merge_male_and_female_recombination_rates.py takes a table of variant data counted against 10kb
    bins by sample_ensembl_for_recombination.py and adds columns for male and female recombination
    rates. Only required if it is desired to analyse male and female recombination rates separately.

    Sample run command is:
    nohup python3 /Users/helmutsimon/repos/MutationProbVariance/recombination/merge_male_and_female_recombination_rates.py
    merg001 /Users/helmutsimon/Google Drive/Genetics/Data sets/Decode recombination maps/addendum
    recomb_table_SW_sex-averaged_ch > merg_001.txt &"""

import numpy as np
import pandas as pd
import os
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('folder', type=click.Path())
@click.argument('fname', type=click.Path())
@click.option('-d', '--dir', default='Recombination_data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, folder, fname, dir):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + job_no + ".log"    # change
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__, label="Imported module ")
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__, label="Imported module ")

    csv_filename = '/male_noncarrier.rmap'
    male = pd.read_csv(folder + csv_filename, sep='\t')
    male = male.sort_values(['chr', 'pos'])

    csv_filename = '/female_noncarrier.rmap'
    female = pd.read_csv(folder + csv_filename, sep='\t')
    female = female.sort_values(['chr', 'pos'])

    for chrom in range(1, 23):
        chrom = str(chrom)
        m = male[male['chr'] == 'chr' + chrom]
        f = female[female['chr'] == 'chr' + chrom]
        csv_name = dir + '/' + fname + chrom + '.csv'
        infile = open(csv_name, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        table = pd.read_csv(csv_name, sep=',', index_col=0)
        #Note we are actually matching on hg37 coordinates used by deCODE
        xtable = table.merge(m, 'left', ['chr', 'pos'], sort=True, suffixes=['_sexav', '_male'], \
                             indicator='indicm', validate='1:1')
        xtable.rename(columns={"stdrate_sexav": "stdrate", "seqbin_sexav": "seqbin"}, inplace=True)
        xtable = xtable.merge(f, 'left', ['chr', 'pos'], sort=True, suffixes=['_sexav', '_female'], \
                              indicator='indicf', validate='1:1')
        assert np.all(xtable['indicm'] == 'both'), 'Merge error with male.'
        assert np.all(xtable['indicf'] == 'both'), 'Merge error with female.'
        csv_filename = 'Recombination_data/recomb_table_all_sexes_ch' + chrom + '.csv'
        xtable.to_csv(csv_filename, sep=',')
        outfile = open(csv_filename, 'r')
        LOGGER.output_file(outfile.name)
        outfile.close()
        print(xtable.head())

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()