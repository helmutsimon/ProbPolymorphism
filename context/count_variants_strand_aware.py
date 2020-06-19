# coding=utf-8

""" count_variants_strand_aware.py processes file of intronic variants produced by
    sample_ensembl.py and generates a Counter object whose keys contain 9-mer context as well as
    ancestral and derived allele for variants. All SNVs were oriented with respect to the
    annotated strand of the gene.
    Parameters are job number and input filenames.
    Ensembl release and number of parallel joblib jobs are options.
    Sample run statement is:
    nohup python3 ~/helmutsimonpython/mutationprobvariance/probpoly/count_variants.py
    ch7b -v data/var_locations_humanch7a.pklz -i data/all_locations_humanch7a.pklz human -j 6
    >o.txt &"""


import os
import numpy
import numpy as np
from time import time
import gzip
import pickle
from operator import itemgetter
import bisect
import cogent3
from cogent3 import DNA
import ensembldb3
from ensembldb3 import HostAccount, Genome
from collections import Counter
from joblib import Parallel, delayed
import click
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)


def check_variant_strand(var_details, intron_locs):
    var_details.sort(key=itemgetter(2))
    count, reverse = 0, 0
    var_locs_reversed = list()
    var_locs = [v[2] for v in var_details]
    for intron in intron_locs:
        count += 1
        if intron[3] == 1:
            continue                   #intron is on forward strand
        reverse += 1
        intron_start = intron[1]
        intron_end = intron[2]
        if intron_end < var_locs[0]:
            continue
        a = bisect.bisect(var_locs, intron_start)
        b = bisect.bisect(var_locs, intron_end)
        for i in np.arange(a, b):
            var_details[i] = (var_details[i][0], var_details[i][1], var_details[i][2], \
                              DNA.complement(var_details[i][3]), DNA.complement(var_details[i][4]))
            var_locs_reversed.append(i)
    print('Number of introns processed'     , count)
    print('Number of reverse strand introns', reverse)
    print('Number of variants           ', len(var_details))
    print('Number of variants on (-) strand', len(var_locs_reversed))
    return var_details, var_locs_reversed

def get_contexts(var, coord_dict):
    if var[1] in coord_dict.keys():
        region = genome.get_region(coord_name=coord_dict[var[1]], start=var[2] - 4,
                              end=var[2] + 4, ensembl_coord=True)
        item = str(region.seq)
        #item = item[0:3] + var[3] + item[4:7] + var[4]  Only gets nhood for 7mers. Region start & end also changed.
        item = item[0:4] + var[3] + item[5:9] + var[4]
    else:
        item = None
    return item



def main_core(job_no, species, varfile_name=None, intronfile_name=None, release=89, n_jobs=5, dir='data'):
    global genome
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    start_time = time()
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(25))
    LOGGER.log_message('Name = ' + numpy.__name__ + ', version = ' + numpy.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + cogent3.__name__ + ', version = ' + cogent3.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + ensembldb3.__name__ + ', version = ' + ensembldb3.__version__,
                       label="Imported module".ljust(25))
    account = HostAccount(*os.environ['ENSEMBL_ACCOUNT'].split())
    genome = Genome(species, release=release, account=account, pool_recycle=3600)
    human_seq_region_dict = dict(
        {'1': 131550, '2': 131545, '3': 131551, '4': 131552, '5': 131542, '6': 131555, '7': 131559,
         '8': 131560, '9': 131540, '10': 131544, '11': 131556, '12': 131546, '13': 131541,
         '14': 131547, '15': 131558,
         '16': 131549, '17': 131554, '18': 131548, '19': 131537, '20': 131538, '21': 131543,
         '22': 131557,
         'X': 131539, 'Y': 131553})
    chimp_seq_region_dict = dict({"21": 212405, "7": 212407, "15": 212409, "16": 212395, "1": 212403, "17": 212411,
                                  "18": 212410, "19": 212394, "20": 212404, "22": 212390, "3": 212392, "4": 212393,
                                  "5": 212391, "6": 212388, "8": 212397, "9": 212396, "10": 212387, "11": 212389,
                                  "12": 212402, "13": 212408, "14": 212401, "Y": 212406, "X": 212399})
    if species == 'human':
        coord_dict = dict([(v, k) for k, v in human_seq_region_dict.items()])
        tag = 'human'
    elif species == 'chimp':
        coord_dict = dict([(v, k) for k, v in chimp_seq_region_dict.items()])
        tag = 'spec_'
    else:
        assert False, 'Unknown species: ' + species
    if varfile_name is None:
        varfile_name = dir + '/var_locations_' + tag + job_no + '.pklz'
    infile = open(varfile_name, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    with gzip.open(varfile_name, 'rb') as var_details:
        var_details = pickle.load(var_details)
    LOGGER.log_message(str(len(var_details)), label="Number of variants read".ljust(25))

    if intronfile_name is None:
        intronfile_name = dir + '/all_locations_' + tag + job_no + '.pklz'
    infile = open(intronfile_name, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    with gzip.open(intronfile_name, 'rb') as intron_locs:
        intron_locs = pickle.load(intron_locs)
    LOGGER.log_message(str(len(intron_locs)), label="Number of introns read".ljust(25))
    with gzip.open(intronfile_name, 'rb') as intron_locs:
        intron_locs = pickle.load(intron_locs)
    var_details, var_locs_reversed = check_variant_strand(var_details, intron_locs)

#   var_details fields are: (variant name, seq region id, location, ancestral_allele, derived_allele)
    item_list = Parallel(n_jobs=n_jobs)(delayed(get_contexts) (var, coord_dict) for var in var_details)
    var_count_dict = Counter(item_list)
    del var_count_dict[None]
    outfile_name = dir + '/var_dict_' + tag + job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(var_count_dict, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(25))


@click.command()
@click.argument('job_no')
@click.argument('species', type=click.Choice(['human', 'chimp']))
@click.option('-v', '--varfile_name', default=None, help='Variant details file.')
@click.option('-i', '--intronfile_name', default=None, help='Intron locations file.')
@click.option('-r', '--release', default=89, type=int, help='Ensembl release. Default is 89.')
@click.option('-j', '--n_jobs', default=10, type=int, help='Number of parallel jobs. Default is 10.')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, species, varfile_name, intronfile_name, release, n_jobs, dir):
    main_core(job_no, species, varfile_name, intronfile_name, release, n_jobs, dir)


if __name__ == "__main__":
    main()
