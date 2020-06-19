# coding=utf-8


""" count_intronic_sites.py scans the Ensembl database and creates a string which consists of the
    chained sequences of canonical introns. The introns are separated by 'XXXXXXXXXX' so that no
    spurious contexts across introns are found.

    Arguments are job no, chromosome, sequence start, sequence end and species. Sample run statement is:

    nohup python3 ~/helmutsimonpython/mutationprobvariance/probpoly/count_intronic_sites.py
    ch1 1 1000000 248000000 >o.txt &"""


import os
from time import time
import gzip
import pickle
import ensembldb3
from ensembldb3 import HostAccount, Genome
import click
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('coord_name')
@click.argument('start', type=int)
@click.argument('end', type=int)
@click.argument('species', type=click.Choice(['human', 'chimp']), default='human')
@click.option('-r', '--release', default=89, type=int, help='Ensembl release. Default is 89.')
@click.option('-d', '--folder', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, coord_name, start, end, species, release, folder):
    start_time = time()
    if not os.path.exists(folder):
        os.makedirs(folder)
    LOGGER.log_file_path = folder + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    LOGGER.log_message('Name = ' + ensembldb3.__name__ + ', version = ' + ensembldb3.__version__,
                       label="Imported module".ljust(30))
    account = HostAccount(*os.environ['ENSEMBL_ACCOUNT'].split())

    dupl_introns, intron_count, sequence_length = 0, 0, 0
    intron_list, human_list, species_list, intron_list = list(), list(), list(), list()
    genome = Genome(species, release=release, account=account, pool_recycle=3600)
    genes = genome.get_features(coord_name=coord_name, start=start, end=end, feature_types='gene')
    intron_sequence = 'X'
    for gene in genes:
        if gene.canonical_transcript.introns is None:
            continue
        for intron in gene.canonical_transcript.introns:
            if intron in intron_list:
                dupl_introns += 1
                continue
            intron_list.append(intron)
            intron_count += 1
            sequence_length += len(intron)
            intron_sequence = intron_sequence + 'XXXXXXXXXX' + str(intron.seq)
    outfile_name = folder + '/intronic_sequence' + species + job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(intron_sequence, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    LOGGER.log_message(str(dupl_introns), label='Number of duplicate introns rejected'.ljust(30))
    LOGGER.log_message(str(intron_count), label='Number of introns processed'.ljust(30))
    LOGGER.log_message(str(sequence_length), label='Total intron_length'.ljust(30))

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

