# coding=utf-8


""" sample_ensembl_intergenic_sites.py is analogous to count_intronic_sites.py, but extracts
    intergenic sites.

    Arguments are job no, chromosome, sequence start, sequence end and species. Sample run statement is:

    nohup python3 ~/helmutsimonpython/mutationprobvariance/probpoly/sample_ensembl_intergenic_sites.py ch1 1 1000000 2000000 >o.txt &"""


import os
from time import time
import gzip
import pickle
import ensembldb3
from ensembldb3 import HostAccount, Genome
import click
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)


def interval_complement(inpairs):
    """Input is a list of integer 2ples representing start and endpoints of intervals, ordered by endponts.
    Returns a complementary set of intervals (the gaps)."""
    outpairs = list()
    for i, x in enumerate(inpairs[:-1]):
        upper = inpairs[i + 1:]
        nextg = min(y[0] for y in upper)
        if nextg > x[1]:
            outpairs.append((x[1] + 1, nextg))
    return outpairs


@click.command()
@click.argument('job_no')
@click.argument('coord_name')
@click.argument('start', type=int)
@click.argument('end', type=int)
@click.argument('species', type=click.Choice(['human', 'chimp']), default='human')
@click.option('-r', '--release', default=89, type=int, help='Ensembl release. Default is 89.')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, coord_name, start, end, species, release, dir):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    LOGGER.log_message('Name = ' + ensembldb3.__name__ + ', version = ' + ensembldb3.__version__,
                       label="Imported module".ljust(30))
    account = HostAccount(*os.environ['ENSEMBL_ACCOUNT'].split())

    ig_count, sequence_length = 0, 0
    genome = Genome(species, release=release, account=account, pool_recycle=3600)
    gene_count = 0
    gene_intervals = list()
    genes = genome.get_features(coord_name=coord_name, start=start, end=end, feature_types='gene')
    for gene in genes:
        if gene.location.coord_name != coord_name:
            break
        gene_count += 1
        gene_intervals.append((gene.location.start, gene.location.end))
    gene_intervals = sorted(gene_intervals, key=lambda x: x[1])
    intergenic = interval_complement(gene_intervals)
    intergenic_sequence = ""
    for ig_interval in intergenic:
        ig_count += 1
        sequence_length += ig_interval[1] - ig_interval[0]
        region = genome.get_region(coord_name=coord_name, start=ig_interval[0], end=ig_interval[1])
        intergenic_sequence = intergenic_sequence + 'XXXXXXXXXX' + str(region.seq)

    LOGGER.log_message(str(ig_count), label='Number of integenic intervals processed'.ljust(30))
    LOGGER.log_message(str(sequence_length), label='Sequence length'.ljust(30))

    outfile_name = dir + '/intergenic_sequence_' + species + job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(intergenic_sequence, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

