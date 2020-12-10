# coding=utf-8


""" Match table of DNMs created by tabulate_de_novo_mutations.py with Ensembl variant database.
    This script matches all variants at the site, regardless of derived allele.

    Arguments are job no, infile name, Ensembl release. Sample run statement is:

    nohup python3 ~/git/ProbPolymorphism/ProbPolymorphism/total_variance/match_snps_to_dnms.py 001
    dnms_from_PRJEB21300_with_ancestral > mstd001.txt &"""


import os
import pandas as pd
from time import time
import decimal
import ensembldb3
from ensembldb3 import HostAccount, Genome
import sqlalchemy
from sqlalchemy.sql import and_, select
import click
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('infile_name')
@click.option('-r', '--release', default=92, type=int, help='Ensembl release. Default is 89.')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, infile_name, release, dir):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + ensembldb3.__name__ + ', version = ' + ensembldb3.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + sqlalchemy.__name__ + ', version = ' + sqlalchemy.__version__,
                       label="Imported module".ljust(30))
    human_seq_region_dict = dict(
        {'1': 131550, '2': 131545, '3': 131551, '4': 131552, '5': 131542, '6': 131555, '7': 131559,
         '8': 131560, '9': 131540, '10': 131544, '11': 131556, '12': 131546, '13': 131541,
         '14': 131547, '15': 131558,
         '16': 131549, '17': 131554, '18': 131548, '19': 131537, '20': 131538, '21': 131543,
         '22': 131557,
         'X': 131539, 'Y': 131553})
    account = HostAccount(*os.environ['ENSEMBL_ACCOUNT'].split())
    genome = Genome('human', release=release, account=account, pool_recycle=3600)

    variation_feature_table = genome.VarDb.get_table('variation_feature')
    id_1KG = set([str(x) for x in range(42, 55)])
    var_details = pd.read_csv(infile_name, sep=',', index_col=0)
    infile = open(infile_name, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    loc_count, match_count, count1KG, derived_mismatch_count = 0, 0, 0, 0
    col_alleles, col_name, col_val_id = list(), list(), list()
    for row in var_details.iterrows():
        chrom = row[1].loc['chr']
        chrom = chrom[3:]
        seq_region_id = human_seq_region_dict[chrom]
        loc38 = row[1].loc['pos38']
        loc_count += 1
        whereclause1 = and_(variation_feature_table.c.seq_region_id == seq_region_id,
                            variation_feature_table.c.seq_region_start == loc38,
                            variation_feature_table.c.class_attrib_id == 2,
                            variation_feature_table.c.variation_name.contains("rs"),
                            variation_feature_table.c.somatic == 0,
                            variation_feature_table.c.alignment_quality == decimal.Decimal(1),
                            variation_feature_table.c.minor_allele_freq.isnot(None))
        query = select([variation_feature_table.c.variation_name,
                        variation_feature_table.c.allele_string,
                        variation_feature_table.c.variation_set_id], whereclause1)
        snps = list(query.execute())

        if len(snps) > 0:
            if len(snps) > 1:
                print('More than one SNP at ', chrom, ':', loc38)
            alleles = snps[0][1]
            name = snps[0][0]
            match_count += 1
            if len(set(snps[0][2]).intersection(id_1KG)) > 0:
                val_id = '1KG'
                count1KG += 1
            else:
                val_id = 'Other'
        else:
            val_id = 'No match'
            name = None
            alleles = None
        col_alleles.append(alleles)
        col_name.append(name)
        col_val_id.append(val_id)
    assert var_details.shape[0] == len(col_val_id), 'Column mismatch.'
    var_details['alleles'] = pd.Series(col_alleles)
    var_details['name'] = pd.Series(col_name)
    var_details['val_id'] = pd.Series(col_val_id)
    LOGGER.log_message(str(loc_count), label='Variants read      = ')
    LOGGER.log_message(str(derived_mismatch_count), label='Derived mismatches = ')
    LOGGER.log_message(str(match_count), label='Variants matched   = ')
    LOGGER.log_message(str(count1KG), label='1KG Variants       = ')
    filename = 'data/dnms_from_PRJEB21300_matched_' + job_no + '.csv'
    var_details.to_csv(filename)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

