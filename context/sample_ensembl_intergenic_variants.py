# coding=utf-8


""" sample_ensembl_intergenic_variants.py is analogous to sample_ensembl.py, but samples
    intergenic variants.

    Arguments are job no, chromosome, sequence start, sequence end and species. Sample run statement is:

    nohup python3 ~/helmutsimonpython/mutationprobvariance/probpoly/sample_ensembl_intergenic_variants.py
    ch1 1 1000000 2000000 >o.txt &"""


import os
from time import time
import gzip
import pickle
import ensembldb3
from ensembldb3 import HostAccount, Genome
from fnmatch import fnmatch
import sqlalchemy
from sqlalchemy.sql import and_, select
import decimal
import click
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)


def remove_clusters_of_snps(var_list):
    """Removes any elements of the variant list which are within 3 bp of another."""
    #sorted_list = sorted(set(var_list), key=lambda x: x[2])
    sorted_list = sorted(var_list, key=lambda x: x[2])
    discards = list()
    for item in sorted_list[:-1]:
        next_item = sorted_list[sorted_list.index(item) + 1]
        if item[2] >= next_item[2] - 3:
            discards.append(item)
            discards.append(next_item)
    clean_list = [x for x in sorted_list if x not in discards]
    return clean_list


def get_variant_details(genome, coord_name, start, end):
    """Get and edit variants from intron.  """
    human_seq_region_dict = dict(
        {'1': 131550, '2': 131545, '3': 131551, '4': 131552, '5': 131542, '6': 131555, '7': 131559,
         '8': 131560, '9': 131540, '10': 131544, '11': 131556, '12': 131546, '13': 131541,
         '14': 131547, '15': 131558,
         '16': 131549, '17': 131554, '18': 131548, '19': 131537, '20': 131538, '21': 131543,
         '22': 131557,
         'X': 131539, 'Y': 131553})
    seq_region_id = human_seq_region_dict[coord_name]
    variation_table = genome.VarDb.get_table('variation')
    variation_feature_table = genome.VarDb.get_table('variation_feature')
    var_table = variation_table.join(variation_feature_table,
                                     variation_feature_table.c.variation_id == variation_table.c.variation_id)
    whereclause4 = and_(var_table.c.variation_feature_seq_region_id == seq_region_id,
                        var_table.c.variation_feature_class_attrib_id == 2,
                        var_table.c.variation_feature_evidence_attribs.contains('370'),  # 1000 Genomes
                        var_table.c.variation_feature_variation_name.contains('rs'),
                        var_table.c.variation_feature_somatic == 0,
                        var_table.c.variation_feature_alignment_quality == decimal.Decimal(1),
                        var_table.c.variation_feature_minor_allele_freq.isnot(None),
                        var_table.c.variation_feature_seq_region_start > start,
                        var_table.c.variation_feature_seq_region_start < end,
                        var_table.c.variation_feature_consequence_types == {'intergenic_variant'})
    var_table_ed = var_table.select(whereclause4, use_labels=True)
    variations = list()
    for snp in var_table_ed.execute():
        if snp['variation_ancestral_allele'] is None:
            continue
        else:
            ancestral_allele = snp['variation_ancestral_allele']
        name = snp['variation_name']
        alleles = snp['variation_feature_allele_string']
        if fnmatch(alleles, ancestral_allele + '/?'):
            derived_allele = alleles[2]
        elif fnmatch(alleles, '?/' + ancestral_allele):
            derived_allele = alleles[0]
        else:
            continue
        var_details = (name, seq_region_id, snp['variation_feature_seq_region_start'], ancestral_allele, derived_allele)
        variations.append(var_details)
    variations = remove_clusters_of_snps(variations)
    return variations


def confirm_variation_set(genome, var_set_id):
    variation_set_table = genome.VarDb.get_table("variation_set")
    whereclause = and_(variation_set_table.c.variation_set_id == var_set_id)
    query = select([variation_set_table.c.name], whereclause)
    LOGGER.log_message(list(query.execute())[0][0], label='Variation set.'.ljust(30))


@click.command()
@click.argument('job_no')
@click.argument('coord_name')
@click.argument('start', type=int)
@click.argument('end', type=int)
@click.argument('species', type=click.Choice(['human', 'chimp']), default='human')
@click.option('-r', '--release', default=89, type=int, help='Ensembl release. Default is 89.')
@click.option('-v', '--var_set_id', default=42, type=int, help='Variation set id. Default is 42: 1KG All.')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, coord_name, start, end, species, release, var_set_id, dir):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    LOGGER.log_message('Name = ' + ensembldb3.__name__ + ', version = ' + ensembldb3.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + sqlalchemy.__name__ + ', version = ' + sqlalchemy.__version__,
                       label="Imported module".ljust(30))
    account = HostAccount(*os.environ['ENSEMBL_ACCOUNT'].split())

    genome = Genome(species, release=release, account=account, pool_recycle=3600)
    confirm_variation_set(genome, var_set_id)
    var_locations = get_variant_details(genome, coord_name, start, end)
    LOGGER.log_message(str(len(var_locations)), label='Length of var_locations list'.ljust(30))

    outfile_name = dir + '/intergenic_variants_' + species + '_' + job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(var_locations, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()