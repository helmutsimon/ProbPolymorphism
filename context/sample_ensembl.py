# coding=utf-8


""" sample_ensembl.py queries the Ensembl variant database and creates files of intronic variant
    details and intron lengths.

    Arguments are job no, chromosome, sequence start, sequence end and species. Sample run statement is:

    nohup python3 ~/helmutsimonpython/mutationprobvariance/probpoly/sample_ensembl.py ch1 1 1000000 2000000 chimp >o.txt &"""


import os
from time import time
import gzip
import pickle
import ensembldb3
from ensembldb3 import HostAccount, Genome
from fnmatch import fnmatch
import sqlalchemy
from sqlalchemy.sql import and_, not_, select
import decimal
import click
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)


def remove_clusters_of_snps(var_list):
    """Removes any elements of the variant list which are within 3 bp of another."""
    sorted_list = sorted(set(var_list), key=lambda x: x[2])
    discards = list()
    for item in sorted_list[:-1]:
        next_item = sorted_list[sorted_list.index(item) + 1]
        if item[2] >= next_item[2] - 3:
            discards.append(item)
            discards.append(next_item)
    clean_list = [x for x in sorted_list if x not in discards]
    return clean_list


def edit_human_variants(genome, aln_location):
    """Qualifies snp variants based on properties of the individual snp. Returns list of snps to be rejected."""
    variation_feature_table = genome.VarDb.get_table("variation_feature")
    whereclause1 = and_(variation_feature_table.c.seq_region_id == aln_location.seq_region_id,
                        variation_feature_table.c.seq_region_start > aln_location.start - 3,
                        variation_feature_table.c.seq_region_end < aln_location.end + 3)
    whereclause2 = and_(variation_feature_table.c.class_attrib_id == 2,
                        variation_feature_table.c.variation_name.contains("rs"),
                        variation_feature_table.c.variation_set_id.contains('42'),
                        variation_feature_table.c.somatic == 0,
                        variation_feature_table.c.alignment_quality == decimal.Decimal(1),
                        variation_feature_table.c.minor_allele_freq.isnot(None))
    whereclause2 = not_(whereclause2)
    whereclause3 = and_(whereclause1, whereclause2)
    query = select([variation_feature_table.c.variation_name], whereclause3)
    bad_vars = list(query.execute())
    bad_vars = [x[0] for x in bad_vars]
    return bad_vars


def get_variant_details(genome, species, intron, filter):
    """Get and edit variants from intron. Chimp variants are not edited because there are no retrieved variants that do not
    have class_attrib = 2, 'rs' in name or are somatic (chromosome 1). The other conditions are too restrictive. """
    variation_table = genome.VarDb.get_table('variation')
    variation_feature_table = genome.VarDb.get_table('variation_feature')
    snps = intron.variants
    if species == 'human':
        bad_variants = edit_human_variants(genome, intron.location)
        lbv = len(bad_variants)
    else:
        lbv = 0
    variations = list()
    for snp in snps:
        if species == 'human':
            if snp.symbol in bad_variants:
                continue
        ancestral_allele = str(snp.ancestral)
        name = str(snp.symbol)
        if ancestral_allele is None:
            continue
        whereclause1 = variation_table.c.name == name
        query = select([variation_table.c.variation_id], whereclause1)
        variation_id = list(query.execute())[0][0]
        whereclause2 = variation_feature_table.c.variation_id == variation_id
        query = select([variation_feature_table.c.allele_string,
                        variation_feature_table.c.seq_region_id,
                        variation_feature_table.c.seq_region_start], whereclause2)
        alleles = list(query.execute())[0][0]
        if fnmatch(alleles, ancestral_allele + '/?'):
            derived_allele = alleles[2]
        elif fnmatch(alleles, '?/' + ancestral_allele):
            derived_allele = alleles[0]
        else:
            continue
        var_details = (name, list(query.execute())[0][1], list(query.execute())[0][2],
                        ancestral_allele, derived_allele)
        variations.append(var_details)
    if filter:
        variations = remove_clusters_of_snps(variations)
    return variations, lbv


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
@click.argument('species', type=click.Choice(['human', 'chimp']))
@click.option('-r', '--release', default=89, type=int, help='Ensembl release. Default is 89.')
@click.option('-v', '--var_set_id', default=42, type=int, help='Variation set id. Default is 42: 1KG All.')
@click.option('--filter/--no-filter', default=True, help='Select whether to filter out closely clustered snps.')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, coord_name, start, end, species, release, var_set_id, filter, dir):
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

    var_locations_list, location_list = list(), list()
    dupl_introns, intron_count, bad_var_count, sequence_length = 0, 0, 0, 0
    intron_list, human_list, species_list, intron_list = list(), list(), list(), list()
    genome = Genome(species, release=release, account=account, pool_recycle=3600)
    confirm_variation_set(genome, var_set_id)
    genes = genome.get_features(coord_name=coord_name, start=start, end=end, feature_types='gene')
    for gene in genes:
        if gene.canonical_transcript.introns is None:
            continue
        for intron in gene.canonical_transcript.introns:
            if intron in intron_list:
                dupl_introns += 1
                continue
            intron_list.append(intron)
            intron_length = len(intron)
            intron_count += 1
            sequence_length += intron_length
            loc = intron.location
            location_list.append((str(loc.coord_name), loc.start, loc.end, loc.strand))  # location.coord_name is db3util object
            var_locations, bad_var_num = get_variant_details(genome, species, intron, filter)
            var_locations_list = var_locations_list + var_locations
            bad_var_count += bad_var_num
    LOGGER.log_message(str(dupl_introns), label='Number of duplicate introns rejected'.ljust(30))
    LOGGER.log_message(str(intron_count), label='Number of introns processed'.ljust(30))
    if species == 'human':
        LOGGER.log_message(str(bad_var_count), label='Number of rejected variants'.ljust(30))
    LOGGER.log_message(str(sequence_length), label='Sequence length'.ljust(30))
    LOGGER.log_message(str(len(var_locations_list)), label='Length of var_locations list'.ljust(30))
    LOGGER.log_message(str(len(var_locations_list) / sequence_length), label='Average SNV rate'.ljust(30))

    outfile_name = dir + '/var_locations_' + species + job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(var_locations_list, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    outfile_name = dir + '/all_locations_' + species + job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(location_list, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

