# coding=utf-8


""" sample_ensembl_for_recombination.py samples Ensembl and adds variant counts to the 10 kb bins used
    in deCODE recombination maps, after coordinates are mapped by
    Map deCODE genetic map coordinates.ipynb.

    Arguments are job no, chromosome, sequence start, sequence end and species. Sample run statement is:

    nohup python3 ~/helmutsimonpython/mutationprobvariance/recombination/sample_ensembl_for_recombination.py ch1001 1  >o.txt &"""


import os
from time import time
import numpy as np
import pandas as pd
from itertools import permutations
from collections import Counter
import ensembldb3
from ensembldb3 import HostAccount, Genome
from fnmatch import fnmatch
import sqlalchemy
from sqlalchemy.sql import and_, not_, select
import decimal
import click
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)

@click.command()
@click.argument('job_no')
@click.argument('chrom')
@click.argument('sex', default='sex-averaged')
@click.argument('species', type=click.Choice(['human', 'chimp']), default='human')
@click.option('-r', '--release', default=92, type=int, help='Ensembl release. Default is 89.')
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, chrom, sex, species, release, dir):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
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
    genome = Genome(species, release=release, account=account, pool_recycle=3600)
    variation_table = genome.VarDb.get_table('variation')
    variation_feature_table = genome.VarDb.get_table('variation_feature')
    var_table = variation_table.join(variation_feature_table,
                                     variation_feature_table.c.variation_id == variation_table.c.variation_id)

    seq_region_id = human_seq_region_dict[chrom]

    file_name = sex + '_noncarrier-hg38.csv'
    infile = open(file_name, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    recombination_df = pd.read_csv(file_name, usecols=[0, 1, 2, 3, 4])
    recomb_df = recombination_df.loc[lambda df: df.chr == 'chr' + chrom, :]
    recomb_df = recomb_df.reset_index(drop=True)

    mut_profiles = [i[0] + '->' + i[1] for i in permutations(['C', 'T', 'A', 'G'], 2)]
    counts = np.zeros((recomb_df.shape[0], 21))
    counts = pd.DataFrame(counts, columns=mut_profiles + ['C', 'T', 'A', 'G', 'SW', 'WS', 'SS', 'WW', 'CpG'])
    for index, row in recomb_df.iterrows():
        midpoint = row.loc['pos38']
        region = genome.get_region(coord_name=chrom, start=midpoint - 5000,
                                   end=midpoint + 5000, ensembl_coord=True)
        region = str(region.seq)
        whereclause1 = and_(var_table.c.variation_feature_seq_region_id == seq_region_id,
                            var_table.c.variation_feature_class_attrib_id == 2,
                            var_table.c.variation_feature_evidence_attribs.contains('370'),
                            var_table.c.variation_feature_variation_name.contains('rs'),
                            var_table.c.variation_feature_somatic == 0,
                            var_table.c.variation_feature_alignment_quality == decimal.Decimal(1),
                            var_table.c.variation_feature_minor_allele_freq.isnot(None),
                            var_table.c.variation_feature_seq_region_start > midpoint - 5000,
                            var_table.c.variation_feature_seq_region_start < midpoint + 5000)
        var_table_ed = var_table.select(whereclause1, use_labels=True)

        for snp in var_table_ed.execute():
            if snp['variation_ancestral_allele'] is None:
                continue
            else:
                ancestral_allele = snp['variation_ancestral_allele']
            alleles = snp['variation_feature_allele_string']
            if fnmatch(alleles, ancestral_allele + '/?'):
                derived_allele = alleles[2]
            elif fnmatch(alleles, '?/' + ancestral_allele):
                derived_allele = alleles[0]
            else:
                continue
            mtype = ancestral_allele + '->' + derived_allele
            counts.loc[index, mtype] += 1

            rel_loc = snp['variation_feature_seq_region_start'] - midpoint + 5000
            if (region[rel_loc + 1] == 'G' and ancestral_allele == 'C' and derived_allele == 'T') or \
                    (region[rel_loc - 1] == 'C' and ancestral_allele == 'G' and derived_allele == 'A'):
                counts.loc[index, 'CpG'] += 1
            if ancestral_allele + derived_allele in ['CT', 'CA', 'GT', 'GA']:
                counts.loc[index, 'SW'] += 1
            if ancestral_allele + derived_allele in ['TC', 'AC', 'TG', 'AG']:
                counts.loc[index, 'WS'] += 1
            if ancestral_allele + derived_allele in ['CG', 'GC']:
                counts.loc[index, 'SS'] += 1
            if ancestral_allele + derived_allele in ['TA', 'AT']:
                counts.loc[index, 'WW'] += 1
        base_counts = Counter(region)
        for base in ['C', 'T', 'A', 'G']:
            counts.loc[index, base] = base_counts[base]


    results = pd.concat([recomb_df, counts], axis=1)
    csv_filename = 'recomb_table_SW_' + sex + '_ch' + chrom + '.csv'
    results.to_csv(csv_filename)
    outfile = open(csv_filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="run duration (minutes)".ljust(30))

if __name__ == "__main__":
    main()

