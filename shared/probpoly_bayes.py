# coding=utf-8


import numpy as np
import itertools
from statsmodels.stats.weightstats import DescrStatsW
import pandas as pd
from pymc3 import *


def unpack_var_contexts_to_dataframe(kmer_variable, counts: dict):
    """ This function uses the variant_counts produced by count_variants.py.
        This dictionary maintain counts of 9mers. For a given kmer value, this function
        creates pandas dataFrame context_all_counts counts, with rows indexed by the
        kmer contexts (16, 256 or 4096), with 12 columns given by allele transitions e.g. C to T. """
    contexts_generator = itertools.product('ACGT', repeat=2 * kmer_variable)
    contexts = tuple(''.join(context) for context in contexts_generator)
    mut_types = [i[0] + '->' + i[1] for i in itertools.permutations(['C', 'T', 'A', 'G'], 2)]
    context_counts = np.zeros((len(contexts), len(mut_types)))
    context_counts = pd.DataFrame(context_counts, index=contexts, columns=mut_types)
    lower = 4 - kmer_variable
    upper = 5 + kmer_variable
    for key in counts:
        context = key[lower: 4] + key[5: upper]
        mut_type = key[4] + '->' + key[9]
        if (context in contexts) and (mut_type in mut_types):
            context_counts.loc[context, mut_type] += counts[key]
    return context_counts

def unpack_all_contexts_to_dataframe(kmer_variable, counts: dict):
    """ This function uses the context_counts produced by count_contexts.py.
        This dictionary maintain counts of 9mers. For a given kmer value, this function
        creates pandas dataFrame context_all_counts counts, with rows indexed by the
        kmer contexts (16, 256 or 4096), with 12 columns given by allele transitions e.g. C to T. """
    contexts_generator = itertools.product('ACGT', repeat=2 * kmer_variable)
    contexts = tuple(''.join(context) for context in contexts_generator)
    mut_types = ['C', 'T', 'A', 'G']
    context_counts = np.zeros((len(contexts), len(mut_types)))
    context_counts = pd.DataFrame(context_counts, index=contexts, columns=mut_types)
    lower = 4 - kmer_variable
    upper = 5 + kmer_variable
    for key in counts:
        context = key[lower: 4] + key[5: upper]
        mut_type = key[4]
        if (context in contexts) and (mut_type in mut_types):
            context_counts.loc[context, mut_type] += counts[key]
    return context_counts

def reformat_context_counts(contexts, variants):
    """Reformat the context file by repeating the columns to match columns
    of the variants file."""
    context_counts = np.zeros_like(variants)
    cols = variants.columns
    context_counts = pd.DataFrame(context_counts, columns=cols)
    for col in context_counts.columns:
        fr = str(col)[0]
        context_counts.loc[:, col] = np.array(contexts.loc[:, fr])
    return context_counts, cols

def weighted_stat(x, w, stat):
    """Calculate weighted mean or variance of 1d array."""
    u = DescrStatsW(x, weights=w, ddof=0)
    if stat == 'mean':
        return u.mean
    else:
        return u.var

def weighted_stat_3D(arr, weights, stat):
    assert stat in ['var', 'mean'], 'Invalid stat parameter'
    varis = list()
    assert np.sum(weights) > 0, 'Weights total must be positive.'
    weight = weights / np.sum(weights)
    for i in range(arr.shape[2]):
        x = arr[:, :, i]
        w = weight[:, i]
        v = np.apply_along_axis(weighted_stat, 1, x, w, stat)
        varis.append(v)
    return np.array(varis).T

def weighted_stat_3D2(arr, weights, stat):
    assert stat in ['var', 'mean'], 'Invalid stat parameter'
    varis = list()
    assert np.sum(weights) > 0, 'Weights total must be positive.'
    weight = weights / np.sum(weights)
    for i in range(arr.shape[1]):
        x = arr[:, i, :]
        w = weight
        vs = list()
        for i in range(x.shape[1]):
            v = weighted_stat(x[:, i], w[:, i], stat)
            vs.append(v)
        varis.append(vs)
    return np.array(varis)

def calculate_variances(variants, contexts, split, prior, draws, ncols, columns):
    """Use Bayesian conjugate prior method to sample a mutation rates for contexts
    and thus their variance. The samples need to be split into blocks, analysed and reassembled
    for memory conservation purposes. In effect this employs a 'weighted' version of the
    Law of Total Variance."""
    obs = np.array(variants, dtype=int)
    obs_split = np.split(obs, split, axis=0)
    cc_split = np.split(contexts, split, axis=0)
    block_weights = [np.sum(w, axis=0) for w in cc_split]
    block_weights = np.array(block_weights)
    vrs, means = list(), list()
    for o, cc in zip(obs_split, cc_split):
        size = o.shape[0]
        print(o.shape, cc.shape)
        if np.any(cc - o < 0):
            print(o)
            print(cc)
        samples_part = np.random.beta(o + 1, cc - o + prior, (draws, size, ncols))
        v = weighted_stat_3D(samples_part, cc.values, 'var')
        m = weighted_stat_3D(samples_part, cc.values, 'mean')
        vrs.append(v)
        means.append(m)
        del samples_part
    vrs = np.array(vrs)
    means = np.array(means)
    evs = weighted_stat_3D2(vrs, block_weights, 'mean')
    ves = weighted_stat_3D2(means, block_weights, 'var')
    w_var_samples = evs + ves
    w_var_samples = w_var_samples.squeeze()
    w_var_samples = pd.DataFrame(w_var_samples, columns=columns)
    return w_var_samples