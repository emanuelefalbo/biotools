#!/usr/bin/env python3

import numpy as np
from scipy.stats import describe
import pandas as pd
from scipy.stats import mode
from collections import Counter
import sys
from scipy import stats
import argparse

def cml_parser():
    parser = argparse.ArgumentParser('Otsu_module.py', formatter_class=argparse.RawDescriptionHelpFormatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input file CRISPR-Cas9 matrix', required=True)
    requiredNamed.add_argument('-ref', '--reference', help='Input reference file name', required=True)
    requiredNamed.add_argument('-t', '--tissue', nargs='+', default='all', help='list of tissue to be parsed; all cell lines are employed if == all', required=True)
    opts = parser.parse_args()
    if opts.input == None:
        raise FileNotFoundError('Missing CRISPR-Cas9 input file or None')
    elif opts.reference == None:
         raise FileNotFoundError('Missing reference input file or None')
    return  opts

def OtsuThresholding(A, N=1):    
    if N < 1 or N > 2:
        raise ValueError("N must be either 1 or 2.")
    
    num_bins = 256
    p, minA, maxA = getpdf(A, num_bins)
    assert len(p) > 0, "Cannot compute PDF."
    
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(num_bins))
    mu_t = mu[-1]

    sigma_b_squared = compute_sigma_b_squared(N, num_bins, omega, mu, mu_t)
    
    maxval = np.nanmax(sigma_b_squared)
    assert np.isfinite(maxval), "Cannot find a finite maximum for sigma_b_squared."
    
    if N == 1:
        idx = np.nanargmax(sigma_b_squared)
        #idx = np.where(sigma_b_squared == maxval)[0]
        thresh = np.mean(idx) - 1
    else:
        maxR, maxC = np.unravel_index(np.argmax(sigma_b_squared), sigma_b_squared.shape)
        thresh = np.mean([maxR, maxC]) - 1
    
    thresh = minA + thresh / 255 * (maxA - minA)
    metric = maxval / np.sum(p * ((np.arange(num_bins) - mu_t) ** 2))
    return thresh, metric

def getpdf(A, num_bins):
    A = A.ravel()  # Vectorize A for faster histogram computation

    # replace NaNs with empty values (both in float and integer matrix)
    A = A[np.isfinite(A)]
    # if A was full of only NaNs return a null distribution
    if A.size == 0:
        return np.array([]), np.nan, np.nan

    if np.issubdtype(A.dtype, np.floating):   # if a float A the scale to the range [0 1]
        # apply sclaing only to finite elements
        idx_isfinite = np.isfinite(A)
        if np.any(idx_isfinite):
            minA = np.min(A[idx_isfinite])
            maxA = np.max(A[idx_isfinite])
        else:  # ... A has only Infs and NaNs, return a null distribution
            minA = np.min(A)
            maxA = np.max(A)
        A = (A - minA) / (maxA - minA)
    else:  # if an integer A no need to scale
        minA = np.min(A)
        maxA = np.max(A)
    if minA == maxA:   # if therei no range, retunr null distribution
        return np.array([]), minA, maxA
    else:
        counts, _ = np.histogram(A, bins=num_bins, range=(0, 1))
        distrib = counts / np.sum(counts)
        return distrib, minA, maxA

def compute_sigma_b_squared(N, num_bins, omega, mu, mu_t):
    if N == 1:
        sigma_b_squared = np.ones( (len(omega)) ) * np.nan
        np.divide((mu_t * omega - mu)**2,(omega * (1 - omega)), out=sigma_b_squared, where=(omega * (1 - omega))!=0)
        #sigma_b_squared = (mu_t * omega - mu)**2 / (omega * (1 - omega))
    elif N == 2:
        # Rows represent thresh(1) (lower threshold) and columns represent
        # thresh(2) (higher threshold).
        omega0 = np.tile(omega, (num_bins, 1))
        mu_0_t = np.tile((mu_t - mu / omega), (num_bins, 1))
        omega1 = omega.reshape(num_bins, 1) - omega
        mu_1_t = mu_t - (mu - mu.T) / omega1
        
        # Set entries corresponding to non-viable solutions to NaN
        allPixR, allPixC = np.meshgrid(np.arange(num_bins), np.arange(num_bins))
        pixNaN = allPixR >= allPixC  # Enforce thresh(1) < thresh(2)
        omega0[pixNaN] = np.nan
        omega1[pixNaN] = np.nan
        
        term1 = omega0 * (mu_0_t ** 2)
        term2 = omega1 * (mu_1_t ** 2)
        omega2 = 1 - (omega0 + omega1)
        omega2[omega2 <= 0] = np.nan  # Avoid divide-by-zero Infs in term3
        term3 = ((omega0 * mu_0_t + omega1 * mu_1_t) ** 2) / omega2
        sigma_b_squared = term1 + term2 + term3
    
    return sigma_b_squared

def QuantizeByColumns(T, n):
    Q = np.zeros((T.shape[0], T.shape[1]), dtype=np.uint8)
    ValueForNaNs = np.nanmax(T) + 1
    threshVec = np.zeros(T.shape[1])
    
    for c in range(0, T.shape[1]):
        ColValues = T[:, c]
        ColTh, _ = OtsuThresholding(ColValues, n - 1)
        If1 = ColValues.copy()
        ismiss = np.isnan(ColValues)
        
        if np.any(ismiss):
            If1[ismiss] = ValueForNaNs
            print(f'Cell line {c} has {np.sum(ismiss)} NaNs')
        
        threshVec[c] = ColTh
        ConNans = np.digitize(If1, [np.nanmin(T), ColTh, np.nanmax(T)])
        ConNans[ismiss] = n + 1
        Q[:, c-1] = ConNans
        
    return Q, threshVec

def OtsuCore(df, name):     
# Extract data from the dataframe
    T = df.to_numpy()
    NumberOfClasses = 2
    
    print("Two-class labelling:")
    # Perform quantization
    Q2, Thr = QuantizeByColumns(T, NumberOfClasses)
    Q2 = Q2 - 1
    modeQ2 = stats.mode(Q2, axis=1, keepdims=False).mode
    
    dfout =  pd.DataFrame(index=df.index)
    dfout.index.name = 'name'
    dfout['label'] = modeQ2.ravel()
    dfout = dfout.replace({0: 'E', 1:'NE'})
    print(dfout.value_counts())
    
    print("Three-class labelling:")
    NE_genes = dfout[dfout['label']=='NE'].index
    TNE = df.loc[NE_genes].to_numpy()
    NumberOfClasses = 2
    QNE2, ThrNE = QuantizeByColumns(TNE, NumberOfClasses)
    QNE2 = QNE2 - 1
    modeQ2NE = stats.mode(QNE2, axis=1, keepdims=False).mode
    
    dfout2 =  pd.DataFrame(index=NE_genes)
    dfout2.index.name = 'name'
    dfout2['label'] = modeQ2NE.ravel()
    dfout2 = dfout2.replace({0: 'aE', 1:'sNE'})
    dfout.loc[dfout.index.isin(dfout2.index), ['label']] = dfout2[['label']]
    # Save the results to the output CSV file
    fout = 'otsu_label_' + f'{name}.csv'
    dfout.to_csv(fout)
    print(dfout.value_counts())

    return Q2

def OtsuTissues(df, df_cl, list_tissues):
    depmap_id = df_cl["depmapId"]
    lineage_1 = df_cl["lineage1"]
    lineage_1_unique = list(set(lineage_1))
    lineage_1_unique = [ var.replace('/', '_') for var in lineage_1_unique]
    sum_shape = 0
    Q2_tot = []
    # for tissue in lineage_1_unique:
    for tissue in  list_tissues:
        id_tissue = lineage_1[lineage_1 == tissue].index
        name_cl = depmap_id[id_tissue].tolist()
        id_true = []
        for k, var in enumerate(df.columns):
            if var in name_cl:
                id_true.append(k)
        df_tissue = df.iloc[:, id_true]
        sum_shape += df_tissue.shape[1]
        if df_tissue.shape[1] != 0:
            print(f' Processing {tissue}')
            Q2_tot.append(OtsuCore(df=df_tissue, name=tissue))
    # check if dimensions are correct
    # if sum_shape != df.shape[1]:
    #     print("The summed lenght of cell lines subsets is not equal to the original no of cell lines !")
    # print(df.shape, sum_shape)
    return Q2_tot

def main():
    opts = cml_parser()
    df_map = pd.read_csv(opts.input, index_col=0)
    if df_map.shape[0] < df_map.shape[1]:
        df_map = df_map.T
    df_cl = pd.read_csv(opts.reference)
    Q2_tot = OtsuTissues(df_map, df_cl, opts.tissue)
    print(len(Q2_tot))
    # Execute mode on each tissue and sort'em
    Q2Mode_tot = []
    for k in Q2_tot:
        Q2Mode_tot.append(stats.mode(k, axis=1, keepdims=False).mode)
    Q2Mode_tot = np.vstack(Q2Mode_tot).T
    
    # # Get mode of mode among tissues
    modeOfmode = stats.mode(Q2Mode_tot, axis=1, keepdims=False).mode
    dfout =  pd.DataFrame(index=df_map.index)
    dfout.index.name = 'name'
    dfout['label'] = modeOfmode.ravel()
    dfout = dfout.replace({0: 'E', 1:'NE'})
    dfout.to_csv("Otsu_2Mode_tissues.csv")
    print(dfout.value_counts())
    
    
if __name__ == "__main__":
    main()
