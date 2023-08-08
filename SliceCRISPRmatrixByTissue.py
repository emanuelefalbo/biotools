#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import argparse
import warnings
warnings.filterwarnings('ignore')

def range1(start, end):
    return range(start, end+1)

def cml_parser():
    parser = argparse.ArgumentParser('EnGene.py', formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('option_file', nargs="?", help="json option file")
    # parser.add_argument('filename', nargs="?",  help="input file")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input file CRISPR-Cas9 matrix', required=True)
    requiredNamed.add_argument('-ref', '--reference', help='Input reference file name', required=True)
    requiredNamed.add_argument('-t', '--tissue',default='all', help='Input tissue to be parsed; all cell lines are employed if == all', required=True)
    parser.add_argument('-m', default="impute", choices=["drop", "impute"], help="choose to drop Nan or impute")
    opts = parser.parse_args()
    if opts.input == None:
        raise FileNotFoundError('Missing CRISPR-Cas9 input file or None')
    elif opts.reference == None:
         raise FileNotFoundError('Missing reference input file or None')
    return  opts

def read_input(opts):
    sep = None
    if opts.input[-3:] == "tsv":
        sep == "\t"
        df_map = pd.read_csv(opts.input, sep=sep, engine="python", index_col=0)
        df_cl = pd.read_csv(opts.reference, sep=sep, engine="python")
    elif opts.input[-3:] == "csv":
        sep == ","
        df_map = pd.read_csv(opts.input, sep=sep, engine="python", index_col=0)
        df_cl = pd.read_csv(opts.reference, sep=sep, engine="python")
    elif opts.input[-3:] == "txt":
        df_map = pd.read_table(opts.input, engine="python", index_col=0)
    if df_map.shape[0] < df_map.shape[1]:  # Reverse order : (Rows x Colums) = (Gene x Cell Lines) 
        df_map = df_map.T
    return df_map, df_cl

def drop_na(df):
    print(f" If existing missing values: dropping NaN ...")
    columns_Nans = df.columns[df.isna().any()].to_list()
    if len(columns_Nans) != 0:
        df_dropped = df.dropna()
    else:
        df_dropped = df
    return df_dropped

def impute_data(df):
    # Imputing data by means of K-Nearest Neighbours algo
    print(f" if existing missing values: Imputing data ...")
    knn_imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean')
    df_knn = df.copy()
    columns_Nans = df_knn.columns[df_knn.isna().any()].to_list()
    if len(columns_Nans) != 0:
       df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)
       null_values = df_knn[columns_Nans[0]].isnull()
       fig = df_knn_imputed.plot(x=df_knn.columns[0], y=columns_Nans[0], kind='scatter', c=null_values, cmap='winter', 
                            title='KNN Imputation', colorbar=False, edgecolor='k', figsize=(10,8))
    else:
        df_knn_imputed = df
    return df_knn_imputed


def annote(df_map, df_cl, tissue):
    depmap_id = df_cl["depmapId"]
    lineage_1 = df_cl["lineage1"]
    lineage_1_unique = list(set(lineage_1))
    # print(lineage_1_unique)
    if tissue in lineage_1_unique:
        print(f' Selecting {tissue} from the DepMap full matrix ... ')
        id_tissue = lineage_1[lineage_1 == tissue].index
        name_cl = depmap_id[id_tissue].to_list()
        #Parse DepMap to select above cell lines 
        id_true = []
        count = 0
        for k, var in enumerate(df_map.columns):
            if var in name_cl:
                id_true.append(k)
                count +=1
        df_tissue = df_map.iloc[:, id_true]
        df_tissue = df_tissue.add_prefix(f'{tissue} ')
        
        return df_tissue
    else:   
        msg = ' \n '.join(lineage_1_unique)
        print(f' {tissue} not present in lineage1 \n')
        print(f' Select from the following list: \n')
        print(f' {msg} \n')
        sys.exit()
        
def main():
    opts = cml_parser()
    df_map, df_cl = read_input(opts)
    # Select no of Cell lines for specific tissue or keep'em all
    if opts.tissue == "all":
        df_tissue = df_map.copy()
    elif opts.tissue != 'all':
        df_tissue = annote(df_map, df_cl, opts.tissue) 
    # # Drop Nan or Impute data
    if opts.m == "drop":
        df_tissue = drop_na(df_tissue)
    elif opts.m == "impute":
        df_tissue = impute_data(df_tissue)
    df_tissue.to_csv(f"{opts.tissue}_CRISPR.csv")
        
if __name__ == "__main__":
    main()