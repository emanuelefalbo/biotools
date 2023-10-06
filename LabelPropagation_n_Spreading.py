#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore') #we don't wanna see that
np.random.seed(1) #i'm locking seed at the begining since we will use some heavy RNG stuff, be aware


def cml_parser():
    parser = argparse.ArgumentParser('EnGene.py', formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('option_file', nargs="?", help="json option file")
    # parser.add_argument('filename', nargs="?",  help="input file")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input file CRISPR-Cas9 matrix', required=True)
    requiredNamed.add_argument('-eg', '--eg', help='File containing Essential Gene labels ', required=True)
    requiredNamed.add_argument('-neg', '--neg', help='File containing Not-Essential Genes labels ', required=True)
    parser.add_argument('-o', '--output', default="label_output", help="Name of output containing predicted labels")
    parser.add_argument('-m', default="spreading", choices=["propagation", "spreading"], help="Label Spreading or Propagation algorithm")
    opts = parser.parse_args()
   # print(opts.input)
    # if opts.input == None:
    #    raise FileNotFoundError(f'Missing {opts.input} input file or None')
    return  opts

def read_input(opts):
    sep = None
    if opts.input[-3:] == "tsv":
        sep == "\t"
        df_map = pd.read_csv(opts.input, sep=sep, engine="python", index_col=0)
    elif opts.input[-3:] == "csv":
        sep == ","
        df_map = pd.read_csv(opts.input, sep=sep, engine="python", index_col=0)
    # Check correct shape n_samples > m_features: Reverse order : (Rows x Colums) = (Gene x Cell Lines) 
    if df_map.shape[0] < df_map.shape[1]: 
        df_map = df_map.T
    # Delete ( if any in Gene Names:
    index = df_map.index.tolist()
    idx = [ i.split('(', 1)[0].strip() for i in index]
    df_map.reset_index(drop=True)
    df_mod = pd.DataFrame(df_map.to_numpy(), index=idx, columns=df_map.columns)    
        
    return df_mod

def read_labels(fname_EG, fname_NEG):
    EG_list = pd.read_csv(fname_EG)
    NEG_list = pd.read_csv(fname_NEG)
    EG_list['Label'] = 1
    NEG_list['Label'] = 0
    
    return EG_list, NEG_list

def build_data(df_map, EG_list, NEG_list):
    y_all = pd.concat((EG_list, NEG_list), ignore_index=True)
    y_all.set_index('Gene', inplace=True)
    # Finding common genes with DepMap input
    df_idx = df_map.index.to_numpy()
    df_idx = np.asarray([ i.strip() for i in df_idx])
    common_EG = list(np.intersect1d(EG_list.iloc[:,0].to_numpy(), df_idx))
    common_NEG = list(np.intersect1d(NEG_list.iloc[:,0].to_numpy(), df_idx))
    # Building Labeled and Unlabeled data
    X_labeled = df_map.loc[common_EG+common_NEG]#.to_numpy()
    X_unlabeled = df_map.drop(common_EG+common_NEG)#.to_numpy()
    y_labeled = y_all.loc[common_EG+common_NEG]#.to_numpy().flatten()
    #Checking shapes:
    if X_labeled.shape[0]+X_unlabeled.shape[0] != df_map.shape[0]:
        print(f"Shape mismatch ")
        
    return X_labeled, X_unlabeled, y_labeled

def SSL_Labels(method, X_labeled, X_unlabeled, y_labeled):
    if method == 'propagation':
        model = LabelPropagation('knn', max_iter=100000, tol=0.0001)
    elif method == 'spreading':
        model = LabelSpreading('knn', max_iter=1000, tol=0.001)
    print(f'Executing {method} ...')
    n_folds=5
    #Create a cross-validation iterator
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    progress_bar = tqdm(total=n_folds, desc="Cross-Validation Progress")
    #Lists to store performance metrics from each fold
    roc_scores = []
    acc_scores = []
    ba_scores = []
    mcc_scores = []
    #Cross-validation loop
    for _, (train_index, test_index) in enumerate(cv.split(X_labeled, y_labeled)):
        # Create training and testing sets for this fold
        X_train = X_labeled.iloc[train_index]
        y_train = y_labeled.iloc[train_index]
        X_test = X_labeled.iloc[test_index]
        y_test = y_labeled.iloc[test_index]
        #   Combine a portion of the unlabeled data with the labeled data for training
        X_train_combined = np.vstack((X_train, X_unlabeled))
        y_train_combined = np.concatenate((y_train.iloc[:,0], [-1] * len(X_unlabeled)))  #-1 as a placeholder for unlabeled data
        model.fit(X_train_combined, y_train_combined)
        y_pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred)) # accuracy_score(y_test, y_pred_lp)])
        ba_scores.append(balanced_accuracy_score(y_test, y_pred))
        mcc_scores.append(matthews_corrcoef(y_test, y_pred))
        roc_scores.append(roc_auc_score(y_test, y_pred))
        # Update the progress bar
        progress_bar.update(1)
    progress_bar.close()
    scores = pd.DataFrame(np.vstack((acc_scores, ba_scores, roc_scores, mcc_scores)).T, columns=['ACC', 'BA', 'ROC', 'MCC'])
    print(f'Examining Scores Summary:')
    print(f'{scores.describe()}')
        
    return model, scores

def main():
    opts = cml_parser()
    df_map = read_input(opts)
    EG_ref, NEG_ref = read_labels(opts.eg, opts.neg)
    X_labeled, X_unlabeled, y_labeled = build_data(df_map, EG_ref, NEG_ref)
    #Perform Semi-Supervised Label Propagation o Spreading 
    # with a 5-fold Stratified Cross-Validation 
    model, scores = SSL_Labels(opts.m, X_labeled, X_unlabeled, y_labeled)
    scores.to_csv('scores.csv')
    # Predicting labels on X_unlabeled data
    y_pred = model.predict(X_unlabeled)
    dfout = pd.DataFrame({'Gene':X_unlabeled.index.tolist(),'Label':y_pred})
    dfout.set_index('Gene', inplace=True)
    dfout.to_csv(f'{opts.output}.csv')
    
    


if __name__ == "__main__":
    main()
