# -*- coding: utf-8 -*-
"""
@author: saurabh verma
"""

import scipy.io
import pandas as pd
import numpy as np


################# Load Binary Feature Matrix in Sparse COmpressed Row Format #######################

mat = scipy.io.loadmat('data/nell_exp_X_transpose.mat')
X=mat['X']
X=X.transpose()

################# Load All Set Elements and Given Set Seed Elements #######################

df_set=pd.read_csv('data/all_set_elements.csv',sep='<#>',header=None)
df_set.columns=['Set_Element']

df_class_seed_set=pd.read_csv('data/class_seed_set_01.csv',header=None)
df_class_seed_set.columns=['Set_Element']

################# Bayesian Set Code #######################

query_idx=df_set.loc[df_set['Set_Element'].isin(df_class_seed_set['Set_Element'].tolist())].index.tolist()
query_idx=np.array(query_idx)

scf=2; #tunable parameter

XM=np.mean(X,0)+1e-12;
alphap=scf*XM; 
betap=scf*(1-XM);
lal=np.log(alphap);
lbe=np.log(betap);
lab=np.log(alphap+betap);

N=len(query_idx);
labn=np.log(alphap+betap+N);

query_feature_matrix=X[query_idx,:];         #pull out the query set feature vectors from the dataset

query_feature_sum=np.sum(query_feature_matrix,0);
lbp=np.log(betap+N-query_feature_sum);
class_bias=np.sum(lab-labn+lbp-lbe,1);
w_class_vec=np.log(alphap+query_feature_sum)-lal-lbp+lbe;
score= np.sum(X.multiply(w_class_vec),1)
pscore=score+class_bias; 

################# Get Highest Rank Elements #######################

pscore=np.array(pscore).reshape((pscore.shape[0],))
idx=pscore.argsort()[::-1][:100] #Get first '100' highest rank set elements 

print(df_set.iloc[idx])

