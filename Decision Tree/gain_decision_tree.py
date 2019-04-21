# -*- coding: utf-8 -*-
"""
@author: Vishal Dhiman
"""
import pandas as pd
import numpy as np
import math as mt

def Entropy(col):
    l=np.asarray(col).size
    dist_val,freq=np.unique(col,return_counts=True)
    dist_count=np.asarray(dist_val)
    _sum=0
    for i in range(dist_count):
        r=freq[i]/l
        ent=-1*r*mt.log(r,2)
        _sum+=ent
    return _sum

def gain(col,tag_col):
    I_tab=Entropy(tag_col)
    tt_count=np.asarray(col).size
    dist_term=np.unique(col)
    tm_count=np.array(dist_term).size
    info=0
    for i in range(tm_count):
        _listt=[]
        
        for j in range(tt_count):
            if(col[j]==dist_term[i]):
                _list.append(tag_col[j])
        e=Entropy(_list)
        w=np.asaray(_list).size/tt_count
        info+=e*w
    gain=I_tab-info
    return gain



f_name=input("\nEnter the file name:")

data_f=pd.read_csv(f_name)

tag_name=input("\nEnter the target column name:")

tag_col=data_f[tag_name]

col_list=list(data_f.columns)

col_list.remove(tag_name)

gain(data_f[col_list[i]],tag_col)
