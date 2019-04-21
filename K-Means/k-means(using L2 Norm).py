# -*- coding: utf-8 -*-
"""


@author: Vishal Dhiman
"""

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

plt.style.use('ggplot')
fig=plt.figure()
def k_mean(_data):
    k=int(input("\nHow many clusters? :"))
    alpha=float(input("\nEnter learning rate:"))
    n=_data.shape[0]
    c=_data.shape[1]

    _mean=np.mean(_data,axis=0)         
    _std=np.std(_data,axis=0)
    r_center=np.random.randn(k,c)*_std+_mean  #random center to cover whole data
   
    #plt.scatter(_data[:,0],_data[:,1],s=5,color='black')
    
    #plt.scatter(r_center[:,0],r_center[:,1],marker='+',s=25,c='red')
    
    prev_cent=np.zeros(r_center.shape)  # array with all values 0 and size = num of center
    new_cent=deepcopy(r_center)  

    _cluster=np.zeros(n)
    _dist=np.zeros((n,k))
    
    _count=0    
    _err=np.linalg.norm(new_cent-prev_cent)
    
    while _err>=alpha:
        for i in range(k):
            _dist[:,i]=np.linalg.norm(_data-new_cent[i],axis=1)        #distance of every point from every center
        _cluster=np.argmin(_dist,axis=1)

        prev_cent=deepcopy(new_cent)
        
        for i in range(k):
            new_cent[i]=np.mean(_data[_cluster==i],axis=0)
        _err=np.linalg.norm(new_cent-prev_cent)
        plt.scatter(_data[:,0],_data[:,1],s=5)
        plt.scatter(new_cent[:,0],new_cent[:,1],marker='+',s=60)
        plt.show()
        _count+=1
    print("final centers: \n",new_cent)
    print("no of iterations: ",_count)         


def gen_ran_data():
    print("\nWant to generate random data then give details:")
    n=int(input("\nHow many centers?")) 
    _center_list=[]
    for i in range(n):
        mes="\n"+str(i+1)+" center:"
        x,y=map(int,input(mes).split(','))
        _center_list.append([x,y])
    print(_center_list)

    #data generation around different centers
    _data=[]
    for i in range(n):
        _data.append(np.array(_center_list[i])+np.random.randn(200,2))
    _data=np.array(_data)
    
    _data=_data.reshape(-1,2)
    print(_data)
    k_mean(_data)

gen_ran_data()

