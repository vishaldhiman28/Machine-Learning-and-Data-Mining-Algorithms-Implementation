# -*- coding: utf-8 -*-
"""


@author: Vishal Dhiman
"""
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

plt.style.use('ggplot')
fig=plt.figure()
def dbscan(D,eps,minpts):
    _objects=[0]*len(D)  #initially all points are unvisited i.e. object=0
    
    c_id=0 #current cluster
    
    for i in range(0,len(D)):
        
        if _objects[i]==0 :
            
            eps_neighbor_pts=find_neighbor_pts(i,D,eps)
             
            if len(eps_neighbor_pts)>minpts:
                c_id+=1
                _objects[i]=c_id
                _objects=cluster(D,c_id,eps_neighbor_pts,_objects,eps,minpts,i)
                
            else:
                _objects[i]=-1 #mark object=-1 i.e. noise if minpts are not in eps
              
    return _objects,c_id
 
def cluster(D,c_id,neighbor_pts,_objects,eps,minpts,i_obj):
    i=0
    while i<len(neighbor_pts):
        nth_obj=neighbor_pts[i]
        
        if _objects[nth_obj]==0:
            _objects[nth_obj]=c_id
            nth_obj_neighbour=find_neighbor_pts(nth_obj,D,eps)
            
            if len(nth_obj_neighbour)>=minpts:
                neighbor_pts=neighbor_pts+nth_obj_neighbour
        elif _objects[nth_obj]==-1:
            _objects[nth_obj]=c_id
        i+=1
    return _objects        

def find_neighbor_pts(i_obj,D,eps):
    neighbor_pts=[]
    # distance of ith object from all other object
    for i in range(0,len(D)):
        if np.linalg.norm(D[i_obj]-D[i])<eps:  
            neighbor_pts.append(i)
    return neighbor_pts 

def plot_data(D,obj_lab,n):
    d_count=0
    for i in range(0,n):
        i_cluster_data=[]
        for j in range(0,len(D)):
            if(obj_lab[j]==i+1):
                i_cluster_data.append(D[j])
            
        i_cluster_data=np.array(i_cluster_data)
        d_count+=len(i_cluster_data)        
        #print("\n:::",i_cluster_data)
        #print("Data points for cluster ",i)
        
        plt.scatter(i_cluster_data[:,0],i_cluster_data[:,1],s=5)
    print("Number of total points after clustering:  ",d_count)
    print("Number of Noise points: ",len(D)-d_count)     
        
def gen_ran_data():
    print("\nWant to generate gaussian distribution data for testing then give details:")
    n=int(input("\nHow many centers? ")) 
    _center_list=[]
    for i in range(n):
        mes="\n"+str(i+1)+" center:"
        x,y=map(int,input(mes).split(','))
        _center_list.append([x,y])
    print(_center_list)        
    count_samp=int(input("\n How many samples?: "))
    std_samp=float(input("How much standard deviation in the data? "))
    Data, orig_labels = make_blobs(n_samples=count_samp, centers=_center_list, cluster_std=std_samp,random_state=0)

    Data = StandardScaler().fit_transform(Data)
    plt.scatter(Data[:,0],Data[:,1],s=5)
    plt.show()
    return Data



#generating data for testing 

Data=gen_ran_data()

eps=float(input("\nEnter eps value: "))
minpts=int(input("\n Value of minimum points in eps neighbor: "))

#making clusters
obj_label,num_of_cluster=dbscan(Data,eps,minpts)

#plotting cluster with label
plot_data(Data,obj_label,num_of_cluster)
