# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:22:42 2019

@author: Vishal Dhiman
"""

import numpy as np
import sys
from matplotlib import pyplot as plt



class H_clustering:
    
    def __init__(self,_nc,data):
        
        #initializer 
        self.data=data
        self.no_of_cluster=_nc
        self.dimension=data.shape[1]
        self.no_of_dataset=data.shape[0]
        self.clusters={}
        self.centroids={}
        self.available_color=['r','y','b','g','m']
        self.points_in_cluster={}
        self.uniq_cluster=[]
    def plot_clusters(self):
        
        it_count=self.no_of_dataset-self.no_of_cluster
        
       
        req_cluster=self.clusters[it_count]
        #print(self.clusters)
        self.uniq_cluster=[]
        #extracting unique clusters
        self.uniq_cluster=np.unique(req_cluster)
        
        plt.style.use('ggplot')
        fig=plt.figure()
        fig.suptitle('Agglomerative Clustering')
        _mt=fig.add_subplot(1,1,1)
        _mt.set_xlabel('x')
        _mt.set_ylabel('y')
        
        self.points_in_cluster={}
        for _index in self.uniq_cluster:
            self.points_in_cluster[_index]=np.where(req_cluster==_index)
            
    
        print(req_cluster)
        #print(self.points_in_cluster)
        #print(self.centroids)
        _cl=0
        
        for i in self.uniq_cluster:
            for j in np.nditer(self.points_in_cluster[i]):
                _mt.scatter(self.data[j,0],self.data[j,1],c=self.available_color[_cl])
            _cl=_cl+1  
        
        plt.show()
        
    def cal_homogenity(self):
        
        print("\nCalculating Homogenity of Clusters  : ")
        homogenity={}
        for i in self.uniq_cluster:
            
            
            l=[]
            for j in np.nditer(self.points_in_cluster[i]):
                homo_d=0.00
                for k in np.nditer(self.points_in_cluster[i]):
                    homo_d+=np.linalg.norm(np.array(self.data[j])-np.array(self.data[k]))
                l.append(homo_d)
            
            _max=max(l)
            _min=min(l)
            for j in range(0,len(l)):
                  l[j]=(l[j]-_min)/(_max-_min)
            homogenity[i]=sum(l)
            print("\n Homogenity of each point in cluster  {} is : \n {} ".format(str(i),str(homogenity[i])))
    
        
    def cal_seperation(self):
        print("\nCalculating Seperation of Clusters: ")
        
        l_c=self.uniq_cluster
        m=len(l_c)
        
        if m>1:
            for i in self.uniq_cluster:
               
               for j in self.uniq_cluster:
                   
                   if i!=j :
                      
                       sep=0.00
                       count=0
                       for k in np.nditer(self.points_in_cluster[i]):
                           for l in np.nditer(self.points_in_cluster[j]):
                             sep+=np.linalg.norm(np.array(self.data[k])-np.array(self.data[l]))
                             count+=1
                       if count!=0:
                           print("\nSeperation between cluster {} and {} is : {}".format(str(i),str(j),str(sep/count)))    
                       else:
                           print("\nSeperstion errot: count=0 ")
        else :
            print("\nInsufficent number of clusters to find sepearion")
            
            
    def change_no_of_cluster(self,_nc):
        self.no_of_cluster=_nc
        
    def aglo_clustering(self):
        #print(self.data)
        #print("\n",self.dimension)
        #print("\n",self.no_of_dataset)
        dist_mat=np.sqrt(np.sum((self.data[None,:]-self.data[:,None])**2,-1))
        #print(dist_mat)
        np.fill_diagonal(dist_mat,sys.maxsize)
        #print(dist_mat)
        self.do_clustering(dist_mat)
    
    
    def do_clustering(self,dist_mat):
        
        index_r=-1
        index_c=-1
        l=[]
    
        for i in range(0,self.no_of_dataset):
            l.append(i)
            self.centroids[i]=self.data[i]
        self.clusters[0]=l.copy()
        #print(self.clusters)
        
        #calculating min distance between two clusters i,j m,
        for p in range(1,self.no_of_dataset):
            _min=sys.maxsize
            
            for i in range(0,self.no_of_dataset):
                for j in range(0,self.no_of_dataset):
                    if(dist_mat[i][j]<=_min):
                        _min=dist_mat[i][j]
                        index_r=i
                        index_c=j
        
            #merging clusters with min distance between them
            min_i=min(index_c,index_r)
            max_i=max(index_c,index_r)
            for i in range(0,len(l)):
                if(l[i]==max_i):
                    l[i]=min_i
            #print("\n",min_i,max_i)        
            self.clusters[p]=l.copy()
            
            #centroid of merged clusters
            self.centroid_after_merge(min_i,l)
            
            #updating distance matrix after merging 
            for i in range(0,self.no_of_dataset):
                if(i!=index_r and i!=index_c):
                    dist_of_i_from_centroid=self.cal_dist_from_cent(i,min_i)
                    if(dist_mat[min_i][i]!=sys.maxsize):
                        dist_mat[min_i][i]=dist_of_i_from_centroid
                        dist_mat[i][min_i]=dist_of_i_from_centroid
            
            #updating row and column belonging to cluster with index max_i of distance matrix 
            for i in range(0,self.no_of_dataset):
                dist_mat[max_i][i]=sys.maxsize
                dist_mat[i][max_i]=sys.maxsize
            #print(p,dist_mat)
            
    def centroid_after_merge(self,min_i,l):
        count=0
        x_sum=0
        y_sum=0
        for i in range(0,len(l)):
            if(l[i]==min_i):
                x_sum+=self.data[i][0]
                y_sum+=self.data[i][1]
                count+=1
        self.centroids[min_i]=[x_sum/count,y_sum/count]
        
        #del self.centroids[max(index_c,index_r)]
    
    def cal_dist_from_cent(self,i,min_i):
        
        d=np.linalg.norm(np.array(self.centroids[i])-np.array(self.centroids[min_i]))
        return d
    
    
        
        
        
   
        
        
def gen_ran_data():
        print("\nWant to generate random data then give details:")
        n=int(input("\nHow many centers?")) 
        _center_list=[]
        for i in range(n):
            mes="\n"+str(i+1)+" center:"
            x,y=map(int,input(mes).split(','))
            _center_list.append([x,y])
        #print(_center_list)
        
        #data generation around different centers
        _data=[]
        for i in range(n):
            _data.append(np.array(_center_list[i])+np.random.randn(50,2))
        _data=np.array(_data)
        _data=_data.reshape(-1,2)
        
        #plotting origiinal data
        plt.style.use('ggplot')
        fig=plt.figure()
        fig.suptitle('Original Data')
        _mt=fig.add_subplot(1,1,1)
        _mt.set_xlabel('x')
        _mt.set_ylabel('y')
        plt.scatter(_data[:,0],_data[:,1])
        plt.show()
        #print(_data)    
        return _data
    
        
if __name__ == '__main__':
    
    _data=gen_ran_data()
    #print(_data)
    
    _Cn=int(input("\nHow many cluster do you want to find in agglomerative clustering? :\t"))
    #print(_data.shape)
    obj_agglo=H_clustering(_Cn,np.array(_data))
    obj_agglo.aglo_clustering()
    obj_agglo.plot_clusters()
    obj_agglo.cal_homogenity()
    obj_agglo.cal_seperation()
    option=input("\nHave You changed your mind w.r.t number of cluster do you want  to see?(yes,no) :\t")
    
    while option.lower()=='yes':
        _Cn=int(input("\nThen, How many cluster do you want to find? :\t"))
        obj_agglo.change_no_of_cluster(_Cn)
        obj_agglo.plot_clusters()
        
        obj_agglo.cal_homogenity()
        obj_agglo.cal_seperation()
        option=input("\n Again change of mind?(yes,no) :\t")
        
        