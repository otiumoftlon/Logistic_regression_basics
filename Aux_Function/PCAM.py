# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:23:50 2024

@author: npava
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

def tabla(a,b,c,column):
    op = {'Eigenvalues' : a,
          'Variance explained' : b,
          'Columns': column[c]}
    tab = pd.DataFrame(op)
    tab.set_index(tab.columns[2], inplace=True)
    return tab


def pca(Raw,Mat,column):
    [eig_val, eig_vec] = np.linalg.eig(Mat)
    sorted_indices = np.argsort(eig_val)[::-1]  # Reverse the indices to get descending order
    sorted_val = eig_val[sorted_indices]
    sorted_vec = eig_vec[:, sorted_indices]
    explained_variance = np.sum(sorted_val)
    variance_percentages = (sorted_val / explained_variance) * 100
    data_pca =Raw@sorted_vec
    tab1=tabla(sorted_val,variance_percentages,sorted_indices,column)
    colna = ['PC'+'_'+str(i+1) for i in range(len(eig_val))]
    Factor_load = pd.DataFrame(sorted_vec,columns=colna).set_index(column[sorted_indices])
    data_pca.columns = colna
    return tab1,data_pca,Factor_load

def fa(Raw,Mat,column):
    [eig_val, eig_vec] = np.linalg.eig(Mat)
    m,_ = np.shape(eig_vec)
    sorted_indices = np.argsort(eig_val)[::-1]  # Reverse the indices to get descending order
    sorted_val = eig_val[sorted_indices]
    sorted_vec = eig_vec[:, sorted_indices]
    explained_variance = np.sum(sorted_val)
    variance_percentages = (sorted_val / explained_variance) * 100
    tab2=tabla(sorted_val,variance_percentages,sorted_indices,column)
    
    val_fac = sorted_val[sorted_val > 0]
    
    L = np.sqrt(val_fac[0:m])*sorted_vec[:,0:len(val_fac)]
    colna = ['F'+'_'+str(i+1) for i in range(len(val_fac))]
    Factor_load = pd.DataFrame(L,columns=colna).set_index(column[sorted_indices])
    e = np.sum(L**2,axis=1)
    Factor_load['e']=e
    r=np.linalg.solve(Mat, L)
    data_fa =Raw@r
    data_fa.columns = colna
    return tab2,data_fa,Factor_load

def plot_d(data_plot,x,y,title,hue=None):
    plt.figure()
    sns.scatterplot(x=x, y=y, data=data_plot,hue=hue,legend=True, s=10)  
    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['left'].set_color('gray')
    plt.gca().spines['bottom'].set_position(('data', 0))
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.xlabel(x, labelpad=-25, fontsize=11, ha='right', va='center',x=1)
    plt.ylabel(y, labelpad=-10, fontsize=11, ha='center', va='center', rotation=0, y=1)
    plt.grid(True)
    plt.title(title,fontsize=22)
    plt.show()

def euclidean_distance(matrix):
    num_points = len(matrix)
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distances[i, j] = np.sqrt(np.sum((matrix[i] - matrix[j]) ** 2))
    #distances = np.tril(distances)
    return distances
    
def cluster_analysis(dat):
    dista = euclidean_distance(dat)
    Z = hierarchy.linkage(dista)
    
    return Z

def plot_den(Z,title):
    plt.figure()
    dn = hierarchy.dendrogram(Z)
    plt.title('Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.grid(visible=True)
    plt.show()