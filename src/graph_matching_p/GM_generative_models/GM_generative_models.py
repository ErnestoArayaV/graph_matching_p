"""Module containing generative models for statistically correlated random graphs for testing graph matching(GM) algorithms.
TODO: -add more models, eg with community structure.
      -add more exceptions.
"""
import numpy as np
from scipy.spatial import distance
from graph_matching_p.aux_methods.aux_methods import sample_spherical,  perm_mat
def ER(n,prob):
    """ 
    Sample a single Erdos-Renyi graph. 
    Input: n= size of the graph
           prob= connection probability
    Output: an adjacency matrix 
    Raises
    ------
    ValueError if the probability if 0<=prob<=1 is false
    
    """
    if prob<0 or prob>1:
        raise ValueError('probability should be between 0 and 1...')
    Ber=np.random.binomial(n=1,p=prob,size=(n,n))
    UT=np.triu(Ber)
    return UT+UT.T-2*np.diag(np.diag(Ber))

def GaussianWigner(n,var):
    """Sample a single Gaussian Wigner matrix. Start from Gau a Ginibre matrix, there might be a more efficient way to do this. 
       Input: n= size of the graph
              var= variance fo each entry. For G.O.E choose var=1/n.
    Output: a weighted adjacency matrix
    """
    Gau=np.random.normal(loc=0,scale=np.sqrt(var),size=(n,n))
    UT=np.triu(Gau)
    return UT+UT.T

def Corr_Gaussian(n,var, sigma):
    """Sample correlated Gaussian Wigner matrix. Uses GaussianWigner() method. Uses perm_mat().
    Input:  n= size of each graph.
            var= variance fo each entry. For G.O.E choose var=1/n.
            sigma= noise level 
    Output: A,B= two weighted correlated adjacency matrices
            P_rnd= a 'ground truth' permutation used to shuffle the rows/columns of B
    """
    #generate a ground true permutation
    true_perm=np.random.permutation(np.arange(n)) 
    P_rnd=perm_mat(true_perm)
    A=GaussianWigner(n,var)
    C=GaussianWigner(n,var)
    B=np.sqrt(1-sigma**2)*A+sigma*C
    #shuffle rows and columns of matrix B
    aux=B[:,true_perm]
    B=aux[true_perm,:]
    return A,B,P_rnd
   
def Corr_ER(n,prob,s):
    """Sample correlated Erdos-Renyi matrices. Uses ER() method. Sample A and then B conditional to A.
    Input:  n= size of each graph.
            prob= connection probability
            s= correlation level 
    Output: A,B= two correlated adjacency matrices
            P_rnd= a 'ground truth' permutation used to shuffle the rows/columns of B
    """
    #generate a ground true permutation
    true_perm=np.random.permutation(np.arange(n)) 
    P_rnd=perm_mat(true_perm)
    q=prob*(1-s)/(1-prob)
    A=ER(n,prob) #first sample A
    #we sample B conditional to A
    B1=np.random.binomial(n=1,p=s,size=(n,n))
    B1[A==0]=0
    B2=np.random.binomial(n=1,p=q,size=(n,n))
    B2[A==1]=0
    B=B1+B2
    B=B-np.diag(np.diag(B))
    #shuffle rows and columns of matrix B
    aux=B[:,true_perm]
    B=aux[true_perm,:]
    return A,B,P_rnd

def RGG(n,d,tau):
    """Sample a Random Geometric graph on the sphere. Uses sample_spherical().
    TODO: add the reference as there's more than way to define this model
  
    Sample a single Erdos-Renyi graph. 
    Input: n= size of the graph
           d= ambient dimension of latent points
           tau= threshold parameter
    Output: an adjacency matrix 
    Raises
    ------
    ValueError if the probability if -1<=tau<=1 is false
    ...
    """
    if tau<-1 or tau>1:
        raise ValueError('threshold should be between -1 and 1...')
    rnd_points_sphere=sample_spherical(n,d)
    gram=rnd_points_sphere@rnd_points_sphere.T
    adjacency=np.copy(gram)
    adjacency[adjacency<=tau]=0
    return gram,adjacency

def Corr_RGG(n,d,tau,sigma):
    """Sample correlated Random Geometric graphs on the sphere. Uses RGG().
    TODO: add the reference as there's more than way to define this model
  
    Sample a single Erdos-Renyi graph. 
    Input: n= size of the graph
           d= ambient dimension of the latent points
           tau= threshold parameter
           sigma= noise level
    Output: an adjacency matrix 
    Raises
    ------
    ValueError if the probability if -1<=tau<=1 is false
    ...
    """
    if tau<-1 or tau>1:
        raise ValueError('threshold should be between -1 and 1...')
    #generate a ground true permutation
    true_perm=np.random.permutation(np.arange(n)) 
    P_rnd=perm_mat(true_perm)
    rnd_points_gaussian_A=np.random.normal(loc=0,scale=1/np.sqrt(d),size=(n,d))# corresponds to the first cloud of points cloud_A
    rnd_points_gaussian_B=np.random.normal(loc=0,scale=1/np.sqrt(d),size=(n,d))
    cloud_B=np.sqrt(1-sigma**2)*rnd_points_gaussian_A[true_perm]+sigma*rnd_points_gaussian_B
    DistA=distance.cdist(rnd_points_gaussian_A, rnd_points_gaussian_A, 'euclidean')
    DistB=distance.cdist(cloud_B, cloud_B, 'euclidean')
    GraphA=np.copy(DistA)
    GraphA[GraphA<=tau]=1
    GraphA[GraphA>tau]=0
    GraphA=GraphA-np.diag(np.diag(GraphA))
    GraphB=np.copy(DistB)
    GraphB[GraphB<=tau]=1
    GraphB[GraphB>tau]=0
    GraphB=GraphB-np.diag(np.diag(GraphB))
    return DistA, DistB, GraphA, GraphB,P_rnd

#put here other models...    

