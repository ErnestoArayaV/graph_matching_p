"""Module containing some graph matching (GM) algorithms refered to in the paper "seeded graph matching via projected power method"
TODO: -add reference to each method.
      -add different algs. for projecting onto simplex with better theoretical performance
      -check for sintax consistency
      -add different algs. for projecting onto simplex with better theoretical performance

"""
import numpy as np
from numpy import linalg as LA
from graph_matching_p.projection_simplex.projection_simplex import  projection_simplex_sort
from graph_matching_p.aux_methods.aux_methods import lin_assign, greedy_weighted_matching


def Umeyama(A,B):
    """ Umeyama spectral seedless GM algorithm. 
    Input: A,B = pair of 'correlated' matrices,
    Output: Per_mat= a permutation matrix in csr format,
    
    Raises
    ------
    ValueError if the matrices are of different size 
    ...
    
    """
    if np.shape(A)!=np.shape(B):
        raise ValueError('Matrices should be of the same size...')
    Lambda, U = LA.eig(A)
    Mu,V = LA.eig(B)
    idx = Lambda.argsort()[::-1]   
    Lamba_sorted = Lambda[idx]
    U_sorted = U[:,idx]
    idx = Mu.argsort()[::-1]   
    Mu_sorted = Mu[idx]
    V_sorted = V[:,idx]
    X=np.matmul(np.abs(U_sorted),np.transpose(np.abs(V_sorted)))
    
    # Rounding by linear assignment 
    Per_mat=lin_assign(X)
    return Per_mat
    

def EigenAlign(A,B):
    """EigenAlign algorithm, from "Spectral Alignment of Graphs" by S.Feizi, G.Quon, M.Recamonde-Mendoza, M.Médard, M.Kellis, A.Jadbabaie.
    uses np.kron() it might be slow.
    Inputs: A,B= correlated matrices
    Output: Per_mat=a permutation matrix in csr format
    Raises
    ------
    ValueError if the matrices are of different size 
    ...
    """
    if np.shape(A)!=np.shape(B):
        raise ValueError('Matrices should be of the same size...')
    n=len(A[0,:])
    w, vr = LA.eig(A)
    largest_eigenvector_1 = vr[:, np.argmax(w)]
    w_2, vr_2 = LA.eig(B)
    largest_eigenvector_2 = vr_2[:, np.argmax(w_2)]
    X=np.abs(np.outer(largest_eigenvector_1,largest_eigenvector_2)) #eigenvectors are defined modulo sign
    
    # Rounding by linear assignment
    Per_mat=lin_assign(X) 
    #Per_mat=lin_assign(X)
    return Per_mat

def LowRank(A,B,k):
    """Lowrank algorithm, same reference as EigenAlign().
    Inputs: A,B= correlated matrices
    Output: Per_mat=a permutation matrix in csr format
    Raises
    ------
    ValueError if the matrices are of different size 
    ...
    """
    if np.shape(A)!=np.shape(B):
        raise ValueError('Matrices should be of the same size...')
    n=len(A[0,:])
    Lambda, U = LA.eig(A)
    Mu,V = LA.eig(B)
    idx = Lambda.argsort()[::-1]   
    #Lamba_sorted = Lambda[idx]
    U_sorted = U[:,idx]
    idx = Mu.argsort()[::-1]   
    #Mu_sorted = Mu[idx]
    V_sorted = V[:,idx]
    X=np.matmul(np.abs(U_sorted[:,0:k]),np.transpose(np.abs(V_sorted[:,0:k])))
    ###Alternatively, define an array of signs 's' of lenght k
    #X=np.matmul(np.matmul(U_sorted[0,0:k],s),np.transpose(V_sorted[:,0:k]))
    
    # Rounding by linear assignment 
    Per_mat=lin_assign(X)
    return Per_mat
    
def IsoRank(A,B,H,alpha,maxiter,tol):
    """Isorank algorithm, taken from  "Global alignment of multiple protein interaction networks with application to functional orthology detection" by R. Singh, J. Xu, B. Berger. 
    this version is based on the implementation by S. Zhang in https://github.com/sizhang92/FINAL-network-alignment-KDD16/blob/master/IsoRank.m .
    Here the matrix H is a prior similarity between nodes of both graphs
    the algorihtm is essentially PPR on the product graph.
    Inputs: A,B= correlated matrices
    Output: Per_mat=a permutation matrix in csr format
    Raises
    ------
    ValueError if the matrices are of different size 
    ...
"""
    if np.shape(A)!=np.shape(B):
        raise ValueError('Matrices should be of the same size...')
    n = len(A[0,:])
    A_nor=A/A.sum(axis=1)[:,None]
    B_nor=B/B.sum(axis=1)[:,None]
    A_nor[np.isnan(A_nor)]=0
    B_nor[np.isnan(B_nor)]=0
    X=(1/n**2)*np.ones((n,n))
    
    for iter in range(maxiter):
        previous=X
        X = alpha *np.matmul(np.matmul( B_nor.T, X), A_nor) + (1-alpha) * H
        delta = LA.norm(X- previous)        
        if delta < tol: 
            break
    
    # Rounding by linear assignment 
    Per_mat=lin_assign(X)
    return Per_mat

def projected_power_method(A,B,init_point,maxiter=100):
    """GMPPM is an iterative algorithm, using the projected power method, for the seeded GM problem. Reference: 
    Input: A,B= correlated random matrices to be matched
           init_point= initial to star the iteration, corresponds to a 'noisy seed' for the problem.
           maxiter= number of projected power iterations 
    Output: a permutation matrix in csr format
    Raises
    ------
    ValueError if the matrices are of different size
    ...
    """
    if np.shape(A)!=np.shape(B):
        raise ValueError('Matrices should be of the same size...')
    actual_point=init_point
    for i in range(maxiter):
        grad=(A@actual_point)@B
        actual_point=greedy_weighted_matching(grad,-2000)
    return actual_point

def Grampa(A,B,eta):
    """Grampa algorithm, from "Spectral Graph Matching and Regularized Quadratic Relaxations I" by Z.Fan, C.Mao, J.Xu, Y.Wu
    Input: A,B= correlated matrices
           eta= 'ridge regularization' parameter.
    Output: Per_mat= a permutation matrix in csr_format
    """
    n = len(A[0,:])
    Lambda, U = LA.eig(A); Lambda=np.real(Lambda);U=np.real(U)
    Mu,V = LA.eig(B);Mu=np.real(Mu);V=np.real(V)
    coeff = 1 / (np.subtract.outer(Lambda,Mu)**2 + eta**2)
    #print(coeff.shape)
    coeff = coeff* np.matmul(np.matmul(U.T, np.ones((n,n))), V)
    X= np.matmul(np.matmul(U , coeff), V.T)
    
    # Rounding by linear assignment 
    Per_mat=lin_assign(X)
    return Per_mat 

def simplex_GD(A,B,init_point,gamma=1,maxiter=100):
    """Projected gradient descent onto the simplex. Converges to a solution of a convex relaxation of the problem.
    Input: A,B= matrices to be matched
           init_point= initial point for PGD
           gamma= a parameter for balancing terms in the objective function.
    Output: Per_mat= permutation matrix in csr format

     """
    n=np.shape(A)[1]
    A2=np.matmul(A,A)#A@A
    B2=np.matmul(B,B)#B@B
    X=np.copy(init_point.T)
    X=np.reshape(X,n**2,order='F')
    X=projection_simplex_sort(X)
    X=np.reshape(X,(n,n),order='F')
    for i in range(maxiter):
        grad=-2*np.matmul(np.matmul(A,X),B)+np.matmul(A2,X)+np.matmul(X,B2)+0.2*LA.norm(X,'fro')
        X=X-gamma*grad
        X=np.reshape(X,n**2,order='F')
        #print(np.shape(X))
        X=projection_simplex_sort(X,n)
        #print(np.shape(X))
        X=np.reshape(X,(n,n),order='F')
    Per_mat=lin_assign(X)
    return Per_mat
