"""Module containing auxiliary algorithms used by graph matching(GM) algorithms.
TODO: -add reference to each method.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy import linalg as LA
from scipy.sparse import csr_matrix
from random import randint, sample
def lin_assign(X):
    """Max Linear Assigment, using a similarity matrix. Uses linear_sum_assignment.
    TODO: link to the documentation of linear_sum_assignment.
    Input: X= similarity or cost matrix
    Output: conn_matrix= permutation matrix as csr_matrix (scipy sparse).
    """
    row_assign,col_assign = linear_sum_assignment(-X)
    w = np.empty(len(row_assign), dtype=int)
    w[:] = 1
    conn_matrix = csr_matrix((w, (row_assign, col_assign)), shape=(len(row_assign), len(col_assign)), dtype=int)
    return conn_matrix

def GM_objective_fnct(A,B,perm):
    """Objective fnct for the non-convex QAP formulation of GM. 
    Inputs: A,B= correlated matrices
    Output: QAP objective function
    Raises
    ------
    ValueError if the matrices are of different size 
    ...
    """
    if np.shape(A)!=np.shape(B):
        raise ValueError('Matrices should be of the same size...')
    else: return np.trace(np.matmul(np.matmul(np.matmul(A,perm),B),perm.T))
    
#Objective fnct convex formulation of GM
def GM_objective_fnct_conv(A,B,perm):
    """Objective fnct for the convex relaxation of GM. 
    Inputs: A,B= correlated matrices
    Output: convex objective function
    Raises
    ------
    ValueError if the matrices are of different size 
    ...
    """
    if np.shape(A)!=np.shape(B):
        raise ValueError('Matrices should be of the same size...')
    else: return LA.norm(A@perm-perm@B,'fro')
    
def mismatch_frac(perm1,perm2):
    """Mismatch fraction between two permutations matrices
    Inputs: perm1,perm2= two permutation matrices as numpy arrays
    Output: mismatch fraction
    Raises
    ------
    ValueError if the matrices are of different size
    ...
     """
    if np.shape(perm1)!=np.shape(perm2):
        raise ValueError('Matrices should be of the same size...')
    else: return 0.5*np.sum(np.abs(perm1-perm2))/len(perm1[0,:])
    
def perm_mat(perm):
    """"transform an assingment sequence to a sparse permutation matrix
    Input: perm=a permutatin sequence
    Output: permu= a permutation matrix in csr format.
     """
    w = np.empty(len(perm), dtype=int)
    w[:] = 1
    permu=csr_matrix((w, (perm, np.arange(len(perm)))), shape=(len(perm), len(perm)), dtype=int).A
    return permu

def sample_spherical(npoints, ndim=3):
    """Generate a set random points on the euclidean sphere.
    Input: npoints= number of points
           ndim= is ambient dimension.
    Output: matrix of size ndim x npoints containing the random points
     """
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def random_derangement(n):
    """generate random derangements, code was taken from https://stackoverflow.com/questions/25200220/generate-a-random-derangement-of-a-list, user georg
    """
    while True:
        v = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)

def greedy_weighted_matching(simil_matrix,thresh_param):
    """a greedy method to solve linear assignment, corresponds to GMWM algorithm in the ref:
    Input: simil_matrix= cost matrix for LAP
           thresh_param= the threshold to replace the erasing step of the algorithm. 
    Output: a permutation matrix
    """
    n=np.shape(simil_matrix)[1]
    #shape=np.shape(simil_matrix)
    matching=np.zeros((n,n)) #initialize the permutation matrix 
    for i in range(n):
        ii=np.unravel_index(simil_matrix.argmax(),(n,n))
        matching[ii[0],ii[1]]=1
        simil_matrix[ii[0],:]=thresh_param*np.ones(n)
        simil_matrix[:,ii[1]]=thresh_param*np.ones(n)
        #simil_matrix[ii[0],:]=thresh_param*np.ones((1,n))
        #simil_matrix[:,ii[1]]=thresh_param*np.ones((n,1))
    return matching

def rnd_permutation_inNeighborhood(size,neigh_radius, reference_permutation): 
    """creates a permutation in a Frobenius neighborhood (defined by neigh_radius) of reference_permutation"""
    permu=np.copy(reference_permutation)
    sampled_indices=sample(list(range(size)),k=neigh_radius)
    shuf_sampled_indices=np.array(sampled_indices)[np.array(list(random_derangement(neigh_radius)))]
    permu[:,np.array(shuf_sampled_indices)]=permu[:,sampled_indices]
    return permu