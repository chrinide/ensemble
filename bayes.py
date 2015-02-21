"""Naive Bayes classifier
"""

# License: BSD 3 clause

import numpy as np
import unittest


__all__=[
    "Gaussian", 
    "NaiveBayes"    
]

###################################################################################################
class Gaussian(object):
    """Multivariate Guassian distribution. Default is 1D standard normal distribution N(0,1).
      
    Parameters: 
    -----------
    mean. 1D array-like of length N. default, 0.0
        Mean of N-dimensional Guassian distribution
    
    cov. 2D array-like of shape (N, N) . default, 1.0
        Covariance matrix. Must be symmetric and positive-semidefinite.
    """
    def __init__(self,
                 mean = np.array([0.0]), 
                 cov = np.array([1.0])):
        #Check parameters 
        if not (isinstance(mean, np.ndarray) and len(mean.shape) == 1):   
            raise ValueError("mean must be 1D array.")
        
        if not (isinstance(cov, np.ndarray) and     # array
                len(cov.shape) == 2 and             # 2D
                cov.shape[0] == cov.shape[1]):      # symmetric, ignore 'positive semi-definite'
            raise ValueError("cov must be a 2D symmetric array.")
        
        if mean.shape[0] != cov.shape[0]: # mean and cov should be the same
            raise ValueError("Dimension of mean and cov is not consistent.")
        
        
        # Attributes
        self.mean_ = mean 
        self.cov_ = cov 
        
        # Internal variables for efficience.
        self.cov_det = np.linalg.det(self.cov_) 
        self.cov_inv = np.linalg.inv(self.cov_)
        
    
    def _diagonal_inv(self, cov):
        """Invert a diagonal matrix. 
        
        Warning: This method should only be used internally. Parameter-checking is ignored.
        
        Params:
        -------
        cov. array-like. of shape [N, N]
            Covariance matrix(diagnoal).
        
        Return:
        -------
        cov_inv. array-like. of shape [N, N]
            Inverse of the input matrix.
        """    
        epsilon = 1e-10
        cov_diag = np.diag(cov) + epsilon  # Regulization to avoid dividing-by-zero
        cov_inv = np.eye(cov.shape[0], cov.shape[1], np.float)
        np.fill_diagonal(cov_inv, 1.0/cov_diag)
        return cov_inv
        
        
    def pdf(self, X):        
        """Compute the PDF of samples.
        Parameters:
        ----------
        X. array of shape [n_samples, n_features]
            Each row represents a sample. 
            
        Return:
        -------
        PDF of samples
        """
        xx = np.dot(np.dot((X-self.mean_), self.cov_inv), np.transpose(X-self.mean_))
        return np.exp(-0.5*xx.diagonal())        


        
        
        