"""Weight boosting classifiers. 
"""

# License: BSD 3 clause

from sklearn.base import clone
import numpy as np


__all__=[
    "BaseWeightBoosting", "AdaBoostClassifier"
]


EPSILON = 1e-10

class BaseWeightBoosting(object):
    """Base class for Weight Boosting. Implement based on sklearn. 
   
    Warning: This class should not be used directly. Use derived class instead.
    
    Parameters:
    -----------
    base_learner.  object 
        At each iteration, a base learner is trained on the weighted samples. In the end, 
        the weighted combination of all the base learner forms the final strong classifier.              
            
    n_learners. int, default 100, optional
        Number of the base learners. Note that if some kind of early-stopping strategies are 
        applied, the actual number of learned base learners can be less than n_learners.
        
    Attributes:
    -----------
    base_learners_: list of object. 
        Learned base learners.
        
    learner_weights_: List of float.
        Weights for combining base learners.
        
    """
    def __init__(self, 
                 base_learner=None,
                 learner_params=tuple(),  # Params passed to base learners
                 n_learners=100):
        
        # Parameters
        self.base_learner=base_learner
        self.n_learners=n_learners
        self.learner_params=learner_params        
                
        self.n_samples = 0   
        self.n_features = 0
        self.iterator = 0       
        self.early_stopping= False
        self.sample_weights = []
        
        # Attributes
        self.base_learners_ = []
        self.learner_weights_ = []
                
       
    def _make_learner(self, append = True):
        """Copy and properly initialize a base learner."""
        learner = clone(self.base_learner)
        learner.set_params(**dict((p, getattr(self, p)) for p in self.learner_params))
        if append:
            self.base_learners_.append(learner)            
        return learner


    def _check_stop_criteria(self):
        """Check all the stop criteria, return True if anyone of them is satified, 
            otherwise return False"""
        return self.early_stopping
    
    
    def _update_sample_weights(self):
        """Update samples' weights. Override it in derived classes."""               
        pass 
        

    def fit(self, X, y):
        """Train on a train set (X,y)
        
        Parameters:
        -----------
        X. array-like. shape [n_samples, n_features]
            Each row is a sample.
            
        y. array-like shape [n_samples]
            Corresponding label of samples.        
        Returns:
        --------
        self. object.
            return the object itself.
        """
        # Params check
        if not isinstance(X, (np.ndarray, np.generic)) or \
           not isinstance(y, (np.ndarray, np.generic)):
            raise ValueError("Numpy array is the only acceptable format.")
        
        if len(X.shape) != 2: 
            raise ValueError("X must be of shape [n_samples, n_features]")
        
        if len(y.shape) != 1:
            raise ValueError("Y must be of shape [n_samples]")
                
        
        # Initialize sample weights uniformly.
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.sample_weights = np.array([1.0/self.n_samples]*self.n_samples)
        self.X = X
        self.y = y
        
        
        # Main loop
        self.iterator = 0
        self.early_stopping = False
        while self.iterator < self.n_learners:
            # Train a new weak learner on the weighted samples.
            learner = self._make_learner()            
            bl = learner.fit(X,y,sample_weight=self.sample_weights)
            self.predicted_y = bl.predict(X)            
            # Update sample weights            
            self._update_sample_weights()
            # Check if stop criteria is satisfied.
            if self._check_stop_criteria():
                break         
            self.iterator += 1   
                    
        self.n_learners = len(self.base_learners_)
        return self


    def predict(self, X):
        """Predict y given a set of samples X.   
            
        Parameters:
        -----------
        X. array-like. shape [n_samples, n_features]
            
        Returns:
        --------
        y. array-like shape [n_samples]                
        """
        y_sum = 0
        for k in range(len(self.base_learners_)):
            y_sum += self.base_learners_[k].predict(X)*self.learner_weights_[k]
        return np.sign(y_sum)                
                
                
    def score(self, X, y):
        """Calculate the predict error given samples and targets.
        
        Parameters:
        -----------
        X. array-like. shape [n_samples, n_features]            
        y. array-like shape [n_samples]

        Returns:
        --------
        score. float.     
            Successful rate
        """
        predict_y = self.predict(X)
        # compute the successful rate        
        successful_rate = 1 - np.sum(np.abs(predict_y - y)*0.5)/X.shape[0]
        return successful_rate


##################################################################################################
class AdaBoostClassifier(BaseWeightBoosting):
    """Adaboost for classification.
            
    Parameters:
    -----------
    base_learner.  object 
        At each iteration, a base learner is trained on the weighted samples. In the end, 
        the weighted combination of all the base learner forms the final strong classifier.              
            
    n_learners. int, optional (default=100)
        Number of the base learners.
    
    Reference:
    ---------
    Yoav Freund and Robert E. Schapire. 1997. A decision-theoretic generalization of on-line 
    learning and an application to boosting. J. Comput. Syst. Sci. 55, 1 (August 1997), 119-139.
    """
    def __init__(self,
                 base_learner=None,
                 learner_params=tuple(), 
                 n_learners=100):        
        super(AdaBoostClassifier, self).__init__(base_learner, 
                                                 learner_params, 
                                                 n_learners)        
        
        self.alpha = 0              
        self.predict_error = []        
        self.weighted_errors = []
        
        self.staged_predict_y = []
        self.staged_sample_weights = []


    def _check_stop_criteria(self):
        """Overridden. Check all the stop criteria, return True if anyone of them is satified, 
        otherwise return False"""
        early_stopping = False
        # Return true, if no improvement any more. 
        error = np.abs(self.predicted_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        if np.sum(error) < EPSILON:
            early_stopping = True
        
        return early_stopping or super(AdaBoostClassifier, self)._check_stop_criteria()


    def _update_sample_weights(self):
        """Overridden. Update samples' weight."""               
        error = np.abs(self.predicted_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        weighted_error = np.inner(error, self.sample_weights)
        learner_weight = 0.5 * np.log(
                            (1.0 - weighted_error + EPSILON) / (weighted_error + EPSILON)
                        )        

        self.sample_weights *= np.exp(learner_weight * (error - 0.5)*2.0)
        self.sample_weights /= np.sum(self.sample_weights)
                    
        self.learner_weights_.append(learner_weight)
        self.weighted_errors.append(weighted_error)

        self.staged_predict_y.append(np.copy(self.predicted_y))
        self.staged_sample_weights.append(np.copy(self.sample_weights))



 

