import numpy as np
import scipy as sp
import random


class VRPCA(object):
    def __init__(self, init_weights, train_data, ground_truth, learning_rate, m, T, log_freq=0):
        """
        init_weights: k by d
        ground_truth: rows of top k eigenvectors of training data, k by d
        train_data: rows of data points, n by d
        """
        self.X = train_data
        self.W_init = init_weights.transpose()
        #self.W = None
        self.gt = ground_truth.transpose()
        self._set_eta(learning_rate)
        self.m = m
        self.T = T
        self.log_freq = log_freq
    
    def _train(self):
        self._reset_train_logs()
        W_tilde = self.W_init
        #self._register_eval_metrics(W_tilde)
        #print(f"The initial loss is {self._train_log[-1]}")
        epoch = 0
        for t in range(self.T):
            U_tilde = 1.0/len(self.X) * np.matmul(np.matmul(self.X.T, self.X), W_tilde)
            epoch += 1
            W_tilde = self._train_inner(U_tilde, W_tilde, epoch)
            epoch += 1
            #self._register_eval_metrics(W_tilde)
            print(f"The loss at the {epoch+1}-th epoch is {self._train_log[-1]}")
        self.final_weights_ = W_tilde.copy()
    
    
    def _train_inner(self, U_out, W_out, epoch):
        
        W_ = W_out.copy()
        for ti in range(self.m):
            B = get_rotation_matrix(W_, W_out)
#             if t == 0 or t == 1:
#                 Diff = W - np.matmul(W_out, B)
            #x = self._get_sample_from_train()
            x = self.X[ti]
            
            W = svrg_power_iteration_update(x, W_, W_out, U_out, B, self.eta)
            W_ = orthonormalize(W)
            if self.log_freq > 0 and ti % self.log_freq == 0:
                self._register_eval_metrics(W_, epoch, ti)
        return W
            
            
    def _set_eta(self, lr):
        self.eta = lr
        
            
    def _get_sample_from_train(self):
        # random indx
        ridx = random.randint(0, len(self.X)-1)
        return self.X[ridx]
    
    
    def _reset_train_logs(self):
        self._train_log  = []
        
    
    def _register_eval_metrics(self, weights, epoch, t):
        """
        register the evaluation metrics to train log during training 
        """
        loss = eval_with_groundtruth(self.gt.transpose(), weights.transpose())
        print('The loss at the {}-th epoch {}-th iteration is {}'.format(epoch, t, loss))
        self._train_log.append(loss)


def get_rotation_matrix(W_1, W_2):
    """
    W_1/W_2: d by k
    Find the optimal rotation matrix B of W_2
    so that it's closest to W_1 in Frobenius norm
    """
    u, _, vh = np.linalg.svd(np.matmul(W_1.T, W_2))
    return np.matmul(u, vh).transpose()


def svrg_power_iteration_update(x, W, W_out, U_out, B, learning_rate):
    """
    Perform one iteration of SVRG update on W
    """
    A_tilde = np.matmul(x.reshape([-1,1]), x.reshape([1,-1])) # check A_tilde
    Dif = W - np.matmul(W_out, B)
    #print(A_tilde.shape)
    #print(Dif.shape)
    Imp = np.matmul(A_tilde, Dif) + np.matmul(U_out, B)
    W_ = W + learning_rate * Imp 
    return orthonormalize(W_)


# def orthonormalize(W):
#     W_sqrt = sp.linalg.sqrtm(np.matmul(W.T, W))
#     #print(sp.linalg.sqrtm(np.matmul(W.T, W)))
#     W_sqrt_inv = np.linalg.inv(W_sqrt) 
#     return np.matmul(W, W_sqrt_inv) 

def orthonormalize(W):
    # W is d by k!
    ortho_weights_, R = np.linalg.qr(W)
    return ortho_weights_


def eval_with_groundtruth(groundtruth, weights):
    """
    Input: 
        groundtruth, rows of top k eigenvectors, k by d
        weights, np array, shape k by d (row-orthonormal)
    Return:
        tr(U(I-P)): U-projection matrix of eigenvectors, 
                    P: projection matrix of weights
    """
    #eigenvecs = get_eigenvecs(groundtruth)
    eigenvecs = groundtruth.copy()
    U = np.matmul(eigenvecs.T, eigenvecs)
    P = np.matmul(weights.T, weights)
    I = np.eye(weights.shape[1])
    
    return np.trace(np.matmul(U, I-P))

def get_default_learning_rate(X):
    rbar = np.linalg.norm(X, ord='fro')
    return 1/rbar * (len(X) ** 0.5)
