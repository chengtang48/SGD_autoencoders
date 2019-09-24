import numpy as np
import math
import tensorflow as tf


class KrasulinaPCA(object):
    
    def __init__(self, init_weights, train_data, groundtruth=None,
                 learning_rate=[None, 0.0001], mini_batch_size=1, 
                 max_iter=100, sampling=False, log_freq=100):
        
        self._init_weights = init_weights
        #self._weights = tf.Variable(init_weights, dtype=tf.float64)
        self._train_data = train_data
        self._train_data_size = train_data.shape
        #assert learning_rate[0] is None, "Decaying learning rate not implemented yet!"
        self._learning_rate = learning_rate
        #self._minibatch_iter = 0
        #assert groundtruth, "training with real data is not implemented yet!"
        self._groundtruth = groundtruth
        self._k, self._d = init_weights.shape[0], init_weights.shape[1]
        self._mbsize = mini_batch_size
        self._T = max_iter
        self._sampling = sampling
        self._log_freq = log_freq
        self._global_step = 0
        self._train_mse_log = list()
    
    def _update_eta(self):
        if self._learning_rate[0] is None:
            self._eta = self._learning_rate[1]
        else:
            #self._eta = self._learning_rate[0] / math.log(self._global_step + self._learning_rate[1])
            self._eta = self._learning_rate[0] / (self._global_step + self._learning_rate[1])
            #self._eta = self._learning_rate[0] / math.sqrt(self._global_step + self._learning_rate[1])
    
    def _update_global_step(self):
        self._global_step += 1
        
    def _train(self):
        # create train log for objective loss (mse)
        self._train_mse_log.append(eval_mse_loss(self._train_data, self._init_weights))
        #if self._groundtruth:
            # create another log for groundtruth metric 
        self._groundtruth_eval_log = list()
        self._groundtruth_eval_log.append(eval_with_groundtruth(self._groundtruth, self._init_weights))
        print(f"The initial mse: {self._train_mse_log[0]}")
        print(f"The initial loss: {self._groundtruth_eval_log[0]}")
       
        mb_iter = 0
        self._epoch = 0
        weights = self._init_weights.copy() 
        ## start training      
        for t in range(1, self._T+1):
            
            epoch_old = self._epoch
            if not self._sampling:
                # get mini batches by iterating over the dataset
                mini_batch, mb_iter = self._get_mini_batch(mb_iter, t)
            else:
                # sampling u.a.r.
                mini_batch = self._get_mini_batch_from_sampling()
            self._update_global_step()
            self._update_eta()
            # run optimization
            # W^{t+1} = W^t + eta^t (W^t x x^T - W^t x x^T(W^t)^TW^t)
            
            #print("mini batch shape", mini_batch.shape)
            emp_cov = 1.0/self._mbsize * np.matmul(mini_batch.T, mini_batch)
            
            #print("Computing empirical matrix takes {}".format(te-ts))
            
            A = np.dot(weights, emp_cov) # k by d and d by d
            
            #print("Computing A matrix takes {}".format(te-ts))
            
            #B = np.matmul(weights.T, weights) # d by k and k by d
            #print(weights.flags)
            #B = np.dot(weights.T, weights)
            B = np.dot(A, weights.T)
           
            #print("Computing B matrix takes {}".format(te-ts))
            
            #C = np.matmul(A, B)
            C = np.dot(B, weights)
          
            #print("Computing C matrix takes {}".format(te-ts))
            weights = weights + self._eta * (A - C)
            # orthonormalization
            weights = row_orthonormalization(weights)
            
            #print("Row orthonormalization takes {}".format(te-ts))
            """
            TODO: check that rows of _weights are orthonormal (difference with eye should be small)
            """ 
            if not self._log_freq and epoch_old < self._epoch:
                self._add_to_train_logs(weights, t)
            elif self._sampling or (self._log_freq > 0 and t % self._log_freq == 0):
                self._add_to_train_logs(weights, t)
            
            #print("One iteration takes {}".format(Te-Ts))
        self.final_weights_ = weights.copy()

    
    def _reset_train_logs(self):
        self._train_log = list()
        self._groundtruth_eval_log = list()
    
    def _add_to_train_logs(self, weights, t):
        self._train_mse_log.append(eval_mse_loss(self._train_data, weights))
        self._groundtruth_eval_log.append(eval_with_groundtruth(self._groundtruth, weights))
        print(f"The loss at the {self._epoch}-th epoch {t}-th iteration is {self._groundtruth_eval_log[-1]}")
        
    
    
    def _get_mini_batch(self, mb_iter, t):
        if mb_iter + self._mbsize < len(self._train_data):
            mini_batch = self._train_data[mb_iter : mb_iter+self._mbsize, :]
            mb_iter_new = mb_iter + self._mbsize
        else:
            self._epoch += 1
            print(f"Finished training {self._epoch}-th epoch with total {t} iterations")
            print(f"The current learning rate is {self._eta}")
            mb_iter_new = (mb_iter + self._mbsize) % (len(self._train_data)-1) - 1
            mini_batch_1 = self._train_data[mb_iter :, :]
            mini_batch_2 = self._train_data[: mb_iter_new, :]
            if mini_batch_2.size > 0:
                mini_batch = mini_batch_1
            else:
                mini_batch = np.concatenate((mini_batch_1, mini_batch_2), axis=0)
        return mini_batch, mb_iter_new
    
    def _get_mini_batch_from_sampling(self):
        rand_idx = np.random.randint(self._train_data_size[0], size=self._mbsize)
        return self._train_data[rand_idx,:]
    


# def row_orthonormalization(weights):
#     """
#     Description: implements Gram-schmidt on rows of the tf weights matrix
#     Input: weights as a tf variable: k by d (k <= d)
#     Return: orthonormalize operation in tf graph
#     """
#     assert weights.shape[0] <= weights.shape[1], "k cannot exceed d!"
#     # add batch dimension for matmul
#     #print(weights.shape, weights[0,:].shape)
#     #ortho_weights = np.expand_dims(weights[0,:]/np.linalg.norm(weights[0,:]),0)
#     ortho_weights = weights[0,:].reshape([1,-1]) / np.linalg.norm(weights[0,:])
#     #print(ortho_weights.shape)
#     #print(weights[0:10])
#     #ortho_weights = weights[0,:]/np.linalg.norm(weights[0,:])
#     #print(ortho_weights.shape)
#     for i in range(1, weights.shape[0]):
#         v = weights[i,:]
#         #print(v.shape)
#         #print(v.shape)
#         # add batch dimension for matmul
#         #v = np.expand_dims(v, 0) 
#         v = v.reshape([1,-1])
#         #print(v.shape)
#         #r = v - np.matmul(np.matmul(v, ortho_weights.T), ortho_weights)
#         #print(np.squeeze(np.matmul(v, ortho_weights.T)))
#         r = v - np.dot(np.squeeze(np.matmul(v, ortho_weights.T)), ortho_weights)
#         ortho_weights = np.concatenate([ortho_weights, r/np.linalg.norm(r)], axis=0)
#     return ortho_weights

def row_orthonormalization(weights):
    ortho_weights_, R = np.linalg.qr(weights.T)
    return ortho_weights_.T


def eval_with_groundtruth(groundtruth, weights):
    """
    Input: 
        groundtruth, list in the form [2, 1.4, 1, 0, 0]
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


def get_eigenvecs(groundtruth):
    """
    Input: groundtruth in the form of list [1,1,0]
    Return: np array, eigenvecs: k by d
    """
    d = len(groundtruth)
    eigenvecs = None
    for dim in range(d):
        if groundtruth[dim] > 0:
            eigenvec = np.zeros([1, d])
            eigenvec[:, dim] = 1.0
            if eigenvecs is None:
                eigenvecs = eigenvec
            else:
                eigenvecs = np.concatenate((eigenvecs, eigenvec), axis=0)
        else:
            break
    return eigenvecs


def eval_mse_loss(batch_data, _weights):
    """
    Input: np array, batch_data: n by d, 
           np array, weights: k by d
    """
    projection = np.matmul(batch_data, _weights.T)
    xhat = np.matmul(projection, _weights)
    #print(xhat.shape)
    return np.sum(np.square(batch_data-xhat))


def check_orthonormality(weights):
    """
    TODO: Check that rows of weights are nearly orthonormal
    """
    pass
        
def get_default_learning_rate(X):
    rbar = np.linalg.norm(X, ord='fro')
    return 1/rbar * (len(X) ** 0.5)
    
## initializers

def get_random_orthogonal_initializer(k, d, gain=1, seed=None):
    tf.reset_default_graph()
    init_fn = tf.orthogonal_initializer(gain, seed, dtype=tf.float64)
    init_weights = tf.get_variable('init_weights', initializer=init_fn, shape=[k, d])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())   
        _init_weights = sess.run([init_weights])[0]
    return _init_weights    
