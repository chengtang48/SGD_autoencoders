import numpy as np
from scipy.stats import ortho_group #generator for random orthogonal matrix
from sklearn.preprocessing import normalize

class DataGenerator(object):
    def __init__(self, data_params, gt=None):
        self.data_dim = data_params['data_dim']
        self.model_name = data_params['model']
        self.model_params = data_params['model_params']
        self.gt = gt
        if self.model_name == 'sparse_dict':
            assert self.model_params, 'Please provide parameters for model construction!'


    def __call__(self, batch_size):
        if self.model_name == 'sparse_dict':
            return batch_sparse_dict_model_generator(self.data_dim, self.model_params, batch_size, gt_dict=self.gt)


def sparse_dict_model_generator(dim, noise_bound, gt_dict=None):
    if gt_dict is not None:
        W = gt_dict
    else:
        W = ortho_group.rvs(dim)
    s = np.random.multinomial(1, [float(1/dim)]*dim)
    sigma = float(noise_bound)/(dim)**(0.5)
    eps = np.random.normal([0.0]*dim, sigma)
    norm = np.linalg.norm(eps)
    if norm > noise_bound:
        eps = noise_bound * eps/norm
    return np.dot(W,s)+eps, eps

def batch_sparse_dict_model_generator(dim, noise_bound, batch_size, gt_dict=None):
    def map_function(null):
        return sparse_dict_model_generator(dim, noise_bound, gt_dict)[0]

    return list(map(map_function, [0]*batch_size))
