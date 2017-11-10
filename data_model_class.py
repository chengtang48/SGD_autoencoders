import numpy as np
from scipy.stats import ortho_group #generator for random orthogonal matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_mldata
from keras.datasets import cifar10

class DataGenerator(object):
    def __init__(self, data_params, gt=None):
        self.data_dim = data_params['data_dim']
        self.model_name = data_params['model']
        self.model_params = data_params['model_params']
        self.gt = gt
        if self.model_name == 'sparse_dict':
            assert self.model_params, 'Please provide parameters for model construction!'
        elif self.model_name == 'cifar10':
            assert self.model_params, 'Please provide filter size'

    def __call__(self, batch_size):
        if self.model_name == 'sparse_dict':
            return batch_sparse_dict_model_generator(self.data_dim, self.model_params, batch_size, gt_dict=self.gt)
        elif self.model_name == 'cifar10':
            return batch_cifar10_data_generator(batch_size, self.model_params)
        elif self.model_name == 'mnist':
            return batch_mnist_data_generator(batch_size)



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

def batch_mnist_data_generator(batch_size):
    mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
    ridx = np.random.choice(len(mnist.target), size=batch_size)
    return list(mnist.data[ridx])

# def batch_norb_data_generator():
#     pass

def batch_cifar10_data_generator(batch_size, filter_size):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    batch_data = list()
    train_size = len(x_train)
    (dim_x, dim_y) = x_train.shape[2], x_train.shape[3]
    for count in range(batch_size):
        batch_data.append(sample_patch(x_train[count%train_size], dim_x, dim_y, filter_size))
    return batch_data



###### random sampling
def sample_patch(data_pt, dim_x, dim_y, filter_size):
    ridx_x = np.random.choice(dim_x-filter_size, size=1)
    ridx_y = np.random.choice(dim_y-filter_size, size=1)
    data_downsampled = data_pt[:, ridx_x, ridx_y]

    return data_downsampled.flatten()


###### preprocessing utilities
def contrast_normalize(batch_data):
    """
    Input dimension: n_data by n_dim
    Output dimension: n_data by n_dim
    """
    normalized_batch = batch_data - np.mean(batch_data, axis=0)# centering
    return normalized_batch / np.std(normalized_batch)

def whitening(batch_data, transform='pca'):
    covariance = 1.0/len(batch_data)*np.matmul(np.transpose(batch_data), batch_data)
    V,d,V_T = np.linalg.svd(covariance, full_matrices=True)
    if transform == 'pca':
        return np.matmul(np.matmul(batch_data, V), np.power(d, -0.5))
    elif transform == 'zca':
        return np.matmul(np.matmul(np.matmul(batch_data, V), np.power(d, -0.5)), V)
    else:
        print('The method %s is not supported for whitening'%transform)
