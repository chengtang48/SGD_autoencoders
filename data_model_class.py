import numpy as np
from scipy.stats import ortho_group #generator for random orthogonal matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_mldata
from keras.datasets import cifar10
import matplotlib.pyplot as plt
#####
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
    mnist = fetch_mldata('MNIST original')
    #print(mnist.data[0])
    ridx = np.random.choice(len(mnist.target), size=batch_size)
    batch_data = mnist.data[ridx]
    batch_data = contrast_normalize(batch_data)
    batch_data = whitening(contrast_normalize(batch_data), transform='zca')
    assert not np.isnan(batch_data).any(), 'mini batch contains illegal value!'
    return list(batch_data)

# def batch_norb_data_generator():
#     pass

def batch_cifar10_data_generator(batch_size, filter_size):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() # data shape n_samples by 32-32-3 (documentation in keras wrong)
    batch_data = list()
    train_size = len(x_train)
    (dim_x, dim_y) = x_train.shape[1], x_train.shape[2]
    for count in range(batch_size):
        #print('x dim and y dim are %d and %d'%(dim_x, dim_y))
        batch_data.append(sample_patch(x_train[count%train_size], dim_x, dim_y, filter_size))
    batch_data = contrast_normalize(np.array(batch_data))
    batch_data = whitening(batch_data, transform='zca')
    return list(batch_data)

def test_plot(filter_size):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() # data shape n_samples by 32-32-3 (documentation in keras wrong)
    (dim_x, dim_y) = x_train.shape[1], x_train.shape[2]
    fig, ax = plt.subplots(5,5)

    for i in range(5):
        for j in range(5):
            patch = sample_patch(x_train[0], dim_x, dim_y, filter_size)
            ax[i,j].set_axis_off()
            ax[i,j].imshow(patch.reshape(filter_size, filter_size, 3))
    plt.show()


###### random sampling
def sample_patch(data_pt, dim_x, dim_y, filter_size):
    ridx_x = np.random.choice(dim_x-filter_size, size=1)[0]
    ridx_y = np.random.choice(dim_y-filter_size, size=1)[0]
    #print(data_pt.shape)
    data_downsampled = data_pt[ridx_x:(ridx_x+filter_size), ridx_y:(ridx_y+filter_size), :]

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
    #V,d,V_T = np.linalg.svd(covariance, full_matrices=True)
    d, V = np.linalg.eigh(covariance)
    offset = 10e-5
    if transform == 'pca':
        return np.matmul(np.matmul(batch_data, V), np.diag(1.0/np.sqrt(d+offset)))
    elif transform == 'zca':
        return np.matmul(np.matmul(np.matmul(batch_data, V), np.diag(1.0/np.sqrt(d+offset))), V.T)
    else:
        print('The method %s is not supported for whitening'%transform)



if __name__ == '__main__':
    test_plot(8)
