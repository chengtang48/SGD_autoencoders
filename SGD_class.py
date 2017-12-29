import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from data_model_class import DataGenerator

class SGD(object):
	"""
	Example
	param_inits = {'weights':(width, activation_fn, control_norm)
				   'bias': (init_b_batch_size,use_mini_batch=False, bbatch_size=None)
	               }
    data_params = {'data_dim': 3, 'train_batch_size':1,
					'model': 'sparse_dict', 'model_params':noise_bound
				   }

	"""
	def __init__(self, data_params, param_inits,
					 variable_ops_construction, gt_dict=None, use_same_init_for_network=True,
	                 loss='squared', evaluation_metric='None',
					 eta=None, c_prime=None, t_o=None):
		tf.reset_default_graph()
		self.data_generator = DataGenerator(data_params)
		self.train_batch_size = data_params['train_batch_size']
		self.gt_dict = gt_dict
		## network parameters init
		self.param_inits = param_inits
		self.data_params = data_params
		#self.control_norm = param_inits['weights']
		(self.init_b_batch_size,self.use_mini_batch, self.bbatch_size) = param_inits['bias']
		self.bias_trainable = True
		if bool(self.bbatch_size) or self.use_mini_batch:
			self.bias_trainable = False

		self.width, self.activation_fn, self.norm = self.param_inits['weights']
		if bool(self.norm):
			self.rescale_param = 1/float(self.norm**2) - 1
		## initialze weight and bias
		self.init_weights = None
		self.init_bias = None
		self.use_same_init_for_network = use_same_init_for_network
		# if use_same_init_for_network:
		# 	(self.weights, self.init_weights_, self.bias_init, self.init_bias_,
		#        self.xhat, self.place_holder) = self.initialize_variables(variable_ops_construction,
		#                                      init_b_batch_size, use_bias_init)
		# else:
		# 	(self.weights, init_weights_, self.bias_init, init_bias_,
		#        self.xhat, self.place_holder) = self.initialize_variables(variable_ops_construction,
		#                                      init_b_batch_size, use_bias_init)
		self.initialize_parameters() #get self.init_weights_ and self.init_bias_
		self.variable_ops_construction = variable_ops_construction
		self.loss = loss
		if not evaluation_metric:
			self.evaluation_metric = cos_sq_avg_distances
		### learning rate params
		self.eta_fn = eta
		self.eta_params = (c_prime, t_o)
		### flag: whether to re-initialize at train
		self.reinitialize = False



	def initialize_parameters(self):
		if not self.use_same_init_for_network:
			self.init_weights_ = None
			self.init_bias_ = None
		else:
			init_weights = tf.random_normal([self.width, self.data_params['data_dim']], dtype=tf.float64) # init random weights
			self.init_bias_ = None
	        if not bool(self.init_b_batch_size):
				## if sample based init is disabled
	            self.init_bias_ = np.zeros(self.width)
	        else:
				init_batch_data = self.data_generator(self.init_b_batch_size)
				assert self.norm, 'Norm is not provided to initialize bias!'
				init_bias = bias_init(init_weights, self.norm, init_batch_data)

			## parameter initialization
	        with tf.Session() as sess:
	            self.init_weights_ = sess.run(init_weights)
	            if self.init_bias_ is None:
	                self.init_bias_ = sess.run(init_bias)

	def add_variables_to_graph(self):
		## update parameters maintained by algorithm
		self.param_inits['weights'] = self.width, self.activation_fn, self.norm
		self.param_inits['bias'] = self.init_b_batch_size,self.use_mini_batch, self.bbatch_size
		self.data_params['train_batch_size'] = self.train_batch_size
		return self.variable_ops_construction(self.param_inits['weights'], self.data_params,
		            bias_trainable=self.bias_trainable, init_weights=self.init_weights_ ,
					init_bias=self.init_bias_)

	def get_eta(self):
		if not self.eta_fn:
			eta = 0.001
			self.global_step = tf.Variable(0, trainable=False)
		elif self.eta_fn == 'decay':
			c_prime, t_o = self.eta_params
			learning_rate = c_prime
			decay_steps = 1
			decay_rate = 1.0
			self.global_step = tf.Variable(t_o, trainable=False)
			eta = tf.train.inverse_time_decay(tf.cast(learning_rate, tf.float64), self.global_step, decay_steps, decay_rate)
		else:
			self.global_step = tf.Variable(t_o, trainable=False)
			eta = self.eta_fn(self.global_step)
		return eta


	def get_train_op(self):
		return tf.train.GradientDescentOptimizer(self.get_eta()).minimize(get_loss(self.x, self.xhat))

	def get_update_bias_op(self):
		#width,_,_ = self.param_inits['weights']
		#(_, _, _, update_b_batch_size) = self.param_inits['bias']
		#tf.placeholder(tf.float64, [self.bbatch_size, dim])
		if not self.use_mini_batch:
			assert bool(self.bbatch_size), 'Please provide batch size to update bias'
			## get a fresh mini-batch sample
			bbatch_size = self.bbatch_size
		else:
			#use the same mini_batch as for weights
			bbatch_size = self.train_batch_size
		bbatch_x = tf.placeholder(tf.float64, [bbatch_size, self.data_params['data_dim']])
		new_bias = get_bias_update(bbatch_x, self.weights, self.bias, self.width,
		                            bbatch_size, self.rescale_param)
		return tf.assign(self.bias, new_bias), bbatch_x

	def train(self, train_steps, verbose=True):
		if self.reinitialize:
			self.initialize_parameters()
		evaluations = list()
		## add variables to computation graph with desired intial values
		self.weights, self.init_weights_, self.bias, self.init_bias_, self.xhat, self.x = self.add_variables_to_graph()
		#print('initial weights', self.init_weights_)
		## add initial error
		if self.gt_dict is not None:
			print(self.gt_dict)
			score_init = self.evaluation_metric(self.init_weights_, self.gt_dict)
			evaluations.append(score_init)
			print('Evaluation at init %f' %score_init)

		train_op = self.get_train_op() ## define optimization op
		if bool(self.bbatch_size) or self.use_mini_batch:
			bias_update_op, bbatch_x = self.get_update_bias_op() ## define operation for bias update
		increment_global_step_op = get_increment_global_step_op(self.global_step)
		if bool(self.norm):
			## define row normalization if control_norm is not False (0)
			row_normalize_op = get_row_normalize_op(self.weights, self.norm)
		########
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for step in range(train_steps):
				self.mini_batch = self.data_generator(self.train_batch_size)
				#print('mini-batch shape', self.mini_batch[0].shape)
				#init_weights_ = sess.run(init_weights)
				_, weights_, bias_ = sess.run([train_op,
				       self.weights, self.bias], feed_dict={self.x: self.mini_batch})
				if bool(self.norm):
					_, weights_ = sess.run([row_normalize_op, self.weights])

				## update bias using designed method if newton update is enabled
				if self.use_mini_batch or bool(self.bbatch_size):
					if self.use_mini_batch:
						_, bias_ =sess.run([bias_update_op, self.bias],
                              feed_dict={bbatch_x: self.mini_batch})
					else:
						# fresh sample
						bbatch = self.data_generator(self.bbatch_size)
						_, bias_ =sess.run([bias_update_op, self.bias],
                              feed_dict={bbatch_x: bbatch})
				n_steps = sess.run(increment_global_step_op)
				if self.gt_dict:
					score_now = self.evaluation_metric(weights_, self.gt_dict)
					evaluations.append(score_now)
				if verbose:
					if train_steps > 20:
						epoch = train_steps / 20
						if step % epoch == 0:
							print('Training at %d th epoch' %(step/epoch+1))
							if self.gt_dict is not None:
								print('Current evaluation: %f' %score_now)
					else:
						print('Training at %d th iteration' %step)
						if self.gt_dict is not None:
							print('Current evaluation: %f' %score_now)
		return evaluations, weights_, bias_

####
def get_loss(x, xhat, loss='squared'):
	if loss == 'squared':
		return tf.reduce_mean(tf.square(x - xhat))
	else:
		print('loss %s not implemented' %loss)

#### Automatic bias update
def get_bias_update(batch_x, weights, bias, width, batch_size, rescale_param):
    if tf.shape(batch_x)==1:
        print('wrong')
    else:
        #projection = tf.matmul(weights, batch_x)    # width by batch_size
		projection = tf.transpose(tf.matmul(batch_x, tf.transpose(weights)))    # width by batch_size
		relu_activation = tf.nn.relu(tf.transpose(tf.add(tf.transpose(projection), bias)))
		zero = tf.constant(0, dtype=tf.float64)
		where = tf.not_equal(relu_activation, zero) ## logical indexing of relu_activation
		indices = tf.where(where) ## indices of nonzero entries in relu_activation
        ## calculate empirical prob of firing
		nnz = tf.cast(tf.count_nonzero(where, axis=1), tf.float64)
		zero_of_nnz = tf.equal(nnz, zero)
		offsetted_nnz = tf.where(zero_of_nnz, tf.ones([width], dtype=tf.float64), nnz)
		prob_firing = tf.divide(nnz, tf.cast(batch_size,tf.float64))

        ## calculate empirical mean of projected value
		shape = tf.constant([width, batch_size], dtype=tf.int64)
        #print(indices.get_shape())
		updated = tf.scatter_nd(indices, tf.gather_nd(projection, indices), shape)
		updated = tf.reduce_sum(updated, axis=1)#
        #offsetted_nnz = tf.scatter_add(nnz, indices_of_zero_in_nnz, tf.ones_like(indices_of_zero_in_nnz))
		updated = tf.divide(updated, offsetted_nnz) * rescale_param ## shape = (width,)
        #print('updated shape', updated.get_shape())
        #print(relu_activation.get_shape())
		new_bias = tf.add(tf.multiply(prob_firing,updated), tf.multiply(tf.subtract(tf.cast(1,tf.float64),prob_firing),bias))
		return new_bias

#### bias initialization
def bias_init(weights, norm, init_batch_data):
    assert norm > 1, 'the norm value is invalid!'
    ## calculate average inner product
    avg_proj = tf.reduce_mean(tf.matmul(weights, tf.transpose(np.array(init_batch_data))), axis=1)
    return tf.subtract(tf.divide(avg_proj, norm**2), avg_proj)


#### Auxiliary functions
def get_increment_global_step_op(global_step):
	return tf.assign(global_step, global_step+1)

def get_row_normalize_op(weights, norm):
	return tf.assign(weights, norm * tf.nn.l2_normalize(weights, dim=1))

## evaluation metric
def cosine_squared_distances(weights, dictionary):
    #print(weights.shape)
    #print(normalize(weights))
    w_normalized = normalize(weights)
    #w_normalized = 0.5*weights
    raw_scores = np.square(np.matmul(w_normalized, np.transpose(dictionary)))
    max_scores = np.max(raw_scores, axis=0) # best approximation for each dict item
    return max_scores, raw_scores, min(max_scores)
def cos_sq_avg_distances(weights, dictionary):
    max_scores,_,_ = cosine_squared_distances(weights, dictionary)
    max_scores = [1]*len(max_scores)-max_scores
    return np.mean(max_scores)
def cos_sq_min_distances(weights, dictionary):
    max_scores,_,_ = cosine_squared_distances(weights, dictionary)
    max_scores = [1]*len(max_scores)-max_scores
    return np.min(max_scores)
