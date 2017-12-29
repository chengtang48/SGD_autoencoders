"""
Usage: exp.py <algo> <n_inits> [--paramname=ARG --paramvalues=<ARGS> --varname=ARG --values=<ARGS>]
"""
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import functools
import os
from six.moves import cPickle as pickle
from SGD_class import SGD
from scipy.stats import ortho_group #generator for random orthogonal matrix
from variable_definition import autoencoder_ops
from data_model_class import batch_sparse_dict_model_generator, batch_mnist_data_generator, batch_cifar10_data_generator
############### Global variables
data_params = {'data_dim': 2, 'train_batch_size':500,
                'model': 'mnist', 'model_params':8}
#real_data_params = {'train_batch_size':1, 'model': 'cifar10', 'model_params':8}

param_inits = {'weights':(2*data_params['data_dim'], 'relu', 2),
               #'bias': (500, False, 500)
               'bias':(None, False, None)
                 }
# param_inits = {'weights':(4, 'relu', 0),
#                'bias': (None, False, None)}
variable_ops_construction = autoencoder_ops
###############
def initialize_algo(arguments, use_real=False):
    ## parse input args and initialize SGD variants
    if arguments['--paramname'] == 'data_dim':
        data_params['data_dim'] = int(arguments['--paramvalues'])
    if arguments['--paramname'] == 'model':
        data_params['model'] = arguments['--paramvalues']
    if arguments['--paramname'] == 'model_params':
        if data_params['model']=='sparse_dict':
            data_params['model_params'] = float(arguments['--paramvalues'])
        else:
            data_params['model_params'] = int(arguments['--paramvalues'])
    gt_dict = None
    if not use_real:
        gt_dict = ortho_group.rvs(data_params['data_dim']) # generate ground-truth dictionary
    else:
        _, activation_fn, norm = param_inits['weights']
        if data_params['model']=='cifar10':
            data_params['data_dim'] = 192 # 3 by 8 by 8
            param_inits['weights'] = 5*data_params['data_dim'], activation_fn, norm
        elif data_params['model']=='mnist':
            data_params['data_dim'] = 784 # 28 by 28
            param_inits['weights'] = 1000, activation_fn, norm
    if arguments['<algo>'] == 'original':
        if arguments['--paramname'] == 'learn_rate':
            argvalues = arguments['--paramvalues']
            argvalues = argvalues.split(',')
            c_prime, t_o = float(argvalues[0]), int(argvalues[1])
        else:
            # use default value
            c_prime = 10
            t_o = 100
        if arguments['--paramname'] == 'norm':
            ## norm value can take 0, meaning no control
            norm = float(arguments['--paramvalues'])
            #global param_inits
            width, activation_fn, _ = param_inits['weights']
            param_inits['weights'] = (width, activation_fn, norm)
        if arguments['--paramname'] == 'batch_size':
            if len(arguments['--paramvalues'])==1:
                mb_size = int(arguments['--paramvalues'])
            else:
                print('There can be at most 1 arguments for batch_size')
                exit(0)
            #global data_params
            data_params['train_batch_size'] = mb_size

    elif arguments['<algo>'] == 'norm_controlled':
        if arguments['--paramname'] == 'learn_rate':
            argvalues = arguments['--paramvalues']
            argvalues = argvalues.split(',')
            c_prime, t_o = float(argvalues[0]), int(argvalues[1])
        else:
            # use default value
            c_prime = 10
            t_o = 100
        if arguments['--paramname'] == 'norm':
            ## norm value can take 0, meaning no control
            norm = float(arguments['--paramvalues'])
            #global param_inits
            width, activation_fn, _ = param_inits['weights']
            param_inits['weights'] = (width, activation_fn, norm)
        else:
            norm_default = 2
            width, activation_fn, _ = param_inits['weights']
            param_inits['weights'] = (width, activation_fn, norm_default)
            print('Norm set to default value %f' %norm_default)

        if arguments['--paramname'] == 'batch_size':
            '''
            format of batch_size args
                mbsize,init_b_batch_size-use_mini_batch-bbatch_size
            NOTE: use_mini_batch here must be fed as 0 or 1 (not string)
            '''
            ## parse params
            argvalues = arguments['--paramvalues']
            argvalues = argvalues.split(',')
            assert len(argvalues)==2, 'There must be 2 arguments for batch_size'
            mb_size, b_params = int(argvalues[0]), argvalues[1]
            #global data_params
            data_params['train_batch_size'] = mb_size
            # parse b_params
            b_params = b_params.split('-')
            #print('Use mini batch for b?',bool(int(b_params[1])))
            param_inits['bias'] = (int(b_params[0]), bool(int(b_params[1])), int(b_params[2]))

    algo = SGD(data_params, param_inits, variable_ops_construction, gt_dict=gt_dict,
                use_same_init_for_network=True, loss='squared',
                evaluation_metric=None, eta='decay', c_prime=c_prime, t_o=t_o)
    return algo



def set_algo_states(algo, varname, value):
    """
    Example: 'norm', 2
    """
    if varname == 'norm':
        algo.norm = float(value)
    elif varname == 'learn_rate':
        print
        value_tuple = value.split(',')
        algo.eta_params = (float(value_tuple[0]), int(value_tuple[1])) #must be of format (c_prime, t_o)
    elif varname == 'mini_batch_size':
        algo.train_batch_size = int(value)
    elif varname == 'bbatch_size':
        algo.bbatch_size = int(value)
    elif varname == 'init_b_batch_size':
        algo.init_b_batch_size = int(value)
        algo.reinitialize = True
    elif varname == 'width':
        algo.width = int(value)
        algo.reinitialize = True
    else:
        print('Variable %s access is not implemented' %varname)
        exit(0)

#############################################
############# Visualize filter activation pattern
def train_and_plot_activation_hist(algo, test_size, train_steps=1000, verbose=True):
    (weights, bias) = algo.train(train_steps, verbose)[1], algo.train(train_steps, verbose)[2]
    print('bias', bias)
    def get_per_data_activation(data_pt):
        return np.max(np.matmul(weights, data_pt)+bias, 0)
    ###
    data_model = algo.data_params['model']
    if data_model == 'sparse_dict':
        test_data = batch_sparse_dict_model_generator(algo.data_params['data_dim'],
                          algo.data_params['model_params'], test_size, gt_dict=algo.gt_dict)
    elif data_model == 'mnist':
        test_data = batch_mnist_data_generator(test_size)
    elif data_model == 'cifar10':
        test_data = batch_cifar10_data_generator(test_size, algo.data_params['model_params'])

    hist_array = 1.0/len(test_data)*functools.reduce(lambda a,b: np.add(a,b), list(map(get_per_data_activation, test_data)))
    ## plot histogram
    plt.hist(hist_array)
    plt.show()


#############################################
############## training on real data
def train_and_visualize_dict(algo, train_steps=1000, verbose=True):
    learned_weights = algo.train(train_steps, verbose)[1]
    if algo.data_params['model'] == 'cifar10':
        filter_size = int(algo.data_params['model_params'])
    elif algo.data_params['model'] == 'mnist':
        filter_size = 28
    else:
        print('Dataset %s is not implemented in our experiment' %algo.data_params['model'])
    print('filter size %f' %filter_size)
    if algo.data_params['model'] == 'cifar10':
        learned_filters = np.reshape(learned_weights, [len(learned_weights), filter_size, filter_size, -1])
    elif algo.data_params['model'] == 'mnist':
        learned_filters = np.reshape(learned_weights, [len(learned_weights), filter_size, filter_size])
    ## plot all filters
    #fig, ax = plt.subplots(12,8)
    x_length = filter_size
    y_length = filter_size
    fig, ax = plt.subplots(x_length, y_length)
    idx = 0
    for i in range(x_length):
        for j in range(y_length):
            ## use RGB channels
            ax[i,j].set_axis_off()
            if algo.data_params['model'] == 'mnist':
                ax[i,j].imshow(learned_filters[idx], cmap='gray')
            elif algo.data_params['model'] == 'cifar10':
                ax[i,j].imshow(learned_filters[idx])
            #print(learned_filters[idx,:,:,0])
            idx += 1
    fig.savefig('test_fig.png')


#############################################
############## training on simulated data

def train_over_runs(algo, n_runs=10, train_steps=1000, verbose=True):
    # run algorithm n_runs times with same inits (if use_same_init_for_network is True)
    evaluations_over_runs = list()
    for run in range(n_runs):
        evaluations_over_runs.append(algo.train(train_steps, verbose)[0])
    avg_evals = np.mean(np.array(evaluations_over_runs), axis=0)
    std_evals = np.std(np.array(evaluations_over_runs), axis=0)
    return avg_evals, std_evals

def train_n_inits_over_runs(arguments, n_inits, n_runs=10, train_steps=1000, verbose=True):
    # run algorithm n_runs with n_inits different initializations
    evaluations_over_inits = list()
    for _ in range(n_inits):
        algo  = initialize_algo(arguments)
        evaluations_over_inits.append(train_over_runs(algo, n_runs, train_steps, verbose)[0])
    avg_over_inits = np.mean(np.array(evaluations_over_inits), axis=0)
    std_over_inits = np.std(np.array(evaluations_over_inits), axis=0)
    return avg_over_inits, std_over_inits

def train_n_runs_against_variable(algo, varname, value_list, n_runs=10, train_steps=1000, verbose=True):
    """
    Run algorithm and report the effect of changing a single variable value
    using a single initialization (if use_same_init_for_network is True)
    """
    evaluations_over_var = list()
    for value in value_list:
        set_algo_states(algo, varname, value)
        evaluations_over_var.append(train_over_runs(algo, n_runs, train_steps, verbose)[0])
    # avg_evals = np.mean(np.array(evaluations_over_var), axis=0)
    # std_evals = np.std(np.array(evaluations_over_var), axis=0)
    return evaluations_over_var

def train_n_inits_against_variables(n_inits, arguments, varname, value_list,
                                  n_runs=10, train_steps=1000, verbose=True):
    """
    Run train_one_against_variable n_inits times and averaging over different
    random initializations
    """
    evaluations_over_inits = list()
    for _ in range(n_inits):
        algo  = initialize_algo(arguments)
        evaluations_over_inits.append(train_n_runs_against_variable(algo, varname, value_list,
                                    n_runs, train_steps, verbose))
    avg_over_inits = np.mean(np.array(evaluations_over_inits), axis=0)
    std_over_inits = np.std(np.array(evaluations_over_inits), axis=0)
    return avg_over_inits, std_over_inits

def train_n_inits_against_variables_over_runs(n_inits, arguments, varname, value_list,
                                  n_runs=10, train_steps=1000, verbose=True):
    evaluations_over_var = list()
    arguments['--paramname'] = varname
    for value in value_list:
        arguments['--paramvalues'] = value
        evaluations_over_var.append(train_n_inits_over_runs(arguments,n_inits,n_runs,train_steps,verbose))
    # avg_evals = np.mean(np.array(evaluations_over_var), axis=0)
    # std_evals = np.std(np.array(evaluations_over_var), axis=0)
    print(evaluations_over_var)
    return evaluations_over_var


def train_and_plot(algo, varname, value_list,
                            n_runs=10, train_steps=1000, verbose=True):
    evaluations_over_var = train_n_runs_against_variable(algo, varname, value_list,
                                n_runs=n_runs, train_steps=train_steps, verbose=verbose)
    cprime, t_o = algo.eta_params
    x = range(t_o, t_o+train_steps+1)
    appx_func = [40/float(t) for t in x]
    fig, ax = plt.subplots()
    for idx, y in enumerate(evaluations_over_var):
        string = ('%s = %s' %(varname, str(value_list[idx])))
        ax.plot(value_list, y, label=string)
    # ax.plot(time_pts, avg1[:101])
    # ax.plot(time_pts, avg2[:101])
    #ax.errorbar(x, y, yerr=yerr, errorevery=200, label=arguments['<algo>'])
    ax.plot(x, appx_func, label='theoretical bound')
    ax.set_title('squared sin loss vs %s' %varname)
    ax.legend(loc=1)
    plt.show()

def train_all_and_plot(n_inits, arguments, varname, value_list,
                            n_runs=10, train_steps=1000, verbose=True):
    # y_array, y_err_array = train_n_inits_against_variables(n_inits, arguments, varname, value_list,
    #                             n_runs=n_runs, train_steps=train_steps, verbose=verbose)
    eval_over_values = train_n_inits_against_variables_over_runs(n_inits, arguments, varname, value_list,
                                      n_runs=10, train_steps=1000, verbose=True)
    avg_over_values, std_over_values = zip(*eval_over_values)
    cprime, t_o = algo.eta_params
    x = range(t_o, t_o+train_steps+1)
    appx_func = [40/float(t) for t in x]
    fig, ax = plt.subplots()
    y = np.array(avg_over_values)[:,-1]
    y_err = np.array(std_over_values)[:,-1]
    if varname == 'learn_rate':
        myxticks = value_list
        print(myxticks)
        x = range(len(myxticks))
        ax.set_xticks(x)
        ax.set_xticklabels(myxticks)
        ax.scatter(x, y)
    else:
        #ax.scatter(value_list, y)
        ax.plot(value_list, y, 'ro-')
        ax.set_title(varname)
    # for idx, y in enumerate(y_array.tolist()):
    #     string = ('%s = %s' %(varname, str(value_list[idx])))
    #     y_err = y_err_array[idx]
    #     ax.errorbar(x, y, yerr=y_err, errorevery=200, label=string)
    # ax.plot(time_pts, avg1[:101])
    # ax.plot(time_pts, avg2[:101])
    #ax.errorbar(x, y, yerr=yerr, errorevery=200, label=arguments['<algo>'])
    #ax.plot(x, appx_func, label='theoretical bound')
    ax.set_title('squared sin loss vs %s' %varname)
    #ax.legend(loc=1)
    #plt.show()
    pvalues = arguments['--paramvalues']
    pvalues_list = pvalues.split(',')
    pvalues = '_'.join(pvalues_list)
    if varname == 'learn_rate':
        value_list = ['-'.join(v.split(',')) for v in value_list]
    vvalues = '_'.join(value_list)
    fig.savefig(arguments['<algo>']+'_'+arguments['--paramname']
                     +'_'+pvalues+'_'+varname+'_'+vvalues+'.eps')

def train_all_and_pickle(n_inits, arguments, varname, value_list,
                            n_runs=10, train_steps=1000, verbose=True):
    # y_array, y_err_array = train_n_inits_against_variables(n_inits, arguments, varname, value_list,
    #                             n_runs=n_runs, train_steps=train_steps, verbose=verbose)
    eval_over_values = train_n_inits_against_variables_over_runs(n_inits, arguments, varname, value_list,
                                      n_runs, train_steps, verbose)

    pvalues = arguments['--paramvalues']
    pvalues_list = pvalues.split(',')
    pvalues = '_'.join(pvalues_list)
    if varname == 'learn_rate':
        value_list = ['-'.join(v.split(',')) for v in value_list]
    vvalues = '_'.join(value_list)
    width = param_inits['weights'][0]
    fname = arguments['<algo>']+'_'+arguments['--paramname']+'_'+pvalues+'_'+varname+'_'+vvalues+'width'+str(width)
    _ = maybe_pickle(fname, data=eval_over_values, force=True)


##################################
# Utility function: pickle or get pickled data with desired dataname
def maybe_pickle(dataname, data = None, force = False, verbose = True):
    """
    Process and pickle a dataset if not present
    """
    filename = dataname + '.pickle'
    if force or not os.path.exists(filename):
        # pickle the dataset
        print('Pickling data to file %s' % filename)
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save to', filename, ':', e)
    else:
        print('%s already present - Skipping pickling.' % filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f)

    return data





if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(arguments['--paramvalues'])
    algo = initialize_algo(arguments, use_real=True)
    if arguments['--values']:
        print('pass')
        varname = arguments['--varname']
        value_list = arguments['--values']
        if varname != 'learn_rate':
            value_list = [x for x in value_list.split(',')]
        else:
            value_list = [x for x in value_list.split(':')]
            #print('value list', value_list)
        #train_and_plot(algo, varname, value_list, n_runs=10, train_steps=1000, verbose=False)
    n_inits = arguments['<n_inits>']
        # train_all_and_plot(int(n_inits), arguments, varname, value_list,
        #                             n_runs=10, train_steps=1000, verbose=False)
        # train_all_and_pickle(int(n_inits), arguments, varname, value_list,
        #                             n_runs=10, train_steps=1000, verbose=False)
    #train_and_visualize_dict(algo, train_steps=1000, verbose=True)
    train_and_plot_activation_hist(algo, 1000, train_steps=1000, verbose=True)
    ## run the algorithm with n_inits number of different random inits
