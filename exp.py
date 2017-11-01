"""
Usage: exp.py <algo> <n_inits> [--paramname=ARG --paramvalues=<ARGS> --varname=ARG --values=<ARGS>]
"""
from docopt import docopt
from SGD_class import SGD
from scipy.stats import ortho_group #generator for random orthogonal matrix
from variable_definition import autoencoder_ops
import matplotlib.pyplot as plt
import numpy as np
############### Global variables
data_params = {'data_dim': 2, 'train_batch_size':1,
                'model': 'sparse_dict', 'model_params':0.1}

param_inits = {'weights':(2*data_params['data_dim'], 'relu', 0),
               'bias': (None, False, None)
                 }
variable_ops_construction = autoencoder_ops
###############
def initialize_algo(arguments):
    ## parse input args and initialize SGD variants
    gt_dict = ortho_group.rvs(data_params['data_dim']) # generate ground-truth dictionary
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
            data_dim, activation_fn, _ = param_inits['weights']
            param_inits['weights'] = (data_dim, activation_fn, norm)
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
            data_dim, activation_fn, _ = param_inits['weights']
            param_inits['weights'] = (data_dim, activation_fn, norm)
        else:
            norm_default = 2
            data_dim, activation_fn, _ = param_inits['weights']
            param_inits['weights'] = (data_dim, activation_fn, norm_default)
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


    algo = SGD(gt_dict, data_params, param_inits, variable_ops_construction,
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
        algo.width = int(width)
        algo.reinitialize = True
    else:
        print('Variable %s access is not implemented' %varname)
        exit(0)


def train_single(algo, n_runs=10, train_steps=1000, verbose=True):
    # run algorithm n_runs times with same inits (if use_same_init_for_network is True)
    evaluations_over_runs = list()
    for run in range(n_runs):
        evaluations_over_runs.append(algo.train(train_steps, verbose))
    avg_evals = np.mean(np.array(evaluations_over_runs), axis=0)
    std_evals = np.std(np.array(evaluations_over_runs), axis=0)
    return avg_evals, std_evals

def train_all(arguments, n_inits, n_runs=10, train_steps=1000, verbose=True):
    evaluations_over_inits = list()
    for _ in range(n_inits):
        algo  = initialize_algo(arguments)
        evaluations_over_inits.append(train_single(algo, n_runs, train_steps, verbose)[0])
    avg_over_inits = np.mean(np.array(evaluations_over_inits), axis=0)
    std_over_inits = np.std(np.array(evaluations_over_inits), axis=0)
    return avg_over_inits, std_over_inits

def train_against_variable(algo, varname, value_list,
                            n_runs=10, train_steps=1000, verbose=True):
    """
    Run algorithm and report the effect of changing a single variable value
    using a single initialization (if use_same_init_for_network is True)
    """
    evaluations_over_var = list()
    for value in value_list:
        set_algo_states(algo, varname, value)
        evaluations_over_var.append(algo.train(train_steps, verbose))
    # avg_evals = np.mean(np.array(evaluations_over_var), axis=0)
    # std_evals = np.std(np.array(evaluations_over_var), axis=0)
    return evaluations_over_var

def train_all_against_variables(n_inits, arguments, varname, value_list,
                            n_runs=10, train_steps=1000, verbose=True):
    """
    Run train_against_variable n_inits times and averaging over different
    random initializations
    """
    evaluations_over_inits = list()
    for _ in range(n_inits):
        algo  = initialize_algo(arguments)
        evaluations_over_inits.append(train_against_variable(algo, varname, value_list,
                                    n_runs, train_steps, verbose))
    avg_over_inits = np.mean(np.array(evaluations_over_inits), axis=0)
    std_over_inits = np.std(np.array(evaluations_over_inits), axis=0)
    return avg_over_inits, std_over_inits

def train_and_plot(algo, varname, value_list,
                            n_runs=10, train_steps=1000, verbose=True):
    evaluations_over_var = train_against_variable(algo, varname, value_list,
                                n_runs=n_runs, train_steps=train_steps, verbose=verbose)
    cprime, t_o = algo.eta_params
    x = range(t_o, t_o+train_steps+1)
    appx_func = [40/float(t) for t in x]
    fig, ax = plt.subplots()
    for idx, y in enumerate(evaluations_over_var):
        string = ('%s = %s' %(varname, str(value_list[idx])))
        ax.plot(x, y, label=string)
    # ax.plot(time_pts, avg1[:101])
    # ax.plot(time_pts, avg2[:101])
    #ax.errorbar(x, y, yerr=yerr, errorevery=200, label=arguments['<algo>'])
    ax.plot(x, appx_func, label='theoretical bound')
    ax.set_title('squared sin loss vs %s' %varname)
    ax.legend(loc=1)
    plt.show()

def train_all_and_plot(n_inits, arguments, varname, value_list,
                            n_runs=10, train_steps=1000, verbose=True):
    y_array, y_err_array = train_all_against_variables(n_inits, arguments, varname, value_list,
                                n_runs=n_runs, train_steps=train_steps, verbose=verbose)
    cprime, t_o = algo.eta_params
    x = range(t_o, t_o+train_steps+1)
    appx_func = [40/float(t) for t in x]
    fig, ax = plt.subplots()
    y = y_array[:,-1]
    if varname == 'learn_rate':
        myxticks = value_list
        print(myxticks)
        x = range(len(myxticks))
        ax.set_xticks(x)
        ax.set_xticklabels(myxticks)
        ax.scatter(x, y)
    else:
        ax.scatter(value_list, y)
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
    vvalues = '_'.join(value_list)
    fig.savefig(arguments['<algo>']+'_'+arguments['--paramname']
                     +'_'+pvalues+'_'+varname+'_'+vvalues+'.eps')





if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(arguments['--paramvalues'])
    algo = initialize_algo(arguments)
    if '--varname' in arguments:
        varname = arguments['--varname']
        value_list = arguments['--values']
        if varname != 'learn_rate':
            value_list = [x for x in value_list.split(',')]
        else:
            value_list = [x for x in value_list.split(':')]
            #print('value list', value_list)
        #train_and_plot(algo, varname, value_list, n_runs=10, train_steps=1000, verbose=False)
        n_inits = arguments['<n_inits>']
        train_all_and_plot(int(n_inits), arguments, varname, value_list,
                                    n_runs=10, train_steps=1000, verbose=False)
    ## run the algorithm with n_inits number of different random inits
