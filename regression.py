from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import time

import numpy as np
import argparse
from scipy import stats
import gpflowSlim as gfs
from gpflowSlim.neural_kernel_network import NKNWrapper_t, NeuralKernelNetwork
import tensorflow as tf

from utils.create_logger import create_logger
from data import get_data
from utils.functions import median_distance_local, get_kemans_init
from kernels import KernelWrapper
from dppy.finite_dpps import FiniteDPP
import gpflow 
from scipy.linalg import qr
from numpy.random import rand, randn
####################### test inducing_points ######################
# data_fetch = get_data('my_data')
# for i in range(1):
#     data = data_fetch('t_energy', i)
#     x_train_source = data.x_train_s.astype('float32')
#     inducing_points = get_kemans_init(x_train_source, 100)
#     print("inducint points")
#     print(len(inducing_points))
#     print(inducing_points)
#     print(type(inducing_points))

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='6'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Neural-Kernel-Network')
parser.add_argument('--data', type=str, default="boston", help='choose data')
parser.add_argument('--split', type=str, default="uci_woval", help='way to split dataset')
parser.add_argument('--kern', type=str, default='nkn')
args = parser.parse_args()


############################# basic info #############################
N_RUNS=dict(
    my_data = 2,
    airquality =2,
    concrete =2,
    indoor = 2,
    wine = 2,
    boston =2,
    scarcos =1,
    airdelay =1,
    emulate = 1 
)
FLOAT_TYPE = tf.float32 #gfs.settings.float_type
SMALL_DATASETS=['boston', 'concrete', 't_energy', 'wine','airquality','indoor','scarcos','airdelay','emulate']
epochs = 5000 #if args.data in SMALL_DATASETS else 10000

########################## DPP data select ###########################
def dppselect(x_train_source, y_train_source, select_num):
    # r, N = 8, 600

    # # Random orthogonal vectors
    # eig_vecs, _ = qr(randn(N, r), mode='economic')
    # # Random eigenvalues
    # eig_vals = rand(r)  # 0< <1
    # #eig_vals = np.random.randint(2, size=r) # 0 or 1 i.e. projection
    # DPP = FiniteDPP('likelihood',
    #             **{'L': (eig_vecs*(eig_vals/(1-eig_vals))).dot(eig_vecs.T)})
    dim = x_train_source.shape[1]
    k = gpflow.kernels.RBF(dim)
    M = select_num 
    # with tf.Session() as sess:
    kff = k.compute_K_symm(x_train_source)

    DPP = FiniteDPP('likelihood', **{'L': kff})
    DPP.flush_samples()
    DPP.sample_exact_k_dpp(size=M)
    ind = DPP.list_of_samples[0]
    x = x_train_source[ind]
    y = y_train_source[ind]
    return x, y

############################# random data select ######################
def randselect(x_train_source, y_train_source, select_num):
    permutation = np.random.permutation(x_train_source.shape[0])
    x_train_source = x_train_source[permutation[:select_num]]
    y_train_source = y_train_source[permutation[:select_num]]
    return x_train_source, y_train_source

# ############################# my inducing points ####################
def standardize(data):
    std = np.std(data, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data, 0, keepdims=True)
    data_train_standardized = (data-mean) / std
    output = data_train_standardized
    return output

def get_selectpoints(x_train_source, y_train_source, x_train_target, y_train_target, select_num):
    p_x_s = standardize(x_train_source)
    p_y_s = standardize(y_train_source)    
    p_x_t = standardize(x_train_target)
    p_y_t = standardize(y_train_target)
    
    ########## cluster #########
    centers_x = {}
    centers_y = {}
    clf_x = {}
    clf_y = {}
    clf_d_x = {}
    clf_d_y = {}
    cluster_target = len(p_x_t)
    cluster_source = len(p_x_s)
    
    for i in range(cluster_target):
        centers_x[i] = p_x_t[i]
        centers_y[i] = p_y_t[i]
        clf_x[i] = []
        clf_y[i] = []
        clf_d_x[i] = []
        clf_d_y[i] = []
        
    for i in range(cluster_source):
        distances = []
        for center in centers_x:
            distances.append(np.linalg.norm(p_x_s[i] - centers_x[center]))
        min_d = min(distances)
        classification = distances.index(min_d)
        clf_x[classification].append(x_train_source[i])
        clf_y[classification].append(y_train_source[i])
        clf_d_x[classification].append(min_d)
        d_y = np.linalg.norm(p_y_s[i] - centers_y[classification])
        clf_d_y[classification].append(d_y)
               
    ###### select points #######
    select_num = select_num    
    each_cluster_num = []   
    select_points_x = []
    select_points_y = []
    
    for i in range(cluster_target):
        get_num = len(clf_x[i])
        each_cluster_num.append( round(select_num * get_num / cluster_source) )        
        d_x_s = standardize(clf_d_x[i])
        d_y_s = standardize(clf_d_y[i])    
        get_dis = d_x_s * d_y_s   
        norm = get_dis.tolist()
        for j in range(each_cluster_num[i]):
            get_data = norm.index(min(norm))
            select_points_x.append(clf_x[i][get_data])
            select_points_y.append(clf_y[i][get_data])
            norm[get_data] = 10000
    
    select_points_x = np.array(select_points_x)
    select_points_y = np.array(select_points_y)
    return select_points_x, select_points_y

############################## NKN Info ##############################
def NKNInfo(input_dim, ls):
    kernel = dict(
        nkn=[
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'ARD': True,                    'name': 'Linear1'}},
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'ARD': True,                    'name': 'Linear2'}},
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'lengthscales': ls / 6., 'ARD': True,     'name': 'RBF1'}},
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'lengthscales': ls * 2 / 3., 'ARD': True, 'name': 'RBF2'}},
            {'name': 'RatQuad', 'params': {'input_dim': input_dim, 'alpha': 0.1, 'lengthscales': ls / 3.,    'name': 'RatQuad1'}},
            {'name': 'RatQuad', 'params': {'input_dim': input_dim, 'alpha': 1., 'lengthscales': ls / 3.,     'name': 'RatQuad2'}}],
        rbf=[
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'lengthscales': ls, 'ARD': True, 'name': 'RBF1'}}],
        sm1=[{'name': 'SM', 'params': [
            {'w': 1.,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls, 'ARD': True, 'name': 'SM-RBF0'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls, 'ARD': True, 'name': 'SM-Cosine0'}},
        ]}],
        sm2 = [{'name': 'SM', 'params': [
            {'w': 1. / 2,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-RBF0'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-Cosine0'}},
            {'w': 1. / 2,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-RBF1'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-Cosine1'}},
        ]}],
        sm3=[{'name': 'SM', 'params': [
            {'w': 1. / 3,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls * 2., 'ARD': True, 'name': 'SM-RBF0'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls * 2., 'ARD': True, 'name': 'SM-Cosine0'}},
            {'w': 1. / 3,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-RBF1'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-Cosine1'}},
            {'w': 1. / 3,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-RBF2'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-Cosine2'}},
        ]}],
        sm4=[{'name': 'SM', 'params': [
            {'w': 1. / 4,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls / 4., 'ARD': True, 'name': 'SM-RBF0'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls / 4., 'ARD': True, 'name': 'SM-Cosine0'}},
            {'w': 1. / 4,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-RBF1'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-Cosine1'}},
            {'w': 1. / 4,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-RBF2'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-Cosine2'}},
            {'w': 1. / 4,
              'rbf': {'input_dim': input_dim, 'lengthscales': ls * 2,  'ARD': True, 'name': 'SM-RBF3'},
              'cos': {'input_dim': input_dim, 'lengthscales': ls * 2,  'ARD': True, 'name': 'SM-Cosine3'}}]}],
    )[args.kern]
    
    wrapper_s = dict(
        nkn=[
            {'name': 'Linear',  'params': {'input_dim': 6, 'output_dim': 8, 'name': 'layer1'}},
           
            {'name': 'Product', 'params': {'input_dim': 8, 'step': 2,       'name': 'layer2'}},
            {'name': 'Linear',  'params': {'input_dim': 4, 'output_dim': 4, 'name': 'layer3'}},
            
            {'name': 'Product', 'params': {'input_dim': 4, 'step': 2,       'name': 'layer4'}},
            {'name': 'Linear',  'params': {'input_dim': 2, 'output_dim': 1, 'name': 'layer5'}}],
           
        rbf=[],
        sm1=[],
        sm2=[],
        sm3=[],
        sm4=[]
    )[args.kern]
    
    wrapper_d= dict(
        nkn=[
            {'name': 'Linear_d',  'params': {'input_dim': 6, 'output_dim': 8, 'name': 'layer1'}},
            {'name': 'Product', 'params': {'input_dim': 8, 'step': 2,       'name': 'layer2'}},
            {'name': 'Linear_d',  'params': {'input_dim': 4, 'output_dim': 4, 'name': 'layer3'}},
            {'name': 'Product', 'params': {'input_dim': 4, 'step': 2,       'name': 'layer4'}},
            {'name': 'Linear_d',  'params': {'input_dim': 2, 'output_dim': 1, 'name': 'layer5'}}],
        rbf=[],
        sm1=[],
        sm2=[],
        sm3=[],
        sm4=[]
    )[args.kern]    
    return kernel, wrapper_s, wrapper_d

def run(data, logger, inducing_points):
    tf.reset_default_graph() 

    ############################## setup data ##############################
    x_train_source, y_train_source = data.x_train_s.astype('float32'), data.y_train_s.astype('float32')
    x_train_target, y_train_target = data.x_train_t.astype('float32'), data.y_train_t.astype('float32')
    x_test,  y_test  = data.x_test.astype('float32'),  data.y_test.astype('float32')
    #std_y_train_source = data.std_y_train
    std_y_train_target = data.std_y_train
    
    N_s, nx = x_train_source.shape
    N_t, _ = x_train_target.shape
    x_source = tf.placeholder(FLOAT_TYPE, shape=[None, nx])
    y_source = tf.placeholder(FLOAT_TYPE, shape=[None])
    x_target = tf.placeholder(FLOAT_TYPE, shape=[None, nx])
    y_target = tf.placeholder(FLOAT_TYPE, shape=[None])

    ############################## build nkn ##############################
    ls_source = median_distance_local(data.x_train_s).astype('float32') ##what median_distance source or target?
    ls_source[abs(ls_source) < 1e-6] = 1.
    kernel, wrapper_s, wrapper_d = NKNInfo(input_dim=nx, ls=ls_source)
    
    wrapper_s = NKNWrapper_t(wrapper_s)
    wrapper_d = NKNWrapper_t(wrapper_d)
   
    nkn_s = NeuralKernelNetwork(nx, KernelWrapper(kernel), wrapper_s)
    nkn_t = NeuralKernelNetwork(nx, KernelWrapper(kernel), wrapper_d)
    ############################## build graph ##############################
    # if args.data in SMALL_DATASETS:
    #     model = gfs.models.GPR(x, tf.expand_dims(y, 1), nkn, name='model')
    # else:
    #     inducing_points = get_kemans_init(x_train, 1000)
    #     model = gfs.models.SGPR(x, tf.expand_dims(y, 1), nkn, Z=inducing_points)
    y_s = tf.expand_dims(y_source, 1)
    #y_in = tf.to_float(y_in)
    y_t = tf.expand_dims(y_target, 1)
   
    if args.data in SMALL_DATASETS:
        model = gfs.models.OTGPR(x_source, y_s, x_target, y_t, nkn_s, nkn_t, name='model')
    else:
        #inducing_points = get_kemans_init(x_train_source, 1000)
        model = gfs.models.SGPR(x_source, tf.expand_dims(y_source, 1), nkn_s, Z=inducing_points)

    objective = model.objective
    optimizer = tf.train.AdamOptimizer(1e-3)
    infer = optimizer.minimize(objective)
    pred_mu, pred_cov = model.predict_y(x_test)

    ############################## session run ##############################
    rmse_results = []
    lld_results = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            indices_s = np.random.permutation(N_s)
            indices_t = np.random.permutation(N_t)
            x_train_source, y_train_source = x_train_source[indices_s, :], y_train_source[indices_s]
            x_train_target, y_train_target = x_train_target[indices_t, :], y_train_target[indices_t]
            fd = {x_source: x_train_source, y_source: y_train_source, x_target: x_train_target, y_target: y_train_target}
            _, obj = sess.run([infer, objective], feed_dict=fd)

            if epoch % 100 == 0:
                logger.info('Epoch {}: Loss = {}'.format(epoch, obj))

                # evaluate
                mu, cov = sess.run(
                    [pred_mu, pred_cov], feed_dict={x_source: x_train_source, y_source: y_train_source, x_target: x_train_target, y_target: y_train_target}
                )

                #all_params = tf.all_variables()
                #print(all_params)
                mu, cov = mu.squeeze(), cov.squeeze()
                rmse = np.mean((mu - y_test) ** 2) ** .5 * std_y_train_target

                log_likelihood = np.mean(np.log(stats.norm.pdf(
                    y_test,
                    loc=mu,
                    scale=cov ** 0.5))) - np.log(std_y_train_target)
                logger.info('test rmse = {}'.format(rmse))
                logger.info('tset ll = {}'.format(log_likelihood))
                rmse_results.append(rmse)
                lld_results.append(log_likelihood)
    return rmse_results, lld_results


if __name__ == "__main__":
    start_time = time.time()
    tf.set_random_seed(123)
    np.random.seed(123)
    data_fetch = get_data(args.split)

    logger = create_logger('results/regression/' + args.split + '/' + args.data, 'reg', __file__)
    logger.info('| jitter level {}'.format(gfs.settings.jitter))
    logger.info('| float type {}'.format(FLOAT_TYPE))
    logger.info('| kernel {}'.format(args.kern))
    logger.info('| dataset {}'.format(args.data))
    logger.info('| split {}'.format(args.split))

    rmse_results = []
    ll_results = []
    
    data = data_fetch(args.data)
    x_train_source, y_train_source = data.x_train_s.astype('float32'), data.y_train_s.astype('float32')
    x_train_target, y_train_target = data.x_train_t.astype('float32'), data.y_train_t.astype('float32')
    inducing_points = []
    #x_train_source, y_train_source = get_kemans_init(x_train_source, y_train_source, 1000)
    x_train_source, y_train_source = dppselect(x_train_source, y_train_source,  100)
    #x_train_source, y_train_source = randselect(x_train_source, y_train_source, 1000)
    #x_train_source, y_train_source = get_selectpoints(x_train_source, y_train_source, x_train_target, y_train_target, 1000)
    for i in range(1, N_RUNS[args.split] + 1):
        logger.info("\n## RUN {}".format(i))
        data = data_fetch(args.data, i)
        #print(data1)
        #print(data2)
        rmse_result, ll_result = run(data, logger, inducing_points)
        rmse_results.append(rmse_result)
        ll_results.append(ll_result)
        logger.info('Collapse Time = {} s'.format(time.time() - start_time))
        #print('ok')

    ########################### logging results ###########################
    for i, (rmse_result, ll_result) in enumerate(zip(rmse_results,
                                                      ll_results)):
        logger.info("\n## Result-by-epoch for RUN {}".format(i))
        logger.info('# Test rmse = {}'.format(rmse_result))
        logger.info('# Test log likelihood = {}'.format(ll_result))

    for i in range(len(rmse_results[0])):
        logger.info("\n## AVERAGE for epoch {}".format(i))
        test_rmses = [a[i] for a in rmse_results]
        test_lls = [a[i] for a in ll_results]

        logger.info("Test rmse = {}/{}".format(
            np.mean(test_rmses), np.std(test_rmses) / N_RUNS[args.split] ** 0.5))
        logger.info("Test log likelihood = {}/{}".format(
            np.mean(test_lls), np.std(test_lls) / N_RUNS[args.split] ** 0.5))
        logger.info('NOTE: Test result above output mean and std. errors')

