import itertools
import numpy as np
from pyspark.accumulators import AccumulatorParam
from pyspark import SparkContext
from fileinput import input
from glob import glob


class MatrixAccumulatorParam(AccumulatorParam):
    def zero(self, values):
        return dict((k,0) for k in values)
    def addInPlace(self, value1, value2):
        # value2 is dictionary
        # with (i,j) as key and value = gradient
        for k in value2.keys():
            value1[k] += value2[k]
        return value1


def SGD(x, w, k, eps, reg):
    global w
    global k
    global mse
    global n_updates
    # for given filled position in matrix
    # i = row index (user)
    # j = column index (project)
    i = x[0]
    j = x[1]

    '''GRADIENT DESCENT'''
    # get i-th row of w (should have k columns)
    w_i = []
    w_i_indices = []
    # all indices start at 1
    for k_j in range(1,k+1):
        w_i_indices.append((i,k_j))
        w_i.append(w.value[(i,k_j)])
    # get j-th column of h (should have k rows)
    h_j = []
    h_j_indices = []
    # all indices start at 1
    for k_i in range(1,k+1):
        h_j_indices.append()
        h_j.append(w.value[(k_i,j)])
    # current prediction for V(i,j)
    dot_product = np.dot(w_i, h_j)
    # rating should be 1
    error = 1 - dot_product
    # add to MSE
    mse += error**2
    # if no regularization
    if reg == 0:
        # update for W
        w_grad = np.dot(-2*error, h_j).tolist()
        # update for H
        h_grad = np.dot(-2*error, w_i).tolist()
    else:
        # update for W with L2 loss
        w_grad = (np.dot(-2*error, h_j) + np.dot(2*reg, w_i)).tolist()
        # update for H with L2 loss
        h_grad = (np.dot(-2*error, w_i) + np.dot(2*reg, h_j)).tolist()
    # create update dictionaries
    w_update = dict(zip(w_i_indices, w_grad))
    h_update = dict(zip(h_j_indices, h_grad))
    # update accumulators
    w += w_update
    h += h_update
    n_updates += 1


def extract_ratings(s):
    data = s.split(",")
    return (int(data[0]), int(data[1]))

if __name__ == '__main__':
    sc = SparkContext(appName = "MFRecommender")

    '''SET PARAMETERS'''
    # get number of CPUs
    cpus = sc.defaultParallelism
    # number of users/rows
    n = 4496672
    # number of columns/projects
    m = 3253438
    # number of latent features
    k = 200
    # learning rate
    eps = 0.001
    # regularization parameter
    reg = 0
    # max iterations
    max_iters = 1000

    '''LOAD DATA'''
    # point to S3 bucket
    files = sc.textFile('s3a://github-recommender/sparse/*txt', cpus*3)
    # parse text files into RDD of tuples that represent position in matrix with 1
    v = files.map(extract_ratings).distinct().persist()

    '''INITIALIZE W'''
    # create tuples of indices for W (n, k)
    # user mappings started at 1
    w_indices = list(itertools.product(range(1,n+1), range(1,k+1)))
    # initialize W with small random values
    w_values = [np.random.uniform(0,1./(k*10)) for i in range(n*k)]
    # create dictionary representation for W
    w_dict = dict(zip(w_indices, w_values))
    # create accumulator for W
    w = sc.accumulator(w_dict, MatrixAccumulatorParam())
    '''INITIALIZE H'''
    # create tuples of indices for H (k, m)
    # project mappings started at 1
    h_indices = list(itertools.product(range(1,k+1), range(1,m+1)))
    # initialize H with small random values
    h_values = [np.random.uniform(0,1./(k*10)) for i in range(k*m)]
    # create dictionary representation for H
    h_dict = dict(zip(h_indices, h_values))
    # create accumulator for H
    h = sc.accumulator(h_dict, MatrixAccumulatorParam())

    '''STOCHASTIC GRADIENT DESCENT'''
    mses = []
    i = 0
    while i < max_iters:
        # create accumulator for MSE
        mse = sc.accumulator(0.0)
        # create accumulator for number of updates per epoch
        n_updates = sc.accumulator(0.0)
        # shuffle V
        shuffled_v = v.map(lambda s: (random.randint(0,n*m), s)).sortByKey(True).map(lambda s: s[1]).persist()
        shuffled_v.foreach(lambda x: SGD(x, k, eps, reg))
        shuffled_v.unpersist()
        # store MSE/update of this stage
        curr_mse = mse.value/n_updates.value
        mses.append(curr_mse)
        # to check convergence
        '''
        if len(mses) > 1:
            if abs(curr_mse - mses[-1]) < 0.001:
                mses.append(curr_mse)
                break
            else:
                mses.append(curr_mse)
        '''
        i += 1

    '''SAVE W AND H'''
    w_rdd = sc.parallelize(list(w.value.items()), cpus*3)
    w_rdd.saveAsTextFile('s3a://github-recommender/output/')
    h_rdd = sc.parallelize(list(h.value.items()), cpus*3)
    h_rdd.saveAsTextFile('s3a://github-recommender/output/')
