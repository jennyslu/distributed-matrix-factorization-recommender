import itertools
import numpy as np
from pyspark.accumulators import AccumulatorParam
from pyspark import SparkContext

# class for W
class RowAccumulatorParam(AccumulatorParam):
    def zero(self, values):
        return np.zeros(len(values)).astype('float16')
    def addInPlace(self, value1, value2):
        # value2 is dictionary
        # with i-th row to updated as key and
        # value = vector of gradient descent updates
        for i in value2.keys():
            value1[i] += value2[i]
        return value1

# class for H
class ColAccumulatorParam(AccumulatorParam):
    def zero(self, values):
        return np.zeros(len(values)).astype('float16')
    def addInPlace(self, value1, value2):
        # value2 is dictionary
        # with i-th row to updated as key and
        # value = vector of gradient descent updates
        for j in value2.keys():
            value1[:,j] += value2[j]
        return value1



def SGD(x, k, eps, reg):
    global w
    global h
    global mse
    global n_updates
    # for given filled position in matrix
    # i = row index (user)
    # j = column index (project)
    # user IDs start at 1 but matrix index starts at 0
    i = x[0]-1
    # project IDs start at 1 but matrix index starts at 0
    j = x[1]-1

    '''GRADIENT DESCENT'''
    # get i-th row of w (should have k columns)
    w_i = w.value[i]
    # get j-th column of h (should have k rows)
    h_j = h.value[:,j]
    # current prediction for V(i,j)
    dot_product = np.dot(w_i, h_j)
    # rating should be 1
    error = 1 - dot_product
    # add to MSE
    mse += error**2
    # if no regularization
    if reg == 0:
        # update for W
        w_grad = -2*error*h_j
        # update for H
        h_grad = -2*error*w_i
    else:
        # update for W with L2 loss
        w_grad = -2*error*h_j + 2*reg*w_i
        # update for H with L2 loss
        h_grad = -2*error*w_i + 2*reg*h_j
    # create update dictionaries
    w_update = {i:w_grad}
    h_update = {j:h_grad}
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
    cpus = sc.defaultParallelism*5
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
    v = files.map(lambda x: (int(x.split(",")[0]), int(x.split(",")[1]))).distinct().persist()

    '''INITIALIZE W'''
    # initialize W with small random values
    w_values = np.random.rand(n,k).astype('float16')/(k*10)
    # create accumulator for W
    w = sc.accumulator(w_values, RowAccumulatorParam())
    '''INITIALIZE H'''
    # initialize H with small random values
    h_values = np.random.rand(k,m).astype('float16')/(k*10)
    # create accumulator for H
    h = sc.accumulator(h_values, ColAccumulatorParam())

    '''STOCHASTIC GRADIENT DESCENT'''
    mses = []
    i = 0
    while i < max_iters:
        # create accumulator for MSE
        mse = sc.accumulator(0.0)
        # create accumulator for number of updates per epoch
        n_updates = sc.accumulator(0)
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
        if i%5 == 0:
            '''SAVE W AND H'''
            w_rdd = sc.parallelize(list(w.value.items()), cpus*3)
            w_rdd.saveAsTextFile('s3a://github-recommender/W/')
            h_rdd = sc.parallelize(list(h.value.items()), cpus*3)
            h_rdd.saveAsTextFile('s3a://github-recommender/H/')

            '''SAVE MSE'''
            mse_rdd = sc.parallelize(mses, cpus*3)
            mse_rdd.saveAsTextFile('s3a://github-recommender/MSE/')
