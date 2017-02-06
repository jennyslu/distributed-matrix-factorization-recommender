import numpy as np
from pyspark.accumulators import AccumulatorParam
from pyspark import SparkConf, SparkContext

# class for W
class RowAccumulatorParam(AccumulatorParam):
    '''accumulator to add values to matrix row-wise'''
    def zero(self, values):
        return np.zeros(len(values)).astype('float16')
    def addInPlace(self, value1, value2):
        # value2 is tuple
        # with i-th row to updated as key and
        # value = vector of gradient descent updates
        (i, w_i) = value2
        value1[i] += w_i
        return value1

# class for H
class ColAccumulatorParam(AccumulatorParam):
    '''accumulator to add values to matrix column-wise'''
    def zero(self, values):
        return np.zeros(len(values)).astype('float16')
    def addInPlace(self, value1, value2):
        # value2 is tuple
        # with i-th row to updated as key and
        # value = vector of gradient descent updates
        (j, h_j) = value2
        value1[:,j] += h_j
        return value1

def coalesce_w(x):
    global w_accum
    w_accum += 1

def coalesce_h(x):
    global h_accum
    h_accum += 1

def assign_row_block(i):
    '''takes in row index and assigns to block'''
    global row_block_size
    # return block number from 0 to num_workers-1 (i.e. 0-95)
    return np.floor(i/row_block_size).astype(int)

def assign_col_block(j):
    '''takes in column index and assigns to block'''
    global col_block_size
    # return block number from 0 to num_workers-1 (i.e. 0-95)
    return np.floor(j/col_block_size).astype(int)

def SGD(x):
    global n_updates_acc
    global mse
    for val in x:
        row_block_id = val[0]
        v_iter = val[1][0]
        w_iter = val[1][1]
        h_iter = val[1][2]
    # dictionaries to store W and H
    w = {x[0]:x[1] for x in w_iter}
    h = {x[0]:x[1] for x in h_iter}
    # go through V and update W and H
    for v_ij in v_iter:
        # increment num updates
        n_updates_acc += 1
        i, j = v_ij
        # get row and column
        w_i = w[i]
        h_j = h[j]
        # calculate error
        error = 1 - np.dot(w_i,h_j)
        # increment MSE
        mse += error**2
        # gradients with L2 loss
        w_i -= np.nan_to_num(step_size.value*(-2*error*h_j + 2.0*reg.value*w_i))
        h_j -= np.nan_to_num(step_size.value*(-2*error*w_i + 2.0*reg.value*h_j))
        # update entry
        w[i] = np.nan_to_num(w_i)
        h[j] = np.nan_to_num(h_j)
    # must massage results in something that will return properly
    output = {}
    for row_index in w:
        output[('W', row_index)] = (row_index, w[row_index])
    for col_index in h:
        output[('H', col_index)] = (col_index, h[col_index])
    # return iterator of updated W and H
    return tuple((output.items()))


if __name__ == '__main__':
    conf = (SparkConf()
            .setAppName("SparseMatrixFactorization")
            .set("spark.driver.maxResultSize", "200g"))
    sc = SparkContext(conf = conf)
    # sc = SparkContext(appName = "SparseMatrixFactorization")

    '''PARAMETERS FOR MF'''
    # number of workers
    cpus = sc.defaultParallelism
    # number of latent features
    k = 100
    # max iterations
    max_iters = 100
    # mses list
    mses = []
    # step size parameters
    # decreasing beta will cause step size to decrease more slowly
    # increasing tau will make step size smaller overall
    beta = 0.9
    tau = 5000
    # regularization parameter
    reg = sc.broadcast(0.02)

    '''LOAD DATA'''
    # point to S3 bucket
    files = sc.textFile('s3a://github-recommender/utility/*txt', cpus)
    # parse text files into RDD of tuples that represent position in matrix with 1
    # minus 1 so indices are 0-indexed
    v = files.map(lambda x: (int(x.split(",")[0])-1, int(x.split(",")[1])-1)).distinct()

    '''INITIALIZE W AND H'''
    # initialize W
    # first map gets tuples of (i, 1)
    # reduceByKey to get (unique row indices, number of 1's in that row)
    # second map create W with (i, row array))
    w = v.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], np.random.uniform(0.0001,1,k).astype('float16')))
    # initialize H
    # first map gets tuples of (j, 1)
    # reduceByKey to get (unique column indices, number of 1's in that column)
    # second map create H with (j, column array))
    h = v.map(lambda x: (x[1],1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], np.random.uniform(0.0001,1,k).astype('float16')))

    '''CALCULATE ROWS, COLUMNS, BLOCK SIZE'''
    # number of users/rows
    n = w.count()
    # n = 4496672
    # number of columns/projects
    m = h.count()
    # m = 3253437
    # block size
    row_block_size = np.ceil(n*1./cpus)
    col_block_size = np.ceil(m*1./cpus)

    '''BLOCK V'''
    # separate V into 96 blocks based on row index
    blocked_v = v.keyBy(lambda x: assign_row_block(x[0])).partitionBy(cpus).persist()

    '''STOCHASTIC GRADIENT DESCENT'''
    for i in range(max_iters):
        # create accumulator for MSE
        mse = sc.accumulator(0.0)
        # create accumulator for number of updates per epoch
        n_updates_acc = sc.accumulator(0)
        w_accum = sc.accumulator(0)
        h_accum = sc.accumulator(0)
        # step size is decreasing function of i (learning rate schedule)
        step_size = sc.broadcast(np.power(tau+i, -beta))
        # randomly order strata of V
        perms = np.random.permutation(cpus)
        # create random strata (one sub-chunk of columns from each row block)
        # perms[x[0]] = randomly permuted columns
        # should make strata with rows from x[0] block and cols from perms[x[0]]
        # filter out values that are in that row block but not in correct column block
        filtered_v = blocked_v.filter(lambda x: perms[x[0]] == assign_col_block(x[1][1])).persist()
        n_updates = filtered_v.count()
        # W should have same block ids as V
        blocked_w = w.keyBy(lambda x: assign_row_block(x[0]))
        # block H with the row block id each strata should match with
        blocked_h = h.keyBy(lambda x: np.where(perms == assign_col_block(x[0]))[0][0])
        # group the RDDs together
        # returns [(row_block_id, (V_iter, W_iter, H_iter)), ...]
        stratas = filtered_v.groupWith(blocked_w, blocked_h).partitionBy(cpus)
        # run SGD on each block
        # reduces by W and H so that each value is a list of tuples of newly updated W or H: (index, array)
        w_h = stratas.mapPartitions(SGD).persist()
        filtered_v.unpersist()
        w = w_h.filter(lambda x: x[0][0]=='W').map(lambda x: x[1]).persist()
        h = w_h.filter(lambda x: x[0][0]=='H').map(lambda x: x[1]).persist()
        # call action to actually compute this update!
        w.foreach(coalesce_w)
        h.foreach(coalesce_h)
        # append current MSE
        curr_mse = np.nan_to_num(mse.value)/n_updates_acc.value
        print("{}-th iteration has MSE/update of: {:.5f}".format(i, curr_mse))
        mses.append(curr_mse)

    '''SAVE RESULTS'''
    # # turn RDD into matrix and save
    # w_values = np.zeros((n,k)).astype('float16')
    # h_values = np.zeros((k,m)).astype('float16')
    # w_accum = sc.accumulator(w_values, RowAccumulatorParam())
    # h_accum = sc.accumulator(h_values, ColAccumulatorParam())
    # w.foreach(coalesce_w)
    # h.foreach(coalesce_h)
    # w_accum.value.savetxt('s3a://github-recommender/output/w.txt')
    # h_accum.value.savetxt('s3a://github-recommender/output/h.txt')
    mses_rdd = sc.parallelize(mses)
    mses_rdd.saveAsTextFile('s3a://github-recommender/MSE')
    w.saveAsTextFile('s3a://github-recommender/W')
    h.saveAsTextFile('s3a://github-recommender/H')
