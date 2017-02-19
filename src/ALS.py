import numpy as np
from pyspark.accumulators import AccumulatorParam
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

if __name__ == '__main__':
    # conf = (SparkConf()
    #         .setAppName("SparseMatrixFactorization")
    #         .set("spark.driver.maxResultSize", "200g"))
    # sc = SparkContext(conf = conf)
    sc = SparkContext(appName = "SparkALS")

    '''LOAD DATA'''
    # point to S3 bucket
    files = sc.textFile('s3a://github-recommender/utility/*txt', cpus)
    # parse text files into RDD of tuples that represent position in matrix with 1
    # minus 1 so indices are 0-indexed
    v = files.map(lambda x: (int(x.split(",")[0])-1, int(x.split(",")[1])-1)).distinct()
    ratings = v.map(lambda l: Rating(l[0], l[1], float(5))).persist()

    '''TRAIN MODEL'''
    k = 250
    als_max_iters = 20
    model = ALS.train(ratings, k, als_max_iters)

    # Evaluate the model on training data
    predictions = model.predictAll(v).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))
