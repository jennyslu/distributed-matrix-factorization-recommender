# Distributed Matrix Factorization Recommender

## Motivation

This project was motivated by my initial goal of creating an open source project recommender for GitHub. The data includes approximately 50.1M 'ratings' created from implicit feedback for3.5M projects by 4.5M users.

To give some context to the scale of the data:

 - Netflix's famous challenge consisted of 100M ratings of just 17k movies by 500k users.
  * Utility matrix is 0.05% size and ~3700 times as dense.


 - MovieLens' dataset is another well-known dataset for recommenders. The larger version consists of 10M ratings of 10.7k movies by 71.6k users.
  * Utility matrix is 0.004% size and ~4100 times as dense.

With such a huge matrix I knew I had to use Spark. Spark MLlib has one built-in recommender which is a latent factor collaborative filtering model that uses ALS.

I also knew that my matrix was incredibly sparse. I wanted to add in item side data in an effort to help


## Approach

We want to minimize the Frobenius norm of the reconstruction error.

### Stochastic Gradient Descent

Popularized by Simon Funk,

### Distributed

In order to implement SGD in a distributed way 
