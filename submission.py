#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """

    dict = {}
    list = x.split()
    for str in list:
        if str in dict:
            dict[str] += 1
        else:
            dict[str] = 1

    return dict

    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    cache = {}

    def predictor(x):
        fi = featureExtractor(x)
        score = 0
        for feature in fi:
            if feature in weights:
                score += fi[feature] * weights[feature]
        if score > 0:
            return 1
        return -1

    def sdF(i):
        (str, y) = trainExamples[i]
        gradient = {}

        if not str in cache:
            x = featureExtractor(str)
            cache[str] = x
        else:
            x = cache[str]

        for feature in x:
            if not(feature in weights):
                weights[feature] = 0
            if y*x[feature]*weights[feature] < 1:
                gradient[feature] = -y * x[feature]
            #else:
            #    gradient[feature] = 0
        return gradient


    def SGD(n, eta_glob):
        eta = eta_glob
        for t in range(numIters):

            for i in range(n):

                gradient = sdF(i)
                for feature in gradient:
                    if not (feature in weights):
                        weights[feature] = 0
                    weights[feature] -= eta * gradient[feature]
            eta = 10/((t+1)**3)
            #print(evaluatePredictor(trainExamples, predictor))
            #print(evaluatePredictor(testExamples, predictor))

    SGD(len(trainExamples), eta)
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)


        phi = dict(random.sample(weights.items(), random.randint(1,len(weights))))


        for feature in phi:
            phi[feature] = random.uniform(-40.0, 40.0,)

        score = 0
        for feature in phi:
            score += phi[feature] * weights[feature]
        #score = dotProduct(phi, weights)

        if score >= 0:
            y = 1
        else:
            y = -1

        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):

        words = {}
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        str = x.replace(" ","")
        if(len(str) < n):
            words[str] = 1
            return words
        for i in range(len(str) - n + 1):
            s = str[i:i+n]
            if s in words:
                words[s] += 1
            else:
                words[s] = 1


        # END_YOUR_CODE
        return words
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    def calcLoss(example, centroid):
        sum = 0
        for feature in example:
            if feature in centroid:
                sum += (example[feature] - centroid[feature])**2
            else:
                sum += (example[feature])**2
        return sum

    def addFeatures(newCentroid, example):
        for feature in example:
            if feature in newCentroid:
                newCentroid[feature] += example[feature]
            else:
                newCentroid[feature] = example[feature]
        return newCentroid



    centroids = []
    assignments = [0] * (len(examples))
    #random.seed(24)
    for i in range(K):
        centroid = examples[random.randint(0,len(examples))]
        #for feature in centroid:
        #  centroid[feature] = random.randint(-100, 100)
        centroids.append(centroid)


    for v in range(maxIters):

        for i, example in enumerate(examples):
            min = 999999
            for t, centroid in enumerate(centroids):
                loss = calcLoss(example, centroid)
                if loss < min:
                    min = loss
                    assignments[i] = t

        oldCentroid = centroids.copy()
        for i, centroid in enumerate(centroids):
            newCentroid = {}
            numOfPoints = 0
            for t, example in enumerate(examples):
                if assignments[t] == i:
                    newCentroid = addFeatures(newCentroid, example)
                    numOfPoints += 1
            if numOfPoints != 0:
                for feature in newCentroid:
                    newCentroid[feature] /= numOfPoints
            centroids[i] = newCentroid

        if oldCentroid == centroids:
            loss = 0
            for i, example in enumerate(examples):
                centroid = centroids[assignments[i]]
                loss += calcLoss(example, centroid)
            return (centroids, assignments, loss)


    loss = 0
    for i, example in enumerate(examples):
        centroid = centroids[assignments[i]]
        loss += calcLoss(example, centroid)


    return (centroids, assignments, loss)

    # END_YOUR_CODE

