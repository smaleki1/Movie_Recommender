"""
@author: siavashmaleki
"""

from __future__ import print_function, division
from builtins import range, input

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# loading the dataset saved in the previous code called 
# "preprocess_dictionary_SM.py

with open('user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)


N = np.max(list(user2movie.keys())) + 1
# Note, the following steps should be done since the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

# This if loop was added to make sure N > 10000 is not selected
if N > 10000:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()


K = 25 # number of neighbors I want to consider
limit = 5 # number of common movies users must have in common to be considered
neighbors = [] # storing neighbors in this list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use
for i in range(N):
    
  # finding the 25 closest users to user i
  movies_i = user2movie[i]
  movies_i_set = set(movies_i)

  # calculating avg and deviation
  ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { movie:(rating - avg_i) for movie, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # saving these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(N):
    if j != i:
      movies_j = user2movie[j]
      movies_j_set = set(movies_j)
      common_movies = (movies_i_set & movies_j_set) # intersection
      if len(common_movies) > limit:
          
        # calculating avg and deviation
        ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # calculating correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
        w_ij = numerator / (sigma_i * sigma_j)

        # insert into sorted list and truncate
        sl.add((-w_ij, j))
        if len(sl) > K:
          del sl[-1]

  # storing the neighbors
  neighbors.append(sl)

  # print out useful things
  if i % 1 == 0:
    print(i)


# using neighbors, calculating train and test MSE

def predict(i, m):
  # calculating the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # note, the weight is stored as its negative
    # which is why it gets multiplied by -1

    try:
      numerator += -neg_w * deviations[j][m]
      denominator += abs(neg_w)
    except KeyError:
    # Since I do not want to loop through the dictionary twice 
    # to check whether the movie exists in the dictionary or not
    # I simply throw a keyerror, meaning if the movie does not exists, it passes
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # set min rating to be 0.5
  return prediction


train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
  # calculating the prediction for this movie
  prediction = predict(i, m)

  # saving the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []

# same thing
for (i, m), target in usermovie2rating_test.items():
  # calculating the prediction for this movie
  prediction = predict(i, m)

  # saving the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)


# calculating accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

# Note that here I am not reporting the root mean square error
print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))


