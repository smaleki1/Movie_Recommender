# Movie Recommender 

The goal of this project is to build a movie recommender system using the MovieLens 20 Million dataSet available on Kaggle (https://www.kaggle.com/grouplens/movielens-20m-dataset)

Implementation steps:

1- The code called "preprocess_SM.py" cleans and shrinks the data. Note that the Jupiter notebook of this code is also uploaded.

2- The code called "preprocess_dictionary_SM.py" creates the dictionary of topples. The reason behind creating a dictionary is to look up values similar to SQL instead of looping through the matrix each time,  which significantly reduces the computational time.

3- The code called "itembased_SM.py" performs the item-item collaborative filtering.

4- The code called "userbased_SM.py" performs the user-user collaborative filtering. Note that the Jupiter notebook of this code is also uploaded to show the mean square error of this method.

5- The code called "mf_SM.py" performs matrix factorisation (PCA) to predict the movie rating.
