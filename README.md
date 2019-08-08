# Yelp-Review-Stars-Prediction-with-Machine-Learning
# Yelp Business Stars’ Rating Prediction

**Big Data | Data Cleaning | Data Preporocessing | Text Processing | TF-IDF vectorization |  Regression | Classification | Model Evaluation | Model Performance Comparison**


 Tradition AI Models : KNN | SVM | Logistic Regression | Multinomial Naive Bayes | Linear Regression 
 
 Deep Learning Models : Neural Network ( Regression & Classification )
 
 ## Mini Project 1
 Mansi Patel
 
 February 13, 2019
 
 Prof : Haiquen Chen 
 
 
Class : CSC 215-01

# Problem statement
Predicting the review stars from 1-5 star ratings based on the review given by the user.

Machine Learning project aims

*  learn text vectorization (IF-IDF)
* big data handling & preprocess the data
*  merging two big datasets
*  treat problem as rgression and classification, observe it
* Apply and compare tradition AI models with Deep Learning Nueral Network

Tools and Libraries used
* sklearn
* TensorFlow


# Dataset
https://www.yelp.com/dataset/download
# Load dataset
The data containing json files was converted to a compatible file to load on pandas’ data frame.Used business. json and review.json files to understand the dataset.
```
import os
import json
import csv
import pandas as pd
import numpy as np
import collections
from scipy.stats import zscore
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', -1)
```
# Creating Business Dataframe
business_df= pd.read_csv('business.tsv', delimiter ="\t")

# Creating Review Dataframe
review_df= pd.read_csv('review_stars.tsv', delimiter ="\t")

