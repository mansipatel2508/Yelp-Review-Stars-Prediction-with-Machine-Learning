# Yelp Business Stars’ Rating Prediction
https://colab.research.google.com/drive/1q5rvPOO8DvD8DV5DNLMVc8UDY7ntWHah

**Big Data | Data Cleaning | Data Preporocessing | Text Processing | TF-IDF vectorization |  Regression | Classification | Model Evaluation | Model Performance Comparison**


 Tradition (Standard) AI Models : KNN | SVM | Logistic Regression | Multinomial Naive Bayes | Linear Regression 
 
 Deep Learning Models : Neural Network ( Regression & Classification )

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
* Numpy
* Pandas


# Dataset
https://www.yelp.com/dataset/download
# Load dataset
The data containing json files was converted to a compatible file to load on pandas’ data frame.Used business. json and review.json files to understand the dataset.
Grouped the multiple reviews on bussiness_id to get all reviews given by the user into one text.

Merged the datasets with on BusinessID and got the final dataset shape as below

##    Data Pre-Processing/ Cleaning

* Dropped the rows with categories that have null values
* Filtered the data frame more by removing rows with business Ids having review count less than
a certain threshold
* Cleaned the reviews text data by removing stop words, punctuations and white spaces.
* Used TF-IDF vectorization for Feature Extraction and used its parameters 
* Performed label encoding on the “stars” column (Output Feature)
* Normalized the “ Review_count “ Column to make it comparable with min-max normalization

```
# TF-IDF Vectorization - Feature Extraction
import sklearn.feature_extraction.text as sk_text
Tfidf_vectorizer = sk_text.TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1), min_df = .05 , max_df = .85)
 ```
 # Splitting the data 
 Split the data into 80% train and 20% test
 # Regression Model

## Linear Regression
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/Reg.png "REG")
## Neural Network Using Tensorflow

Used earlystopping to prevent overfitting the model and used checkpointer to save the best model ran in the loop several time to jump out of the local mininum.

Applied paramter tuning by changing following:

* Activation function : relu, sigmoid,tanh
* Number of Dense Layers
* Number of Neurons in each layer
* Learning rate for Activation
* Optimizer : SGD, Adamax, Adam, Adagrad

![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/Reg_NN.png "Reg_NN")

## Comparison
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/Comparison.png "COMP")
# Classification Model
## Logistic Regression
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/LR.png "LR")
## SVM
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/SVM.PNG "SVM")
## KNN
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/KNN.png "KNN")
## MNB
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/MNB.png "MNB")

# Boost up Performances
Output feature - review ratings categorised into categories as high, low and medium to boost the performance of the above applied model and it significantly boosts the performance
## KNN
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/KNN2.png "KNN2")
## Logistic Regression
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/Logs2.png "Logs2")
## SVM
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/SVM2.png "SVM2")

## Neural Network Using Tensorflow

Used earlystopping to prevent overfitting the model and used checkpointer to save the best model ran in the loop several time to jump out of the local mininum.

Applied paramter tuning by changing following:

* Activation function : relu, sigmoid,tanh
* Number of Dense Layers
* Number of Neurons in each layer
* Learning rate for Activation
* Optimizer : SGD, Adamax, Adam, Adagrad

![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/NN1.png "NN1")

# Boost up Performances
Output feature - review ratings categorised into categories as high, low and medium to boost the performance of the above applied model and it significantly boosts the performance

Also applied Grid Search to get the best optimizer using keras wrappers library. This gives the best optimizer from given list for best performing model so far with accuracy, this all boost up the performance and beats the standard AI classification models.


![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/NN2.png "NN2")

## Comparison
Comparing the NN with previously best performed Logistic Regression model

![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/COMP2.png "COMP2")
## comparing all classification models
Observing all the F1 score, clearly the NN performs better than all other models such as Logistic Regression, SVM, KNN and MNB.
![](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning/blob/master/images/COMP3.png "COMP3")


## Mini Project 1 & 2
Mansi Patel

February 13, 2019

Prof : H. Chen

Class : CSC 215-01

