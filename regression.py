
# ### Predicting Age of the Author Using Linear Regression: aclweb.org/anthology/W11-1515.pdf

# ### Importing Packages



import pandas as pd
import numpy as np
import time
import string
import re



import utils
import ridge_closed_form,ridge_regression,lasso_regression,linear_regression
import regression_methods
import dnn
from sklearn.preprocessing import MinMaxScaler


# ### Reading the data into a dataframe



df = pd.read_csv("blogtext.csv")
df.drop('date', axis = 1, inplace = True)
df = df[df.age<40]
df['final_text'] = df.text


# ### Performing Word Count for all the blogs in our dataset


df['wc'] = df.text.apply(utils.w_c)
df = df[df.wc > 20]


# ### Shuffling our Dataset and picking 10000 values for training and testing of our model as chosen in the paper


df = df.sample(frac=1,random_state=0).reset_index(drop=True)
df = df.iloc[0:10000]


# ### Creating a Dictionary of Words present in the English Language ~300,000 words


words_dict = utils.word_dict()


# ### Pre-processing of our Textual Data


df['text'] = df.text.apply(utils.preprocess)


# ### Dropping empty rows of text after Pre-processing


indices = df[df['text'] == ''].index
df.drop(indices,inplace=True)


# ### Performing Linguistic Inquiry using empath to calculate linguistic features from sentences

corpus = df.text.str.lower()
liwc = utils.liwc(corpus)


# ### Converting Gender 'Male' and 'Female' to 0 and 1

df.gender = df.gender.apply(utils.encoding)


# ### Finding the count of Non-Dictionary Words in our blogs

df['non_dict'] = df.text.apply(lambda x: utils.non_dict_words(x,words_dict))



# ### POS Tagging of every Blog

df.text = df.text.apply(lambda x: x.split())
df['text1'] = df.text.apply(utils.tagger)
df.text1 = df.text1.apply(lambda x: [''.join(t) for t in x])  #Converting a tuple('is', VBZ) to (isVBZ)
df.text1 = df.text1.apply(lambda x: ' '.join([i for i in x]))  #Converting the list of tagged words back to a string 



# ### Counting the frequency of 30 tags in every blog

corpus = df.text1.values
tagged_counts = utils.tags(corpus) 

df.text1 = df.text1.apply(utils.pos)


# ### Counting the number of special words in every blog i.e. words with length > 6

df['special'] = df.text.apply(utils.special_words)

df.text = df.text.apply(lambda x: ' '.join([i for i in x]))



# ### Finding the Unigram Counts across the Corpus

corpus = df['text'].values
X = utils.n_gram_vectorize(corpus, (1,1), 5).toarray()          #Choosing min_df = 5 i.e. the term must have a frequency of 5 to be considered


# ### Finding the Bigram Counts of the POS tags across the Corpus

corpus = df['text1'].values
X1 = utils.n_gram_vectorize(corpus, (2,2), 1).toarray()

X_tsvd = np.append(X,X1,axis=1)


# ### Appending all required features into one array
scaler = MinMaxScaler()
gender = np.array(df['gender'].values,dtype='int64')
gender = np.reshape(gender,(-1,1))
gender = scaler.fit_transform(gender)
X_tsvd = np.append(X_tsvd,gender,axis=1)
non_dict_words = np.array(df['non_dict'].values)
non_dict_words= np.reshape(non_dict_words,(-1,1))
non_dict_words = scaler.fit_transform(non_dict_words)
X_tsvd = np.append(X_tsvd,non_dict_words,axis=1)
special = np.array(df['special'].values)
special = np.reshape(special,(-1,1))
special = scaler.fit_transform(special)
X_tsvd = np.append(X_tsvd,special,axis=1)
special = np.array(df['special'].values)
special = np.reshape(special,(-1,1))
special = scaler.fit_transform(special)
X_tsvd = np.append(X_tsvd,special,axis=1)
wc = np.array(df['wc'].values)
wc = np.reshape(wc,(-1,1))
wc = scaler.fit_transform(wc)
X_tsvd = np.append(X_tsvd,wc,axis=1)
tagged_counts = scaler.fit_transform(tagged_counts)
X_tsvd = np.append(X_tsvd, tagged_counts, axis=1)
liwc = scaler.fit_transform(liwc)
X_tsvd = np.append(X_tsvd,liwc,axis=1)


# ### Keeping all the age values into 'Y'

y = np.array(df['age'].values,dtype='int64')


# ### Splitting the Array into Train, Validation and Test Sets

X_train, X_val, X_test, y_train, y_val, y_test,indices = utils.train_val_test_split(X_tsvd,y)


# ### Performing Necessary Pre-Processing of our arrays such as appending the Bias Term into X

X_train = np.concatenate((X_train, X_val))
y_train = np.concatenate((y_train,y_val))
b = np.ones(len(X_train))
b = np.reshape(b,(-1,1))
X_train = np.append(b,X_train,axis=1)
w = [0] * X_train.shape[1]
w = np.asarray(w)
lr = 0.001
b = np.ones(len(X_test))
b = np.reshape(b,(-1,1))
X_test = np.append(b,X_test,axis=1)


# 1) Showing the earlier results before performing other experiments

# ### Closed form Ridge Regression

final_w = ridge_closed_form.solver(X_train,y_train,1)
y_pred = utils.pred(X_test,final_w)
print("The MAE for Closed form Ridge Regression is: ",utils.calc_mae(y_pred,y_test))
print("The Pearson's Correlation Coefficient (r) for Closed form Ridge Regression is: ", utils.r(y_pred,y_test))
print()

# ### Ridge Regression with Gradient Descent


L2 = 1
final_w, cost_history = ridge_regression.gradient_descent_ridge(X_train, y_train, w, 0.00001, 100000,L2)
y_pred = utils.pred(X_test,final_w)
print("The MAE for Ridge Regression with Gradient Descent is: ",utils.calc_mae(y_pred,y_test))
print("The Pearson's Correlation Coefficient (r) for Ridge Regression with Gradient Descent is: ", utils.r(y_pred,y_test))
print()

# ### Lasso Regression with Gradient Descent


L1 = 0.01
final_w, cost_history = lasso_regression.gradient_descent_lasso(X_train, y_train, w, 0.0001, 100000, L1)
y_pred = utils.pred(X_test,final_w)
print("The MAE for Lasso Regression with Gradient Descent is: ",utils.calc_mae(y_pred,y_test))
print("The Pearson's Correlation Coefficient (r) for Lasso Regression with Gradient Descent is: ", utils.r(y_pred,y_test))
print()


# Splitting the Datasets again due to dataset shape: Earlier we required to add the bias term to the train and test sets
# Sklearn handles calculation of bias term implicitly so we do not need to add it into our train and test sets
X_train, X_val, X_test, y_train, y_val, y_test,indices = utils.train_val_test_split(X_tsvd,y)



# Testing various methods for Regression other than Multivariate Linear Regression

# 1) Bayesian Regression with L2 regularization

y_pred = regression_methods.BayesianRegression(X_train, y_train, X_test, y_test)

print("The MAE for Bayesian Regression with L2 regularization is: ",utils.calc_mae(y_pred,y_test))
print("The Pearson's Correlation Coefficient (r) for Bayesian Regression with L2 regularization is: ", utils.r(y_pred,y_test))
print()




# 2) Ensemble Regression methods such as AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor

regression_methods.Ensemble(X_train, y_train, X_test, y_test)



# 3) Support Vector Regression

y_pred = regression_methods.SupportVectorRegression(X_train, y_train, X_test, y_test)
print("The MAE for Support Vector Regression is: ",utils.calc_mae(y_pred,y_test))
print("The Pearson's Correlation Coefficient (r) for Support Vector Regression is: ", utils.r(y_pred,y_test))
print()















