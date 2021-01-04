# ### Predicting Age of the Author Using Linear Regression: aclweb.org/anthology/W11-1515.pdf

# ### Importing Packages



import pandas as pd
import numpy as np
import time
import string
import re


from sklearn.model_selection import train_test_split
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



# ### Shuffling our Dataset and picking 150000 values for training and testing of our model

#Using the advantage of neural networks, i.e. they scale well with increase in data
#Increasing data in Linear models lead to either the same or increase in loss values


df = df.sample(frac=1,random_state=0).reset_index(drop=True)
df = df.iloc[0:150000]


# ### Creating a Dictionary of Words present in the English Language ~300,000 words

words_dict = utils.word_dict()


# ### Pre-processing of our Textual Data


df['text'] = df.text.apply(utils.preprocess)


# ### Dropping empty rows of text


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
X = utils.n_gram_vectorize(corpus, (1,1), 5).toarray()


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


X_train, X_test, y_train, y_test = train_test_split(X_tsvd, y, test_size=0.20, random_state=42)

#Reshaping our train and test sets in the required format for the model
y_train, y_test = np.reshape(y_train,(-1,1)),np.reshape(y_test,(-1,1))
X_train, X_test = np.array(X_train, dtype=np.float32),np.array(X_test, dtype=np.float32)


# Deep Neural Networks with one hidden layer in Keras using TensorFlow backend
history, preds = dnn.dnn_reg(X_train, y_train, X_test, y_test)



#Converting (n,1) shaped arrays to (n,) shape
preds = preds.flatten()
y_test = y_test.flatten()



#Calculating the MAE and r-coefficient for Deep Neural Network regression
mae = utils.calc_mae(preds,y_test)
print("The achieved MAE with Neural Networks is: ", mae)
print()
r = utils.r(preds,y_test)
print("The achieved r coefficient with Neural Networks is: ", r)
print()

#Plotting the train loss vs the test loss

dnn.plot_loss(history)