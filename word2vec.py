#Importing the libraries

import pandas as pd
import numpy as np
import time
import string
import re
import regression_methods,dnn
import utils   
import ridge_closed_form,ridge_regression,lasso_regression,linear_regression
import nltk  
import gensim
import warnings 

#Creating a Stemmer for stemming our words in the corpus
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english") 


from sklearn.model_selection import train_test_split
import seaborn as sns 

#Removing stopwords i.e. unnecessary words from our dataset
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append('nbsp')

# ### Reading the data into a dataframe

df = pd.read_csv("blogtext.csv")


df.drop('date', axis = 1, inplace = True)
df = df[df.age<40]
df['final_text'] = df.text
df = df.sample(frac=1,random_state=0).reset_index(drop=True)
df = df.iloc[0:100000]

#Converting words with punctuations to normal so that they do not lose meaning after punctuation removal
punct_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

df.text = df.text.apply(lambda x: ' '.join([word if word not in punct_mapping else punct_mapping[word] for word in x.split()]))

#Remove characters that contain anything but alphabets
df.text = df.text.str.replace("[^a-zA-Z]", " ")

#Converting the text to lower case and using only those words with length greater than 2
df.text = df.text.apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in stop_words and len(word) > 2]))

#Tokenizing or splitting the sentence into a list i.e. 'playing a game of cricket' to ['playing','a','game','of','cricket']
sentences = df.text.apply(lambda x: x.split())

#Using Stemming to chop off the ends of words
sentences = sentences.apply(lambda x: [stemmer.stem(word) for word in x]) 

#Building the word2vec model on the entire corpus which converts every word into a 2-dimensional vector using the Skip-Gram Model 

model_w2v = gensim.models.Word2Vec(
            sentences,
            size=200,       #Vector dimension
            window=5,     #Window Size
            min_count=100,      #Minimum term frequency required                            
            sg = 1,     #Skip-Gram Model 
            hs = 0,     #Hierarchical softmax = False, we use Negative sampling
            negative = 10,      #Negative sampling
            seed = 34           #Random seed
) 

model_w2v.train(sentences, total_examples= len(sentences), epochs=20)       #Number of epochs to train the model for



def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))           #Adding up all the word vectors in a sentence
            count += 1.
        except KeyError:  # Handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count      #Dividing the sum of all word vectors by the count of the words
    return vec


#Converting the sentences to vectors by averaging out the word vectors in the senteneces
wordvec_arr = np.zeros((len(sentences), 200)) 
for i in range(len(sentences)):
    wordvec_arr[i,:] = word_vector(sentences[i], 200)
    
    
#Splitting the data into test-train splits
y = np.array(df['age'].values,dtype='int64')
X_train, X_test, y_train, y_test = train_test_split(wordvec_arr, y, test_size=0.20, random_state=42)


# ### Performing Necessary Pre-Processing of our arrays such as appending the Bias Term into X

b = np.ones(len(X_train))
b = np.reshape(b,(-1,1))
X_train = np.append(b,X_train,axis=1)
w = [0] * X_train.shape[1]
w = np.asarray(w)
lr = 0.001
b = np.ones(len(X_test))
b = np.reshape(b,(-1,1))
X_test = np.append(b,X_test,axis=1)


#Testing the new vector representation of our corpus on the original models

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




#Testing the new vector representation of our corpus on the best model 
#### Using Neural Networks to train our model with the word2vec features


X_train, X_test, y_train, y_test = train_test_split(wordvec_arr, y, test_size=0.20, random_state=42)
y_train, y_test = np.reshape(y_train,(-1,1)),np.reshape(y_test,(-1,1))
X_train, X_test = np.array(X_train, dtype=np.float32),np.array(X_test, dtype=np.float32)

history, preds = dnn.dnn_word2vec(X_train, y_train, X_test, y_test)

#Converting (n,1) shaped arrays to (n,) shape
preds = preds.flatten()
y_test = y_test.flatten()


#Calculating the MAE and r-coefficient on our results from the Neural Network model
mae = utils.calc_mae(preds,y_test)
print("The achieved MAE with Neural Networks is: ", mae)
r = utils.r(preds,y_test)
print("The achieved r coefficient with Neural Networks is: ", r)
print()

#Plotting the train loss vs the validation loss

dnn.plot_loss(history)
