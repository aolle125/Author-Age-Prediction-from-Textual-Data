from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
import re
import string
import numpy as np
from empath import Empath
lexicon = Empath()


#Defining a list of tags like Verbs, Nouns, Pronouns whose counts for every text is required for analysis
tags_list = ['LS','TO','VBN','WP','UH','VBG','JJ','VBZ','VBP','NN','DT',\
'PRP', 'NNPS','WDT', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP','JJR',\
 'JJS', 'PDT', 'MD', 'VB', 'WRB','NNP', 'EX', 'NNS', 'SYM',\
 'CC', 'CD', 'POS', 'PRP$', 'WP$']


'''Function for cleaning the dataframe containing text'''
def preprocess(text):
    text = re.sub('\.\.+',' ',text)     #Removing multiple fullstops
    text = re.sub('\,\,+',' ',text)     #Removing multiple commas
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))          #Removing punctuations from the string
    text = text.translate(str.maketrans(' ', ' ', string.digits))               #Removing number from the string
    text = re.sub(' +',' ',text.strip())                    #Removing extra spaces from the string
    return text
    

'''Function for counting the words contained in each text'''
def w_c(x):
    x = x.split()
    return len([i for i in x])
    

'''Function for reading the words in an English dictionary'''    
def word_dict():
    words_dict = []
    with open("words_dict.txt",'r') as f:
        for line in f:
            words_dict.append(line)
    words_dict = [word.strip("\n").lower() for word in words_dict]
    words_dict = set(words_dict)
    return words_dict
    

'''Function for converting 'Male' and 'Female' to '0' and '1' '''
def encoding(x):
    return '0' if x == 'male' else '1'
    
    
'''Counting the non-dictionary words in each text'''
def non_dict_words(x,words_dict):
    x = x.split()
    return sum([1 for i in x if i not in words_dict])
    
    
'''Counting the dictionary words in each text''' 
def dict_words(x, words_dict):
    x = x.split()
    return sum([1 for i in x if i in words_dict])


'''Function for counting special words i.e. words with length greater than 6 characters'''    
def special_words(x):
    return len([i for i in x if len(i)>6])
    
    
'''Function for finding the pos tags of text'''
def tagger(x):
    return nltk.pos_tag(x)
    

'''Function for analyzing text across lexical categories similar to LIWC'''
def liwc(corpus):    
    liwc = []
    for element in corpus:
        liwc.append((list(lexicon.analyze(element,normalize=True).values())))
        
    return liwc


'''Count of the POS tags for every word to calculate the Unigrams and Bigrams'''
def tags(x):
    tagged_counts = []
    for el in x:
        el = el.split()
        counts = [0] * len(tags_list)
        for ele in el:
            for tag in range(len(tags_list)):
                if ele.endswith(tags_list[tag]):
                    counts[tag]+=1
        tagged_counts.append(counts)
    return tagged_counts
    
 
'''Creating POS tagged sentences which would be used for unigram and bigram analysis''' 
def pos(x):
    el = x.split()
    new_list = []
    for ele in el:
        for tag in range(len(tags_list)):
            if ele.endswith(tags_list[tag]):
                new_list.append(tags_list[tag])
    return ' '.join([item for item in new_list])
    

'''Using the CountVectorizer to calculate the unigram and bigram counts across matrices within a specific range and with a pre-defined minimum count'''    
def n_gram_vectorize(corpus, ranges, min_df):
    vectorizer = CountVectorizer(ngram_range = ranges ,min_df = min_df)
    X = vectorizer.fit_transform(corpus)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X)     #TF-Idf Transformer helps normalize the values across all texts
    X = tf_transformer.transform(X)
    return X
    
 
'''Function to split the dataset into train-validation-test sets''' 
def train_val_test_split(X,y):
    indices = np.random.permutation(X.shape[0])
    split_train = int(0.8*len(indices))
    split_val = int(0.9*len(indices))
    train,val,test = indices[:split_train], indices[split_train:split_val], indices[split_val:]
    X_train, X_val, X_test = X[train,:], X[val,:], X[test,:]
    y_train, y_val, y_test = y[train], y[val], y[test]
    return X_train, X_val, X_test, y_train, y_val, y_test, indices
    
    
 
'''Function to return the predicted values''' 
def pred(X_test, final_w):
    return X_test.dot(final_w)
    
    
    
'''Function to calculate the 'Mean Absolute Error'''     
def calc_mae(y_pred,y_test):
    return np.mean(np.abs(y_test-y_pred))
    
    
    
'''Function to calculate the Pearson's r coefficient which is given by the formula: cov(x,y)/ (std(x) * std(y))'''    
def r(y_pred, y_test):
    mean_pred = np.sum(y_pred) / y_pred.shape[0]
    mean_test = np.sum(y_test) / y_test.shape[0]
    cov = sum((x - mean_pred) * (y - mean_test) for (x,y) in zip(y_pred,y_test)) 
    variance_pred = sum([((x - mean_pred) ** 2) for x in y_pred]) 
    variance_test = sum([((x - mean_test) ** 2) for x in y_test]) 
    std_pred, std_test = np.sqrt(variance_pred), np.sqrt(variance_test)
    
    return cov / (std_pred * std_test)
    


    
    
    
    
    
