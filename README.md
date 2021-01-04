# Author-Age-Prediction-from-Textual-Data

This project will be improving upon the work performed in the research paper titled
’Author Age Prediction from Text using Linear Regression’ by Dong Nguyen, Noah A.
Smith and Carolyn P. Rose from the Carnegie Mellon University published in the year
2011. The paper aims to predict the age of an individual given contextual features
extracted from the text, that prove as important descriptors towards our continuous value
prediction, using Regularized Multivariate Linear Regression.
Using the paper’s methodology, this project aims to work upon the limitations and results
of the paper and perform experiments that would lead to improvements to the original
results. The three main experiments constructed in this project were to use different types of regression models other than multivariate linear regression, working with different
feature extraction techniques such as word embeddings and using Deep Neural Networks
models towards our continuous value prediction

The methodology of the project consists of two main parts which are Feature Extraction
and using those features as independent variables in our regression task for age prediction.
This work considers a combination of the old feature extraction techniques and using these
features as independent variables with different regression models as well as with Deep
Neural Networks for age prediction, and using a new feature extraction technique using
the Skip-Gram model architecture belonging to the Word2Vec algorithm for feature
extraction and using this feature set for our regression problem.

After analyzing the results, Deep Neural Networks with the new Feature Set using Word2Vec
algorithm performed the best among all the subsets of experiments considered with the
lowest MAE of 4.10 and the highest r-coefficient of 0.64. Neural Networks also perform
well with the original feature set with an MAE and r-coefficient better than the original
results which proves that the Neural Networks with one hidden layer outperforms all the
other models in the age prediction task


The code is made up of 10 files and 3 main files:

1) utils.py - Performs important pre-processing and transformation on the data

2) linear_regression.py - Performs multivariate linear regression

3) ridge_close_form.py - Performs the closed form solution of ridge regression

4) ridge_regression.py - Performs multivariate linear regression with L2 regularization using 
			 mini-batch gradient descent

5) lasso_regression.py - Performs multivariate linear regression with L1 regularization using 
			 mini-batch gradient descent

6) regression_methods.py - Performs bayesian linear regression with L2 regularization, ensemble 
			  regression techniques and the support vector regressor on the data

7) dnn.py -  Creates a deep neural network for regression on the two types of features i.e. TF-IDF
             and the word2vec embedding representation



The code can be run by running the three main files:

1) regression.py - The main file in Python that runs the regression methods on the original set 
		   of features and can be run by "python regression.py"

2) neural_network.py - The main file in Python that runs the neural network on the original set 
		   of features and can be run by "python neural_network.py"

3) word2vec.py - Creates a word embedding from the data in the corpus using the word2vec model
		 and performs regression methods for evaluating whether word2vec features gave
	         better results than our original method from the paper and can be run by 
		"python neural_network.py"


Link to the dataset: https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm



