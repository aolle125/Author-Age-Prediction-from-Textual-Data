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
