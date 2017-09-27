import pandas as pd

from sklearn import model_selection #this one is for spliting the training and testing data

# these ones are models
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# first thing is to read the data that we wrote from last time.
# its formatted like so: 1,2,3,4,5,6,7...n_features, class_label
# each row contains the features and class_label for each image on the dataset.
# now, I understand that Kaggle provided separated training and testing files.
# so, in order to submit our final solution we will train the best model on the whole training data
# however, in order to evaluate the best model, we will split the training data into parts

''' reading trained data '''
# loading data, and separating labels array.
data = pd.read_csv('train_data.csv')
train_data = data.values[:,:-1]
labels = data.values[:,-1]

# K-FOLD
test_size = 0.20

X_train, X_test, y_train, y_test = model_selection.train_test_split(train_data, labels, test_size=test_size, random_state=42)

# uncomment the model you want to test...
# RandomForest got us the best result so far
# Another thing to test: GridSearch to optimize each classifier.
# What it does is, tests it out the classifier with all the possible parameters
# to find the best possible result parameters. Brute Force search.
# take too much to converge may take a while to find the best parameters.
# random forest, for instance, it essentially a bunch of ifs and elses, so it doens't take long
# a neural network, on the other hand, can take a while to train depending on the size of the data
#model = LinearSVC(C=100.0)
#model = DecisionTreeClassifier()
model = RandomForestClassifier()
# model = KNeighborsClassifier()
# model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    # solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    # learning_rate_init=.1)

model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print('Accuracy: {:.3f}%'.format(result*100))

# Suppose we split the data in 80/20:
# ['train', 'train', 'train', 'train', 'test'] - first split
# ['train', 'train', 'train', 'test', 'train'] - second split
# ['train', 'train', 'test', 'train', 'train'] - third split
# . . .
# what KFold does is separate the data in 80/20 K times...
#And finds the optimal split?
# It doesn't find the optimal split, we have to be careful not to split according to the data size...I chose 10... n_splits= 10
# so it will split the data in 10 parts, and use one of the parts as training data and the other as testing data, then it will get
# a different part of the data as training and the other as testing...and so on, dividing it 10 times, which in practice is 10 parts.

# NOW...what the cross_val_score function does is, after each split, it will compute the score of the model...
# the result is a list of scores of each split it did
# and the result I outputting is the mean score found and std deviation of them.

# is it clear?
#yes, but can you have a look at the picture I sent you. Because I have only used classification report before, is this similar?
# to answer your question. They are not similar...
# what you did on your model was generate the report for one split test.
# what we did here was, split the data, train, compute score, split again with a different cut of the training data, and compute the score again, and so on
# and then we took the average of the scores.
# BUT we can use the report here as well, but we would use the report n_split times, since we have n_split results.
#Also, this isn't predicting the REAL test data at the moment (model.predict), I assume this is just to test which classifier is best, and optimise?

#yes, as I said up there, I split this data only to find the best model...since the test data the kaggle privides doens't have label
# instead, our model is supposed to predict on that data, and that's what we upload to them. their data labels by our classifier.

kf = model_selection.KFold(n_splits=10, random_state=42)
results = model_selection.cross_val_score(model, train_data, labels, cv=kf)
# print("Accuracies: {})".format(results))
print("Mean Accuracy: {:.3f}%\nStd Deviation: ({:.3f}%)".format(results.mean()*100, results.std()*100))








