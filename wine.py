import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from operator import itemgetter
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import graphviz 
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize


dataset = np.genfromtxt("red.csv", delimiter=";", skip_header=1)

avr = list()


n_neighbors = 4
# separate the data from the target attributes
X = dataset[:, [0,1,2,3,4,5,6,7,8,9,10]] 
Y = dataset[:,11]


sel = VarianceThreshold(threshold=(.97* (1 - .90)))
X = sel.fit_transform(X)

X = preprocessing.scale(X)





best = (-1, 0, None, None)
for i in range(1, 100):
	clf = neighbors.KNeighborsClassifier(i, weights="distance")
	recall_scores = cross_val_score(clf, X, Y, cv=10, scoring='neg_log_loss')
	accuracy_scores = cross_val_score(clf, X, Y, cv=10)
	recall = recall_scores.mean()
	accuracy = accuracy_scores.mean()
	if accuracy > best[1]:
		best = (i, accuracy, accuracy_scores, recall_scores)

print("KNN-%0.2i: %0.2f (+/- %0.2f). Recall: %0.2f (+/- %0.2f)." % (best[0], best[2].mean(), best[2].std() * 2, best[3].mean(), best[3].std() * 2))



#scores = cross_val_score(clf, X, Y, cv=5)


#print(scores.mean())

best = (-1, 0, None, None)
for i in range(1, 100):
	clf = tree.DecisionTreeClassifier(criterion="gini", min_impurity_decrease=(0.0001 * i))
	recall_scores = cross_val_score(clf, X, Y, cv=10, scoring='neg_log_loss')
	accuracy_scores = cross_val_score(clf, X, Y, cv=10)
	recall = recall_scores.mean()
	accuracy = accuracy_scores.mean()
	if accuracy > best[1]:
		best = (i, accuracy_scores.mean(), accuracy_scores, recall_scores)

print("DT-GINI  %0.2f: %0.2f (+/- %0.2f). Recall: %0.2f (+/- %0.2f)." % (best[0] * 0.001, best[2].mean(), best[2].std() * 2, best[3].mean(), best[3].std() * 2))





clf = tree.DecisionTreeClassifier(criterion="gini", min_impurity_decrease=(0.003)).fit(X, Y)

dot_data = tree.export_graphviz(clf, out_file=None, 
	feature_names=["fixed acidity", "citric acid", "free sulfur dioxide", "density", "alcohol"],  
	class_names=["0","1","2","3","4","5","6","7","8","9"],  
	filled=True, rounded=True,  
	special_characters=True)  
graph = graphviz.Source(dot_data)

graph.render("wine") 




best = (-1, 0, None, None)
for i in ['rbf']:
	clf = svm.SVC(kernel=i, probability=True) 
	recall_scores = cross_val_score(clf, X, Y, cv=10, scoring="neg_log_loss")
	accuracy_scores = cross_val_score(clf, X, Y, cv=10)
	recall = recall_scores.mean()
	accuracy = accuracy_scores.mean()
	if accuracy > best[1]:
		best = (i, accuracy_scores.mean(), accuracy_scores, recall_scores)

print("Support Vector (%s): %0.2f (+/- %0.2f). Recall: %0.2f (+/- %0.2f)." % (best[0], best[2].mean(), best[2].std() * 2, best[3].mean(), best[3].std() * 2 ))



best = (-1, 0, None)
for i in range(0,1):
	clf = GaussianNB()
	recall_scores = cross_val_score(clf, X, Y, cv=10, scoring="neg_log_loss")
	accuracy_scores = cross_val_score(clf, X, Y, cv=10)
	recall = recall_scores.mean()
	accuracy = accuracy_scores.mean()
	if accuracy > best[1]:
		best = (i, accuracy_scores.mean(), accuracy_scores, recall_scores)

print("Bayes: %0.2f (+/- %0.2f). Recall: %0.2f (+/- %0.2f)." % (best[2].mean(), best[2].std() * 2, best[3].mean(), best[3].std() * 2 ))