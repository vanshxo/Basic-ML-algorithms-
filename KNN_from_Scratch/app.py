from KNNclassifier import KNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
X=load_iris()['data']
y=load_iris()['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

my_knn=KNN(5)
my_knn.fit(X_train=X_train,y_train=y_train)

print(my_knn.predict(X_test))