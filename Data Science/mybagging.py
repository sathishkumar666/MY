
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



# Read dataset to pandas dataframe
dataset = pd.read_csv("aw_fb_data.csv")  
#Preview the first five observations

dataset.head()  

#Preprocessing
#split the dataset into its attributes and labels.
#Stores all observations and columns into X except the last column
X = dataset.iloc[:, :-2].values  
y = dataset.iloc[:, 3].values  

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=2,random_state=0)

#The max_samples argument determines the largest amount of the dataset to use in resampling.
#The max_features argument is the max number of features to use in a sample.
#Lastly, the n_estimators is for determining the number of subsamples to draw. 

#told python was to use up to 70% of the samples, 70% of the features,
#and make 100 different KNN models that use seven neighbors to classify.#Feature Scaling
#MinMaxScaler - Applies minmax normalisation
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

X_train.shape
X_test.shape

h=BaggingClassifier(KNeighborsClassifier(n_neighbors=7),max_samples=0.2,max_features=0.2,n_estimators=500)


h.fit(X_train,y_train)
y_pred = h.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

from sklearn.tree import DecisionTreeClassifier
h=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'),max_samples=0.2,max_features=0.2,n_estimators=500)


h.fit(X_train,y_train)
y_pred = h.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
