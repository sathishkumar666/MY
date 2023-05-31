import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd

dataset = pd.read_csv("purchase_logs.csv") 
type(dataset)

dataset.shape

dataset.columns
dataset.info()
dataset.isnull().sum()
dataset.shape
dataset.columns
dataset.info()
dataset.isnull().sum()

X = dataset.iloc[:, 0:-1].values  
y = dataset.iloc[:, -1].values  

from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10 ,train_size=0.90, random_state = 0)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

X_train.shape
X_test.shape

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=36)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
 
error = []

for i in range(1, 41):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 41), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  
 

 













