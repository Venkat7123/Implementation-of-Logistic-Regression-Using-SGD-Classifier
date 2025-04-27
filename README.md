# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and prepare the Iris dataset into features X and target Y.
2. Split the data into training and testing sets.
3. Train an SGDClassifier model using the training data.
4. Predict and evaluate the model using accuracy score and confusion matrix.

## Program:
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Venkatachalam S
RegisterNumber:  212224220121

```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target',axis=1)
Y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: ",accuracy)

cm = confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix:\n",cm)
```
## Output:
![image](https://github.com/user-attachments/assets/991b27ae-ab3d-4d08-8a3e-adc855c7864d)
![image](https://github.com/user-attachments/assets/d1cf55a6-c64b-452d-be96-dc5f721e529a)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
