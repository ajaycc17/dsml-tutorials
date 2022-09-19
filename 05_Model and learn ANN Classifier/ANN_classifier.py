import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Loading the training dataset
data = pd.read_csv("Train.csv")
X = data.iloc[:, :-1].values                                        # Excluding the last column from the training dataset
y = data.iloc[:, -1].values                                         # Only taking the last column of the training dataset

# Loading the test dataset
test_dsml = pd.read_csv("Test-Kaggle-Data.csv")
P = test_dsml.loc[ : , test_dsml.columns != 'Id']                   # Excluding the Id column from the test dataset

# Encoding categorical values
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)               # Considering only 10% of the training dataset as test dataset

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
P = sc.transform(P)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Adding the input and first hidden layer
classifier = Sequential()
classifier.add(Dense(4, kernel_initializer='uniform', activation='relu', input_dim=6))

# Adding second hidden layer
classifier.add(Dense(4, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=100, epochs=300)

# Predicting the test set results from the training dataset
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) *1

# For the test dataset
y_pred1 = classifier.predict(P)
y_pred1 = (y_pred1 > 0.5) *1 +1

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Heatmap for the accuracy check
# sns.heatmap(cm, annot=True) 
# plt.savefig('h.png')

df = pd.read_csv("Test-Kaggle-SampleSubmission.csv", usecols = ['Id'])                  # Using only the first column of the sample file
df["Category"] = y_pred1                                                                # Generating a new column named Category
df.to_csv("18018.csv", index = False)
