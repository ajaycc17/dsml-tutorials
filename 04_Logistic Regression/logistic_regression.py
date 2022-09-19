from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
train_dsml = pd.read_csv("dsml_ass2_train.csv")         # Loading the training dataset
train_dsml.head()                                       # Printing the column heads for reference

# Prepare the training set
X = train_dsml.iloc[:, :-1]                             # Excluding the last column which contains the binary classification
y = train_dsml.iloc[:, -1]                              # Only considering the last column which contains the binary classification

# # Plot the relation of each feature with the category
# plt.xlabel('Features')
# plt.ylabel('Category')

# pltX = train_dsml.loc[:, '1']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='blue', label='1')

# pltX = train_dsml.loc[:, '39']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='red', label='39')

# pltX = train_dsml.loc[:, '0']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='green', label='0')

# pltX = train_dsml.loc[:, '26.97']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='yellow', label='26.97')

# pltX = train_dsml.loc[:, '0.0']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='tan', label='0.0')

# pltX = train_dsml.loc[:, '0.0.1']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='teal', label='1')

# pltX = train_dsml.loc[:, '0.1']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='aliceblue', label='0.1')

# pltX = train_dsml.loc[:, '0.2']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='olivedrab', label='0.2')

# pltX = train_dsml.loc[:, '0.3']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='slategray', label='0.3')

# pltX = train_dsml.loc[:, '195.0']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='steelblue', label='195.0')

# pltX = train_dsml.loc[:, '80.0']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='darksalmon', label='80.0')

# pltX = train_dsml.loc[:, '106.0']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='peru', label='106.0')

# pltX = train_dsml.loc[:, '77.0']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='orange', label='77.0')

# pltX = train_dsml.loc[:, '70.0']
# pltY = train_dsml.loc[:, '0.4']
# plt.scatter(pltX, pltY, color='black', label='70.0')

# plt.legend(loc=4, prop={'size':8})
# plt.show()


# """----------------Just for Checking Accuracy on training dataset------------------"""

# # Splitting the training dataset itself to test the accuracy
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

# # Testing accuracy: Training the model
# model = LogisticRegression()
# model.fit(x_train, y_train)

# # Testing accuracy: Test the model
# predictions = model.predict(x_test)

# # Checking the precision, recall and f1-score
# print(classification_report(y_test, predictions))
# print(accuracy_score(y_test, predictions))

# """-----------------Just for Checking------------------"""



# Train the model: Actual model for test data
model = LogisticRegression()
model.fit(X,y)

# Load the Test dataset
test_dsml = pd.read_csv("dsml_ass2_test_without-answer.csv")        # Loading the test dataset
P = test_dsml.loc[ : , test_dsml.columns != 'Id']                   # Excluding the Id column for testing

# Test the model for Test dataset
predictions = model.predict(P)

# Writing and generating the csv file
df = pd.read_csv("dsml_ass2_test_Kaggle_SampleSubmission.csv", usecols = ['Id'])            # Using only the first column of the sample file
df["Category"] = predictions                                                                # Generating a new column named predictions
df.to_csv("18018.csv", index = False)                                                       # Generating the CSV file
