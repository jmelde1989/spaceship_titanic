import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import data
data = pd.read_csv("Datasets/train.csv")
test = pd.read_csv("Datasets/test.csv")

### PREPROCESSING ###

# separate delimited columns
data[['Group', 'NumberInGroup']] = data['PassengerId'].str.split('_', expand=True)
data[['Deck', 'Num', 'Side']] = data['Cabin'].str.split('/', expand=True)

# fill in missing values
data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
data['Age'] = data['Age'].fillna(data['Age'].mean())

# remove records where HomePlanet and Destination are missing for the time being
data = data.dropna()

# set index to PassengerId for easier dataframe joins
data.set_index('PassengerId', inplace=True)

# remove Cabin and Name
data = data.drop(['Cabin', 'Name'], axis=1)

# convert data types to reduce memory footprint
data = data.astype({'HomePlanet': 'category', 'CryoSleep': 'bool', 'Destination': 'category', 'VIP': 'bool', 'Deck': 'category', 'Side': 'category'})
data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Group', 'NumberInGroup', 'Num']] = data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Group', 'NumberInGroup', 'Num']].apply(pd.to_numeric, downcast='signed')

# one-hot encode categoric fields
data = pd.get_dummies(data)

### MODEL TRAINING ###

X = data.drop('Transported', axis=1)
y = data['Transported']

# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# fit the model to the training data
rf_classifier.fit(X_train, y_train)

# make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy: {accuracy}")

### MODEL PREDICTIONS ###

# prepare kaggle results
# separate delimited columns
test_data = test
test_data[['Group', 'NumberInGroup']] = test_data['PassengerId'].str.split('_', expand=True)
test_data[['Deck', 'Num', 'Side']] = test_data['Cabin'].str.split('/', expand=True)

# fill in missing values
test_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = test_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

test_data = test_data.drop(['Cabin', 'PassengerId', 'Name'], axis=1)

# convert data types
test_data = test_data.astype({'HomePlanet': 'category', 'CryoSleep': 'bool', 'Destination': 'category', 'VIP': 'bool', 'Deck': 'category', 'Side': 'category'})
test_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Group', 'NumberInGroup', 'Num']] = test_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Group', 'NumberInGroup', 'Num']].apply(pd.to_numeric, downcast='signed')

# one-hot encode categoric fields
test_data = pd.get_dummies(test_data)

# predict kaggle results
kaggle_pred = rf_classifier.predict(test_data)

# save results
sub = pd.read_csv('Datasets/sample_submission.csv')

sub['Transported'] = kaggle_pred

sub.to_csv('Datasets/test_submission.csv', index=False)