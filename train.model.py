import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load traffic capture CSV file
df = pd.read_csv('/home/ted/Desktop/int/4.2/project/datasets/dataset.model.csv')

# df.head(5)


# Extract relevant features
features = ["source.port", "protocol", "packet.size"]
X = df[features]

# Define the target variable using "packet.size" column
y = df["packet.size"]

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the feature data and transform the feature data
X_norm = scaler.fit_transform(X)

# Split the data into training and testing sets with a 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Create a Random Forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100)

# Train the Random Forest classifier on the training data
rf.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = rf.predict(X_test)

# Calculate the accuracy, precision, recall, and F1 score of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = 'macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)