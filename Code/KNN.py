# Import necessary packages here.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_dataset(filename):
  return pd.read_csv(filename)

X = load_dataset('dota2Train.csv')
Y = load_dataset('dota2Labels.csv')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17)

def normalize_feature(data, feature):
  feature_set = data[feature]
  mini = np.min(feature_set)
  maxi = np.max(feature_set)

  for val in feature_set:
    val = (val-mini)/(maxi-mini)
  
  return feature_set

# This function should standardize a feature (column) of a dataset.
def standardize_feature(data, feature):
  feature_set = data[feature]
  std = np.std(feature_set)
  mean = np.mean(feature_set)

  for val in feature_set:
    val = (val-mean)/std
  
  return feature_set


# This function should calculated the Euclidian distance between two datapoints.
def euclidian_distance(dp1, dp2):
  # Write your code here!
  x1 = int(dp1["budget"])
  x2 = int(dp1["revenue"])
  y1 = int(dp2["budget"])
  y2 = int(dp2["revenue"])
  return np.sqrt(np.sum([(x1-y1) ** 2, (x2-y2)**2]))

# This function should get the k nearest neighbors for a new datapoint.
def get_neighbors(x_train, new_dp, k):
  distances = []
  neighbors = []
  
  # Write your code here!
  for index, datapoint in x_train.iterrows():
    distances.append((index, euclidian_distance(new_dp, datapoint)))
  distances.sort(key=lambda x: x[1])
  for i in range(k):
    print(distances[i][0])
    neighbors.append(distances[i][0])
  
  return neighbors
  
# This function should determine the class label for the current datapoint
# based on the majority of class labels of its k neighbors.
def predict_dp(neighbors, y_train):
  predictions = None
  
  # Write your code here!
  neighbor_labels = [y_train.loc[i] for i in neighbors]
  print(neighbor_labels)
  unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
  predictions = unique_labels[np.argmax(counts)]
  
  return predictions

# Use the kNN algorithm to predict the class labels of the test set
# with k = 3
k = 3
predictions = []
for index, row in x_test.iterrows():
  neighbors_indices = get_neighbors(x_train, row, k)
  predicted_label = predict_dp(neighbors_indices, y_train)
  predictions.append(predicted_label)
  

# Calculate and print out the accuracy of your predictions!
correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test, predictions)])
accuracy = (correct / len(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")