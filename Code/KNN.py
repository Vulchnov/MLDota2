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

#This function returns a touple that contains two lists, one for each team
def getTeams(match):
  teams = ([], [])
  for i in range(4, len(match)):
    if match[i] == 1:
      teams[0].append(i-3)
    elif match[i] == -1:
      teams[0].append(i-3)
  return teams

def getTeamDif(expect, actual):
  try1 = 0
  try2 = 0
  for member in actual[0]:
    if not member in expect:
      try1 +=1

  for member in actual[1]:
    if not member in expect:
      try2 +=1

  return min(try1, try2)

# This function should calculated the Euclidian distance between two datapoints.
def euclidian_distance(dp1, dp2):
  # Write your code here!
  teams1 = getTeams(dp1)
  teams2 = getTeams(dp2)


  
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