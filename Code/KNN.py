# Import necessary packages here.
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_dataset(filename):
  return pd.read_csv(filename)

X = load_dataset('Code\\dota2Train.csv')
Y = load_dataset('Code\\dota2Labels.csv')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=17)

#This function returns a touple that contains two lists with the champ ids for each team
def getTeams(match):
  #result = match.iloc[0] #get the result of a match (1 or -1)
  teams = ([], [])
  match = match[4:] #isolate the champion IDs
  for champID in range(len(match)):
    #champ *= result #Transforms data such that the champ is either on the winning or losing team
    if match[champID] == 1:
      teams[0].append(champID)
    elif match[champID] == -1:
      teams[1].append(champID)
  return teams
'''
  teams = ([], [])
  for i in range(4, len(match)):
    if match[i] == 1:
      teams[0].append(i-3)
    elif match[i] == -1:
      teams[0].append(i-3) #should this be teams[1]?
  return teams
'''
#compare the team comp of a training dp to the team comp of a test dp)
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

# This function should calculated the Euclidian distance (team comp similarity) between two datapoints.
def euclidian_distance(dp1, dp2):

  teams1 = getTeams(dp1)
  teams2 = getTeams(dp2)

  x1 = getTeamDif(set(teams1[0]), teams2)
  x2 = getTeamDif(set(teams1[1]), teams2)
  
  return np.sqrt(np.sum([(x1) ** 2, (x2)**2]))

# This function should get the k nearest neighbors for a new datapoint.
def get_neighbors(x_train, new_dp, k):
  distances = []
  neighbors = []
  
  distances = [(i, math.inf) for i in range(k)]
  
  for index, datapoint in x_train.iterrows():
    max_distance = distances[0][1]
    max_dp_index = 0

    #OPTIMIZATION - Discard neighbors that aren't k nearest to new_dp
    for i in range(len(distances)): #get the point with the highest distance in distances
      distance = distances[i][1]
      #print(f"i: {i} distance: {distance}")
      if distance > max_distance:
        max_distance = distance
        max_dp_index = i

    #print(f"max_dp_index: {max_dp_index} index:{index} max_distance: {max_distance}")
    new_dp_distance = euclidian_distance(new_dp, datapoint) #get euclidean distance of new datapoint
    if new_dp_distance < max_distance:
      distances[max_dp_index] = (index, new_dp_distance)

  #distances.sort(key=lambda x: x[1])
  for i in range(k):
    print(distances[i][0])
    neighbors.append(distances[i][0])
  
  return neighbors
  
# This function should determine the class label for the current datapoint
# based on the majority of class labels of its k neighbors.
def predict_dp(neighbors, x_train):
  predictions = None
  
  neighbor_labels = [x_train.loc[i][0] for i in neighbors]
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
  predicted_label = predict_dp(neighbors_indices, x_train)
  predictions.append(predicted_label)
  print(f"prediction made {predicted_label}")

# Calculate and print out the accuracy of your predictions!
correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test, predictions)])
accuracy = (correct / len(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")