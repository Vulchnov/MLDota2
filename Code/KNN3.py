# Import necessary packages here.
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import time

def load_dataset(filename):
  return pd.read_csv(filename)

x_train = load_dataset('dota2TrainMOD.csv')
x_test = load_dataset('dota2Test.csv')

#This function returns a touple that contains two lists with the champ ids for each team
def getTeams(match):
  #result = match.iloc[0] #get the result of a match (1 or -1)
  match = match[4:] #isolate the champion IDs
  teams = ([champID for champID, i in enumerate(match) if i == 1], [champID for champID, i in enumerate(match) if i == -1])
  return teams
  '''
  for champID in range(len(match)):
    #champ *= result #Transforms data such that the champ is either on the winning or losing team
    if match[champID] == 1:
      teams[0].append(champID)
    elif match[champID] == -1:
      teams[1].append(champID)
  return teams
  '''

def getTeamsFromTrain(match):
  teams = ([champID for champID, i in enumerate(match) if i == 1], [champID for champID, i in enumerate(match) if i == -1])
  return teams
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
  
  if try1 < try2:
    return (try1, 1)
  else:
    return (try2, -1)

# This function should calculated the Euclidian distance (team comp similarity) between two datapoints.
def euclidian_distance(dp1, dp2):

  x1 = getTeamDif(set(dp1[0]), dp2)[0]
  x2 = getTeamDif(set(dp1[1]), dp2)[0]
  
  return x1+x2

# This function should get the k nearest neighbors for a new datapoint.
def get_neighbors(x_train, new_dp, k):
  distances = []
  neighbors = []
  
  distances = [(i, math.inf) for i in range(k)]
  
  for index, datapoint in x_train.iterrows():
    max_distance = distances[0][1]
    max_dp_index = 0

    #OPTIMIZATION - Discard neighbors that aren't k nearest to new_dp
    for i in range(k): #get the point with the highest distance in distances
      distance = distances[i][1]
      #print(f"i: {i} distance: {distance}")
      if distance > max_distance:
        max_distance = distance
        max_dp_index = i

    #print(f"max_dp_index: {max_dp_index} index:{index} max_distance: {max_distance}")
    teams1 = getTeams(new_dp)
    teams2 = getTeams(datapoint)

    x1 = getTeamDif(set(teams1[0]), teams2)[1]

    new_dp_distance = euclidian_distance(teams1, teams2) #get euclidean distance of new datapoint
    if new_dp_distance < max_distance:
      distances[max_dp_index] = (x1, new_dp_distance)

  #distances.sort(key=lambda x: x[1])
  for i in range(k):
    neighbors.append(distances[i][0])
  
  return neighbors
  
# This function should determine the class label for the current datapoint
# based on the majority of class labels of its k neighbors.
def predict_dp(neighbors):
  predictions = None
  print(neighbors)
  unique_labels, counts = np.unique(neighbors, return_counts=True)
  predictions = unique_labels[np.argmax(counts)]
  return predictions

# Use the kNN algorithm to predict the class labels of the test set
# with k = 5

k = 3
print("----------Calculating knn predictions for test set k = 3----------")
with open("k3.txt", "w") as f:
  for index, row in x_test.iterrows():
    start = time.time()
    neighbors_indices = get_neighbors(x_train, row, k)
    end = time.time()
    print(f"execution time for getNeighbors(): {(end-start)} s")
    predicted_label = predict_dp(neighbors_indices)
    print(f"PREDICTION FOR DATAPOINT {index}:  {predicted_label}")

    if(row.iloc[0] == predicted_label):
      print("CORRECT PREDICTION")
      f.write("1\n")
    else:
      print("Wrong Prediction")
      f.write("0\n")