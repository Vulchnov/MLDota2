import numpy as np
import json
train = np.loadtxt("dota2Train.csv",
                 delimiter=",", dtype=int)

with open("dota2TrainMOD.csv", "w") as file:
    for mat in train:
        winner = mat[0]
        useful = mat[4:]
        useful = useful * winner
        string = ','.join(useful.astype(str))
        file.write(string+ "\n")