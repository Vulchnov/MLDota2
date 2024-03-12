import numpy as np
import json
train = np.loadtxt("dota2Train.csv",
                 delimiter=",", dtype=int)

f = open('heroes.json')
data = json.load(f)

wins = {}
uses = {}

for i in data['heroes']:
    wins[int(i["id"])] = 0
    uses[int(i["id"])] = 0

for mat in train:
    winner = mat[0]
    loser = (winner * -1)
    for i in range(4, len(mat)):
        if mat[i] == winner:
            wins[i-3] += 1
            uses[i-3] += 1
        elif mat[i] == loser:
            uses[i-3] += 1
            
out = open("result.txt", "w")
for i in data['heroes']:
    winrate = 0
    if uses[int(i["id"])]:
        winrate = wins[int(i["id"])]/uses[int(i["id"])]
    out.write(i['localized_name'] + ":" + str(winrate)+"\n")