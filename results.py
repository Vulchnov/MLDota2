
with open("k3.txt", "r") as file:
    right = 0
    total = 0
    read = file.readline()
    while read != "":
        if read == "1\n":
            right += 1
        total += 1
        read = file.readline()

print(f"K = 3: {right/total}% Correct")
        

with open("k5.txt", "r") as file:
    right = 0
    total = 0
    read = file.readline()
    while read != "":
        if read == "1\n":
            right += 1
        total += 1
        read = file.readline()

print(f"K = 5: {right/total}% Correct")

with open("k10.txt", "r") as file:
    right = 0
    total = 0
    read = file.readline()
    while read != "":
        if read == "1\n":
            right += 1
        total += 1
        read = file.readline()

print(f"K = 10: {right/total}% Correct")