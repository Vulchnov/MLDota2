
with open("k5.txt", "r") as file:
    right = 0
    for i in range(1000):
        read = file.readline()
        if read == "1\n":
            right += 1

print(f"{right/1000}% Correct")
        