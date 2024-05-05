
with open("k10.txt", "r") as file:
    right = 0
    for i in range(1001):
        read = file.readline()
        print(read)
        if read == "1\n":
            right += 1

print(f"{right/1001}% Correct")
        