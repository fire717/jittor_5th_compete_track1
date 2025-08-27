import os


count = [0,0,0,0,0,0]
with open("TrainSet/labels/train.txt",'r') as f:
    lines = f.readlines()
    for line in lines:
        num = int(line.strip().split(" ")[-1])
        count[num] += 1
print("train:",count)

count = [0,0,0,0,0,0]
with open("TrainSet/labels/val.txt",'r') as f:
    lines = f.readlines()
    for line in lines:
        num = int(line.strip().split(" ")[-1])
        count[num] += 1
print("val:",count)