
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

logPath = './online/log/9x16_test/'

nameList = ['Alien', 'Conan1', 'Conan2', 'Cooking', 'Rhinos', 'Skiing', 'Surfing', 'War']

num = 48
batch_size = 4
timeList = []
timeListList = [[] for j in range(len(nameList))]

for idx, f in enumerate(nameList):
    for i in range(1, num+1):

        with open(logPath + f"user_{i}/{f}_{i}.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]

            batch_num = (len(rows) - 9) / batch_size
            sum_time = float(rows[-8][1])

            try:
                timeListList[idx].append(round(sum_time/batch_num, 4))
            except IndexError:
                print("IndexError:", idx)

    timeList.append(np.mean(timeListList[idx]))

nameList.reverse()

colors = ['#ff1a1a', '#fff64d', '#e07839', '#1affdb', '#707575', '#2596be', '#feae69', '#e36f34']
colors.reverse()

plt.rcParams['font.size'] = 10
fig, ax = plt.subplots(figsize=(8, 6))

plt.barh(range(len(timeList)), timeList, height=0.5, tick_label=nameList, color=colors)
plt.grid(linestyle='--')
x_major_locator = plt.MultipleLocator(0.1)
ax.xaxis.set_major_locator(x_major_locator)

plt.xlim(0, 1)
plt.ylabel('Videos')
plt.xlabel('TimeConsumption[s]')
plt.tight_layout()

plt.savefig('./online/AverageTime.png')
plt.show()

print("finish!")