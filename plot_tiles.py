import math

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

logPath = './online/log/9x16_test/'

nameList = ['Alien', 'Conan1', 'Conan2', 'Cooking', 'Rhinos', 'Skiing', 'Surfing', 'War']

num = 48
batch_size = 4
tileList = []
tileListList = [[] for j in range(len(nameList))]

for idx, f in enumerate(nameList):
    for i in range(1, num+1):

        with open(logPath + f"user_{i}/{f}_{i}.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]

            tile_list = [float(item[3]) for item in rows[1:-8]]

            try:
                tileListList[idx].append(np.mean(tile_list))
            except IndexError:
                print("IndexError:", idx)

    tileList.append(math.ceil(np.mean(tileListList[idx])))

x = np.arange(len(nameList))
width = 0.4
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(20, 8))
plt.bar(x, tileList, width=width)

plt.grid(linestyle='--')
plt.xticks(x, nameList, fontsize=20)
ax.set_ylabel('AverageTiles')
plt.tight_layout()
plt.savefig('./online/log/9x16_test/48_users/tiles.png')
plt.show()
print("finish!")

fileName = f'./online/log/9x16_test/48_users/tiles.csv'

with open(fileName, 'w', newline='') as f:
    logWriter = csv.writer(f, dialect='excel')
    logWriter.writerow(['AverageMetric']+nameList)
    logWriter.writerows([
        ['Tiles'] + tileList,
    ])