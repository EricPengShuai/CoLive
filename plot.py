import numpy as np
import matplotlib.pyplot as plt
import csv
import os

logPath = 'online/log/9x16_test/'

nameList = ['Alien', 'Conan1', 'Conan2', 'Cooking', 'Rhinos', 'Skiing', 'Surfing', 'War']

num = 48
tileAccList = []
recallList = []
precisionList = []
tileAccListList = [[] for j in range(len(nameList))]
recallListList = [[] for p in range(len(nameList))]
precisionListList = [[] for q in range(len(nameList))]

for idx, f in enumerate(nameList):
    for i in range(1, num+1):

        with open(logPath + f"user_{i}/{f}_{i}.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]

            try:
                tileAccListList[idx].append(round(float(rows[-3][1]), 4))
                recallListList[idx].append(round(float(rows[-2][1]), 4))
                precisionListList[idx].append(round(float(rows[-1][1]), 4))
            except IndexError:
                print("IndexError:", idx)

    tileAccList.append(np.max(tileAccListList[idx]))
    recallList.append(np.max(recallListList[idx]))
    precisionList.append(np.max(precisionListList[idx]))

    # tileAccList.append(np.mean(tileAccListList[idx]))
    # recallList.append(np.max(recallListList[idx]))
    # precisionList.append(np.max(precisionListList[idx]))


# TileMetrics
x = np.arange(len(nameList))
width = 0.2
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(20, 8))
n = 0

plt.bar(x + width * n, tileAccList, width=width, label='AvgTileAccuracy')
plt.bar(x + width * (n+1), recallList, width=width, label='AvgRecall')
plt.bar(x + width * (n+2), precisionList, width=width, label='AvgPrecision')

plt.grid(linestyle='--')
plt.xticks(x + width, nameList, fontsize=20)
plt.ylim(0, 1)
ax.set_ylabel('AverageMetrics')
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

plt.legend(loc='center', ncol=3, bbox_to_anchor=[0.5, 1.05])
# plt.legend(loc='center', bbox_to_anchor=[0.91, 1.12])
# plt.title("Attention offline mode")
plt.tight_layout()
plt.savefig('./online/log/9x16_best.png')
plt.show()

if 'test' in logPath:
    fileName = f'./online/log/9x16_test.csv'
else:
    fileName = f'./online/log/9x16_best.csv'

with open(fileName, 'w', newline='') as f:
    logWriter = csv.writer(f, dialect='excel')
    logWriter.writerow(['AverageMetric']+nameList)
    logWriter.writerows([
        ['Accuracy'] + tileAccList,
        ['Recall'] + recallList,
        ['Precision'] + precisionList
    ])
print(f'AverageMetrics have been saved to into {fileName}, size={os.path.getsize(fileName)} bytes')

