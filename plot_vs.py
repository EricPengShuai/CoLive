import numpy as np
import matplotlib.pyplot as plt
import csv

ConvLSTM = 'D:/VR_project/ViewPrediction/online/log/9x16_train.csv'
LiveDeep = 'D:/VR_project/LiveDeep_All/LiveDeep/normal/9x16.csv'
Attention = 'D:/VR_project/youattention/log/offline.csv'
# Attention = 'D:/VR_project/youattention/log/online.csv'

nameList = []
tileAccList = [[], [], []]
recallList = [[], [], []]
precisionList = [[], [], []]

with open(ConvLSTM, 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

    nameList = rows[0][1:]
    tileAccList[0] = [float(item) for item in rows[1][1:]]
    recallList[0] = [float(item) for item in rows[2][1:]]
    precisionList[0] = [float(item) for item in rows[3][1:]]

with open(LiveDeep, 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

    tileAccList[1] = [float(item) for item in rows[1][1:]]
    recallList[1] = [float(item) for item in rows[2][1:]]
    precisionList[1] = [float(item) for item in rows[3][1:]]

with open(Attention, 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

    tileAccList[2] = [float(item) for item in rows[1][1:]]
    recallList[2] = [float(item) for item in rows[2][1:]]
    precisionList[2] = [float(item) for item in rows[3][1:]]


# TileMetrics
x = np.arange(len(nameList))
width = 0.2
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(20, 8))
plt.bar(x, tileAccList[0], width=width, label='ConvLSTM')
plt.bar(x + width, tileAccList[1], width=width, label='LiveDeep')
plt.bar(x + width * 2, tileAccList[2], width=width, label='Attention Offline')
# plt.bar(x + width * 3, tileAccList[3], width=width, label='Attention Online')
plt.grid(linestyle='--')
# plt.xticks(x + (3 * width) / 2, nameList, fontsize=20)
plt.xticks(x + width, nameList, fontsize=20)
plt.ylim(0, 1)
plt.ylabel('AverageAccuracy[%]')
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.legend(loc='center', ncol=4, bbox_to_anchor=[0.5, 1.05])
# plt.savefig('./online/AvgAcc.png')
# plt.savefig('./online/AvgAccBest.png')
plt.savefig('./online/AvgAccTrain.png')
plt.show()

x = np.arange(len(nameList))
width = 0.2
plt.rcParams['font.size'] = 20
fig1, ax1 = plt.subplots(figsize=(20, 8))
plt.bar(x, recallList[0], width=width, label='ConvLSTM')
plt.bar(x + width, recallList[1], width=width, label='LiveDeep')
plt.bar(x + width * 2, recallList[2], width=width, label='Attention Offline')
# plt.bar(x + width * 3, recallList[3], width=width, label='Attention Online')
plt.grid(linestyle='--')
# plt.xticks(x + (3 * width) / 2, nameList, fontsize=20)
plt.xticks(x + width, nameList, fontsize=20)
plt.ylim(0, 1)
plt.ylabel('AverageRecall[%]')
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.legend(loc='center', ncol=4, bbox_to_anchor=[0.5, 1.05])
# plt.savefig('./online/AvgRecall.png')
# plt.savefig('./online/AvgRecallBest.png')
plt.savefig('./online/AvgRecallTrain.png')
plt.show()


x = np.arange(len(nameList))
width = 0.2
plt.rcParams['font.size'] = 20
fig2, ax2 = plt.subplots(figsize=(20, 8))
plt.bar(x, precisionList[0], width=width, label='ConvLSTM')
plt.bar(x + width, precisionList[1], width=width, label='LiveDeep')
plt.bar(x + width * 2, precisionList[2], width=width, label='Attention Offline')
# plt.bar(x + width * 3, precisionList[3], width=width, label='Attention Online')
plt.grid(linestyle='--')
# plt.xticks(x + (3 * width) / 2, nameList, fontsize=20)
plt.xticks(x + width, nameList, fontsize=20)
plt.ylim(0, 1)
plt.ylabel('AveragePrecision[%]')
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.legend(loc='center', ncol=4, bbox_to_anchor=[0.5, 1.05])
# plt.savefig('./online/AvgPrecision.png')
# plt.savefig('./online/AvgPrecisionBest.png')
plt.savefig('./online/AvgPrecisionTrain.png')
plt.show()
