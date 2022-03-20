import pickle
import _pickle
import torch.cuda
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn import preprocessing
import cv2
import random
import math
import matplotlib.pyplot as plt
import scipy.io
import csv
from pyquaternion import Quaternion
import time

# added arguments and cnn_model
from Arguments import get_args
import convlstm
import os

# CALCULATE DEGREE DISTANCE BETWEEN TWO 3D VECTORS
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def degree_distance(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180

def vector_to_ang(_v):
    # v = np.array(vector_ds[0][600][1])
    # v = np.array([0, 0, 1])
    _v = np.array(_v)
    # degree between v and [0, 1, 0]
    alpha = degree_distance(_v, np.array([0, 1, 0]))
    phi = 90.0 - alpha
    # proj1 is the projection of v onto [0, 1, 0] axis
    proj1 = [0, np.cos(alpha/180.0 * np.pi), 0]
    # proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    proj2 = _v - proj1
    # theta = degree between project vector to plane and [1, 0, 0]
    theta = degree_distance(proj2, np.array([1, 0, 0]))
    sign = -1.0 if degree_distance(_v, np.array([0, 0, -1])) > 90 else 1.0
    theta = sign * theta
    return theta, phi


def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h/2.0 - (_h/2.0) * np.sin(_phi/180.0 * np.pi)
    temp = _theta
    if temp < 0:
        temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0/360 * _w)
    return int(x), int(y)


def data_prepare(idx, videoId, userId, t_list):
    Userdata = []
    UserFile = 'D:/VR_project/LiveDeep_All/vr-dataset/Experiment_' \
               + str(idx) + '/' + str(userId) + "/video_" + str(videoId) + ".csv"
    print('Load user\'s excel info from', UserFile)

    t = []
    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        i, t_len = 0, len(t_list)
        for row in csv_reader:
            if float(row[1]) >= round(t_list[i], 3):
                v0 = [0, 0, 1]
                q = Quaternion([float(row[5]), -float(row[4]), float(row[3]), -float(row[2])])
                Userdata.append(q.rotate(v0))
                t.append([float(row[2]), float(row[3]), float(row[4]), float(row[5])])
                i += 1
                if i == t_len:
                    break
    Userdata = np.array(Userdata)
    return Userdata

def create_fixation_map(_X, _y, _idx, H, W):
    v = _y[_idx]
    theta, phi  = vector_to_ang(v)
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H-hi-1, W-wi-1] = 1
    return result

def de_interpolate(raw_tensor, N):
    """
    :param raw_tensor: [B, C, H, W]
    :param N
    :return: [B, C, H // 2, W // 2]
    """
    out = np.zeros((N, 9, 16))
    for idx in range(10):
        out = out + raw_tensor[:, idx::10, idx::10]
    return out

# 对视频的每一帧生成用户真实的fixation map (N, 90, 160) 
def create_sal_fix(dataset, videoId, userId):
    """
    :param dataset[i] = [timestamp, fixationList, saliencyMap] or dataset = all saliency_maps
    :param videoId: video's Id
    :param userId: user's Id
    """
    saliency_maps = dataset
    if args.salFlag == "attention":
        saliency_maps = np.array([i[2] for i in dataset])
        t_list = [i[0] for i in dataset]
    elif args.salFlag == "sample_attention":
        t_list = np.loadtxt(timeBasePath + f"{videoId}-{fileDict[videoId]}.txt")
    elif args.salFlag == "cnn_sphere":
        t_list = np.loadtxt(timeBasePath + salDict[videoId].split('.')[0] + '.txt')
    else:
        t_list = np.loadtxt(timeBasePath + f"{videoId}-{fileDict[videoId]}.txt")
        print("Warning! Please pay attention to salFlag!")
    print(f"\nLoad timestamp from {timeBasePath}")

    saliency_maps = de_interpolate(saliency_maps, len(saliency_maps))
    mmscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    saliency_maps = mmscaler.fit_transform(saliency_maps.ravel().reshape(-1, 1)).reshape(saliency_maps.shape)

    N, H, W = saliency_maps.shape
    series = data_prepare(1, videoId, userId, t_list)

    mmscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    fixation_map = np.array([create_fixation_map(None, series, idx, H, W) for idx,_ in enumerate(series)])
    headmap = np.array([cv2.GaussianBlur(item, (args.gblur_size_width, args.gblur_size_high), 0) for item in fixation_map])
    fixation_maps = mmscaler.fit_transform(headmap.ravel().reshape(-1, 1)).reshape(headmap.shape)

    return saliency_maps, fixation_maps

class SaliencyDataset(Dataset):
    def __init__(self, saliency_maps,  fixation_maps, predict_time, transform=None):
        self.saliency_maps = saliency_maps
        self.fixation_maps = fixation_maps
        self.predict_time = predict_time
        self.transform = transform

    def __len__(self):
        return len(self.fixation_maps)

    def __getitem__(self, idx):
        if idx < len(self.saliency_maps) - self.predict_time:
            saliency = self.saliency_maps[idx]
            fixation = self.fixation_maps[idx]
            predict_saliency = self.saliency_maps[idx + self.predict_time]
            predict_fixation = self.fixation_maps[idx + self.predict_time]
        else:
            saliency = self.saliency_maps[idx]
            fixation = self.fixation_maps[idx]
            predict_saliency = self.saliency_maps[idx]
            predict_fixation = self.fixation_maps[idx]

        if self.transform:
            # saliency = torch.from_numpy(saliency)
            # fixation = torch.from_numpy(fixation)
            # predict_saliency = torch.from_numpy(predict_saliency)
            # predict_fixation = torch.from_numpy(predict_fixation)
            saliency = self.transform(saliency)
            fixation = self.transform(fixation)
            predict_saliency = self.transform(predict_saliency)
            predict_fixation = self.transform(predict_fixation)

        return saliency, fixation, predict_saliency, predict_fixation

def sequentialData(dataset, idx, windows):
    """
    :param dataset: fixation maps, saliency maps or frames
    :param idx: start frame's id
    :param windows: predict window size
    :return sal, fix, pre_sal, pre_fix, sal_fix
    """
    sal_list, fix_list, pre_sal_list, pre_fix_list = [], [], [], []
    mix_list = []
    for j in range(windows):
        sal, fix, pre_sal, pre_fix = dataset[idx + j]
        sal_list.append(sal)
        fix_list.append(fix)
        pre_sal_list.append(pre_sal)
        pre_fix_list.append(pre_fix)

        mix_sal_fix = torch.stack([sal[0], fix[0]], dim=0)
        mix_list.append(mix_sal_fix)

    sal_output = torch.stack(sal_list, dim=0)
    fix_output = torch.stack(fix_list, dim=0)
    pre_sal_output = torch.stack(pre_sal_list, dim=0)
    pre_fix_output = torch.stack(pre_fix_list, dim=0)
    mix_output = torch.stack(mix_list, dim=0)

    sal_output = torch.unsqueeze(sal_output, 0)
    fix_output = torch.unsqueeze(fix_output, 0)
    pre_sal_output = torch.unsqueeze(pre_sal_output, 0)
    pre_fix_output = torch.unsqueeze(pre_fix_output, 0)
    mix_output = torch.unsqueeze(mix_output, 0)

    return sal_output, fix_output, pre_sal_output, pre_fix_output, mix_output

def train(dataset, frameId):
    loss = 0
    min_loss = float('inf')
    max_loss = -float('inf')
    print(f"Train frameId={frameId}~{frameId+args.windows}...")
    for epoch in range(args.epochs):
        sal, fix, _, labels, inputs = sequentialData(dataset, frameId, args.windows)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)

        output_list, output_last, conv_output, conv_output_list_ret = net(inputs)

        loss = loss_function(conv_output_list_ret, labels)

        loss.backward()
        optimizer.step()

        # scheduler.step(loss)

        max_loss = max(max_loss, loss.item())
        min_loss = min(min_loss, loss.item())

        if (epoch+1) % 2 == 0:
            print('\r[%3d]/[%d] loss: %.6f' % (epoch+1, args.epochs, loss.item()), end='')

    # torch.save(net.state_dict(), modelPath)
    print('\nTrain finished', round(loss.item(), 5))
    # print('Updated model have been saved to', modelPath)
    return loss.item(), max_loss, min_loss

def test(dataset, frameId, startTime):
    global good_test_frame, total_test_frame

    sal, fix, pre_sal, labels, inputs = sequentialData(dataset, frameId, args.windows)
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)

        output_list, output_last, conv_output, conv_output_list_ret = net(inputs)

        for j in range(0, args.windows):
            try:
                pre_image = conv_output_list_ret[0, j, 0].detach().cpu().numpy()
                real_image = labels[0, j, 0].detach().cpu().numpy()
            except IndexError:
                print('j=', j, conv_output_list_ret.shape)

            pre_image[pre_image < args.threshold] = 0
            pre_image[pre_image > args.threshold] = 1
            real_image[real_image < args.threshold] = 0
            real_image[real_image > args.threshold] = 1

            if args.showImage:
                plt.imshow(pre_image)
                plt.axis('off')
                plt.title(f'pre_image[{frameId+j}]')
                plt.show()

                plt.imshow(real_image)
                plt.axis('off')
                plt.title(f'real_image[{frameId+j}]')
                plt.show()

            fit = sum(map(sum, (pre_image + real_image) != 1))
            mistake = pre_image.size - fit
            fetch = sum(map(sum, (pre_image == 1)))
            need = sum(map(sum, (real_image == 1)))
            right = sum(map(sum, (pre_image + real_image) > 1))
            wrong = fetch - right

            eps = 1e-3
            tileAccuracy = round(fit / real_image.size, 4)
            recall = round((right + eps) / (need + eps), 4)
            precision = round((right + eps) / (fetch + eps), 4)
            if recall >= thres_recall:
                good_test_frame += 1
            total_test_frame += 1
            metrics = [frameId+j, fit, mistake, fetch, need, right, wrong, tileAccuracy, recall, precision]
            if right == need:
                metrics.append(True)
            else:
                metrics.append(False)

            tileAccList.append(tileAccuracy)
            recallList.append(recall)
            precisionList.append(precision)

            logWriter.writerow(metrics)

            if (j + 1) % 2 == 0:
                print('\rFrame [{}/{}], Loss: {:.5f}'.format(
                    j+1+frameId, args.windows+frameId, loss_function(conv_output_list_ret, labels).item()), end='')
    endTime = time.time()

    print(f'\nTest frameId={frameId}~{frameId+args.windows} finished, time: {round(endTime - startTime, 3)}')

if __name__ == '__main__':
    # load the settings
    args = get_args()

    net = convlstm.ConvLSTM_model(input_dim=args.input_size,
                                  hidden_dim=args.hidden_size, kernel_size=(5, 5),
                                  num_layers=args.numLayers, batch_first=True)
    print("Total number of parameters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

    if torch.cuda.is_available():
        net = net.cuda()

    loss_function = nn.MSELoss(reduction='mean')

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2, args.epochs], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    fileDict = {0: 'Conan1', 1: 'Skiing', 2: 'Alien', 3: 'Conan2', 4: 'Surfing',
                5: 'War', 6: 'Cooking', 7: 'Football', 8: 'Rhinos'}
    salFileList = ["saliency_ds2_topic0", "saliency_ds2_topic1", "saliency_ds2_topic2",
                   "saliency_ds2_topic3", "saliency_ds2_topic4", "saliency_ds2_topic5",
                   "saliency_ds2_topic6", "saliency_ds2_topic7", "saliency_ds2_topic8"] # attention数据集
    salList = {0: '1-1-Conan Gore Fly.npy', 1: '1-2-Front.npy', 2: '1-3-360 Google Spotlight Stories_ HELP.npy',
               3: '1-4-Conan Weird Al.npy', 4: '1-5-TahitiSurf.npy', 5: '1-6-Falluja.npy',
               6: '1-7-Cooking Battle.npy', 7: '1-8-Football.npy', 8: '1-9-Rhinos.npy'} # 使用s2cnn训练的数据集

    timeBasePath = 'D:/VR_project/ViewPrediction/frames/timeStamp/'

    if args.salFlag == "cnn_sphere":
        salBasePath = 'D:/VR_project/ViewPrediction/frames/saliency/'
    elif args.salFlag == "sample_attention":
        salBasePath = 'D:/VR_project/ViewPrediction/frames/attentionSaliency/'
        timeBasePath = 'D:/VR_project/ViewPrediction/frames/attentionTimeStamp/'
    elif args.salFlag == "attention":
        salBasePath = 'D:/VR_project/PanoSaliency/data/'
    else:
        salBasePath = 'D:/VR_project/ViewPrediction/frames/attentionSaliency/'
        timeBasePath = 'D:/VR_project/ViewPrediction/frames/attentionTimeStamp/'
        print("Warning! Please choose right saliency dataset! Default: sample_attention!")

    idx = 0
    for videoId in range(idx, 9):       # 视频index --> topic
        print(f'\nTest video_{videoId}...')
        video_time = 0
        if videoId == 7:
            continue
        for userId in range(1, 49): # 用户ID

            optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

            if userId == 3:
                continue
            # modelPath = args.save_path + f"convlstm_online_{fileDict[videoId].lower()}_{userId}.pth"

            if args.salFlag == "cnn_sphere":
                saliencyPath = salBasePath + salDict[videoId]
            elif args.salFlag == "attention":
                saliencyPath = salBasePath + salFileList[videoId]
            else:
                saliencyPath = salBasePath + f"{videoId}-{fileDict[videoId]}.npy"

            try:
                saliency_array = np.array(pickle.load(open(saliencyPath, 'rb'), encoding='bytes'), dtype=object)
            except _pickle.UnpicklingError:
                saliency_array = np.load(saliencyPath, allow_pickle=True)
            total_frame = len(saliency_array)

            saliency_maps, fixation_maps = create_sal_fix(saliency_array, videoId, userId)

            transform = transforms.Compose([transforms.ToTensor()])
            face_dataset = SaliencyDataset(saliency_maps, fixation_maps, args.windows, transform)

            log_path = args.log_path + f'user_{userId}'
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logName = log_path + f"/{fileDict[videoId]}_{userId}.csv"
            with open(logName, 'w', newline='') as f:
                logWriter = csv.writer(f, dialect='excel')
                logWriter.writerow(['FrameId', 'Match', 'Mistake', 'PredictTile', 'RealTile',
                                    'RightTile', 'RedundantTile', 'Acc', 'Recall', 'Preicse', 'Frame'])

                total_test_frame = 0
                good_test_frame = 0
                tileAccList, recallList, precisionList = [], [], []
                thres_recall = 0.6
                start_time = time.time()

                for frame_id in range(args.windows, total_frame-2*args.windows, args.windows):
                    net.initialize_weight()
                    window_start_time = time.time()

                    # online train
                    loss, Mloss, mloss = train(face_dataset, frame_id)
                    test(face_dataset, frame_id+args.windows, window_start_time)

                end_time = time.time()
                user_time = round(end_time - start_time, 5)
                video_time += end_time - start_time
                logWriter.writerows([
                    ['total_time', user_time],
                    ['total_test_frame', total_test_frame],
                    ['good_test_frame', good_test_frame],
                    ['threshold_recall', thres_recall],
                    ['FrameAccuracy', round(good_test_frame / total_test_frame, 4)],
                    ['AverageTileAccuracy', np.mean(tileAccList)],
                    ['AverageRecall', np.mean(recallList)],
                    ['AveragePrecision', np.mean(precisionList)]
                ])

            print(f'Test video={fileDict[videoId]} for userId={userId} finished, time={user_time}s')
            # break
        print(f'Test video_{videoId} finished, time={round(video_time, 4)}s')

