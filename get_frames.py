import cv2
import os
import numpy as np

fileList = ['1-1-Conan Gore Fly.mp4', '1-2-Front.mp4', '1-3-360 Google Spotlight Stories_ HELP.mp4',
            '1-4-Conan Weird Al.mp4', '1-5-TahitiSurf.mp4', '1-6-Falluja.mp4',
            '1-7-Cooking Battle.mp4', '1-8-Football.mp4', '1-9-Rhinos.mp4']

fileName = fileList[-1]

videoPath = 'D:/VR_project/LiveDeep_All/videos/' + fileName
savePath = './frames/' + fileName.split('.')[0]
timePath = './frames/timeStamp/'
if not os.path.exists(savePath):
    os.makedirs(savePath)

timeList = []

cameraCapture = cv2.VideoCapture(videoPath)
fps = cameraCapture.get(cv2.CAP_PROP_FPS)
iVideoTime = cameraCapture.get(cv2.CAP_PROP_FRAME_COUNT) / cameraCapture.get(cv2.CAP_PROP_FPS)
frames = cameraCapture.get(cv2.CAP_PROP_FRAME_COUNT)
sample_rate = 4
frameId, idx = 1, 1
dsize = (512, 256)
while idx <= frames:
    success, frame = cameraCapture.read()
    if not success:
        print(f'\nRead finish! Have gotten {frameId-1} sampled frames!')
        break

    milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
    seconds = milliseconds // 1000
    milliseconds = milliseconds % 1000

    if idx % int(fps/sample_rate) == 0:
        cv2.imwrite(savePath+f'/{frameId}.jpg', cv2.resize(frame, dsize))
        frameId += 1
        print('\r{0}.{1}/{2}'.format(int(seconds), int(milliseconds), round(iVideoTime,3)), end='')
        timeList.append(float(str(int(seconds)) + '.' + str(int(milliseconds))))

    idx += 1
np.savetxt(timePath + fileName.split('.')[0]+'.txt', np.array(timeList), fmt='%.3f')

cv2.destroyAllWindows()
cameraCapture.release()

