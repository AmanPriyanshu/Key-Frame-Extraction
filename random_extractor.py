import cv2
import os
import numpy as np
from sys import getsizeof
from tqdm import tqdm

scriptname = os.path.basename(__file__)
scriptname = scriptname[:-3]

file = './dataset/'
path = file+os.listdir(file)[1]
vidcap = cv2.VideoCapture(path)
fps = vidcap.get(cv2.CAP_PROP_FPS)
success,image = vidcap.read()

key_frame = 20
video = []

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

success = True
while success:
	success,image = vidcap.read()
	video.append(image)

height , width , layers =  video[0].shape
new_fps = (fps/key_frame)*2
final_video = cv2.VideoWriter('.\\results\\video_'+scriptname+'.mp4',fourcc,new_fps,(width,height))

count = 0

split_points = np.random.choice(len(video), len(video)//key_frame, replace=False) + 1
split_points.sort()

for index, image in tqdm(enumerate(video), total=len(video)):
	if index in split_points:
		final_video.write(image)
		count += 1

final_video.release()