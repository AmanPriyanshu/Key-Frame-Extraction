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

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

c = 0
success = True
progress = tqdm()
while success:
	success,image = vidcap.read()
	try:
		cv2.imwrite('.\\dataset\\images\\img_'+'0'*(4-len(str(c)))+str(c)+'.png', image)
	except:
		break
	c += 1
	progress.update(1)
progress.close()