import torch
import torchvision
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import cv2

class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, path='./dataset/images/', transform=torchvision.transforms.ToTensor()):
		self.path = path
		self.files = [path+i for i in os.listdir(path)]
		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		img = cv2.imread(self.files[idx])
		img = cv2.resize(img, (16*20, 9*20))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img.astype(np.uint8))

		if self.transform:
			img = self.transform(img)

		return img

class AutoEncoder(torch.nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		# feature encoder
		self.features_conv = torch.nn.Sequential(
			torch.nn.Conv2d(3, 4, kernel_size=3, stride=2),
			torch.nn.ReLU(),
			torch.nn.Conv2d(4, 6, kernel_size=3, stride=2),
			torch.nn.ReLU(),
			torch.nn.Conv2d(6, 8, kernel_size=3),
			torch.nn.ReLU(),
			)
		# feature decoder
		self.features_deconv = torch.nn.Sequential(
			torch.nn.ConvTranspose2d(8, 6, 4, stride=2),
			torch.nn.ReLU(),
			torch.nn.ConvTranspose2d(6, 4, 4, stride=2),
			torch.nn.ReLU(),
			torch.nn.ConvTranspose2d(4, 3, 9),
			torch.nn.ReLU(),
			)

	def forward(self, x):
		x = self.features_conv(x)
		x = self.features_deconv(x)
		x = x[:, :, 1:-1, 1:-1]
		return x

def train():

	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'

	model = AutoEncoder()
	model = model.to(device)
	imageDataset = ImageDataset()
	imageLoader = torch.utils.data.DataLoader(imageDataset, batch_size=4, shuffle=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	for epoch in range(1):
		progression = tqdm(imageLoader, total=len(imageLoader))
		running_loss = []
		for images in progression:
			images = images.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = torch.mean(torch.square(outputs - images))
			loss.backward()
			optimizer.step()
			running_loss.append(loss.item())
			progression.set_description(str({'epoch':epoch+1, 'loss': round(sum(running_loss)/len(running_loss), 4), 'pixel_error': int(255*round(sum(running_loss)/len(running_loss), 4))}))
		progression.close()
		torch.save(model, './models/autoencoder/epoch_'+str(epoch+1)+'.pt')

if __name__ == '__main__':
	path = './dataset/images/'
	files = [path+i for i in os.listdir(path)]
	train()