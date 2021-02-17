import torch
import torchvision
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import cv2
from autoencoder import AutoEncoder
from autoencoder import ImageDataset

def get_model(path='./models/autoencoder/epoch_'+str(1)+'.pt'):
	model = torch.load(path)
	return model.to('cpu')

def encode(idx):
	imageDataset = ImageDataset()
	image = imageDataset[idx]
	encoded_image = model.features_conv(image.unsqueeze(0))
	return encoded_image, image

def decode(encoded_image, model):
	decoded_image = model.features_deconv(encoded_image)
	decoded_image = decoded_image[:, :, 1:-1, 1:-1]
	return decoded_image

if __name__ == '__main__':
	model = get_model()
	encoded_image, image = encode(0)
	decoded_image = decode(encoded_image, model)
	print("Original Image:", image.shape)
	print("Encoded Image:", encoded_image[0].shape)
	print("Decoded Image:", decoded_image[0].shape)
	print()
	print("MAE:", torch.mean(torch.square(image - decoded_image)).item())

	Image2PIL = torchvision.transforms.ToPILImage()
	image = Image2PIL(image[0])
	image.save('./results/autoencoder/og.png')

	decoded_image = Image2PIL(decoded_image[0])
	decoded_image.save('./results/autoencoder/decoded_image.png')