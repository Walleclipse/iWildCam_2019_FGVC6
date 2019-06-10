from __future__ import absolute_import

import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

TRAIN_DATASET = {'CCT': 'iWildCam_2019_CCT', 'iNat': 'iWildCam_2019_iNat_Idaho',
                 'IDFG': 'iWildCam_IDFG'}  # _images_small


def image_augment(p=.5, cut_size=8):
	imgaugment = A.Compose([
		A.HorizontalFlip(p=0.3),
		A.GaussNoise(p=.1),
		# A.OneOf([
		# A.Blur(blur_limit=3, p=.1),
		#	A.GaussNoise(p=.1),
		# ], p=0.2),
		A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT,
		                   value=(0, 0, 0), p=.3),
		A.RandomBrightnessContrast(p=0.3),
		A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.1),
		A.Cutout(num_holes=1, max_h_size=cut_size, max_w_size=cut_size, p=0.3)
	], p=p)

	return imgaugment


class iWildCam(Dataset):
	def __init__(self, params, mode='train'):
		self.mode = mode
		self.clahe = params['clahe']
		self.gray = params['gray']
		if 'train' in mode:
			clahe_prob = params['clahe_prob']
			gray_prob = params['gray_prob']
		elif mode=='infer':
			clahe_prob = 1
			gray_prob = 1
		else:
			clahe_prob =  int(params['clahe_prob']>=1.0)
			gray_prob = int(params['gray_prob']>=1.0)
		if 'train' in mode:
			print('use train augmented mode')
			self.augment = params['aug_proba'] > 0
			self.label_smooth = params['label_smooth']
		else:
			self.augment = False
			self.label_smooth = False
		self.one_hot = params['loss'] != 'focal' if mode != 'infer' else False
		self.num_classes = params['num_classes']
		self.root = params['data_dir']

		mean_values = [0.3297, 0.3819, 0.3637]
		std_values = [0.1816, 0.1887, 0.1877]

		# mean_values = [0.3045, 0.3625, 0.3575]
		# std_values = [0.1801, 0.1870, 0.1920]

		self.resize = A.Resize(int(params['height'] * 1.1), int(params['width'] * 1.1), interpolation=cv2.INTER_CUBIC,
		                       p=1.0)
		self.crop = A.RandomCrop(params['height'], params['width'], p=1.0) if 'train' in mode else A.CenterCrop(
			params['height'], params['width'], p=1.0)

		if self.clahe:
			self.imgclahe = A.CLAHE(clip_limit=2.0, tile_grid_size=(16, 16), p=clahe_prob)
		if self.gray:
			self.imggray = A.ToGray(p=gray_prob)
		if self.augment:
			self.imgaugment = image_augment(params['aug_proba'], params['cut_size'])

		self.norm = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean_values,
			                     std=std_values),
		])
		if mode == 'train':
			self.file_dir = self.root + 'train_file.csv'  # 'train_file_1.csv'
		elif mode == 'dev' or mode == 'val' or mode == 'validation':
			self.file_dir = self.root + 'dev_file.csv'
		elif mode == 'test' or mode == 'infer':
			self.file_dir = self.root + 'test_file.csv'
		elif mode == 'train_dev' or mode == 'train_val':
			self.file_dir = self.root + 'train_file.csv'
			self.file_dir_1 = self.root + 'dev_file.csv'
		else:
			print('does not exisit!', mode)

		data_file = pd.read_csv(self.file_dir)
		if mode == 'train':
			if not params['CCT']:
				data_file = data_file[data_file['dataset'] != 'CCT']
			if not params['iNat']:
				data_file = data_file[data_file['dataset'] != 'iNat']
		if mode == 'train_dev' or mode == 'train_val':
			temp = pd.read_csv(self.file_dir_1)
			data_file = pd.concat([data_file, temp])

		data_file['absolute_file_name'] = data_file['file_name'].map(lambda x: os.path.join(self.root, x))
		self.image_files = data_file['absolute_file_name'].values
		self.image_ids = data_file['id'].values
		print('dataset len:', len(self.image_files))
		if mode != 'infer':
			self.labels = data_file['category_id'].values

	def __getitem__(self, index):
		id = self.image_ids[index]
		image = cv2.imread(self.image_files[index])
		image = self.resize(image=image)['image']
		if self.clahe:
			image = self.imgclahe(image=image)['image']
		if self.augment:
			image = self.imgaugment(image=image)['image']
		if self.gray:
			image = self.imggray(image=image)['image']
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = self.crop(image=image)['image']
		image = self.norm(image)
		if self.mode != 'infer':
			label = self.labels[index]
			if self.one_hot:
				label = np.eye(self.num_classes)[label]
				if self.label_smooth > 0:
					label = (1 - self.label_smooth) * label + self.label_smooth / self.num_classes
		else:
			label = 0
			if self.one_hot:
				label = np.eye(self.num_classes)[label]

		return (image, label, id)

	def __len__(self):
		return len(self.image_files)

	def _category(self):
		category2id = {
			'empty': 0,
			'deer': 1,
			'moose': 2,
			'squirrel': 3,
			'rodent': 4,
			'small_mammal': 5,
			'elk': 6,
			'pronghorn_antelope': 7,
			'rabbit': 8,
			'bighorn_sheep': 9,
			'fox': 10,
			'coyote': 11,
			'black_bear': 12,
			'raccoon': 13,
			'skunk': 14,
			'wolf': 15,
			'bobcat': 16,
			'cat': 17,
			'dog': 18,
			'opossum': 19,
			'bison': 20,
			'mountain_goat': 21,
			'mountain_lion': 22
		}

		id2category = [
			'empty',
			'deer',
			'moose',
			'squirrel',
			'rodent',
			'small_mammal',
			'elk',
			'pronghorn_antelope',
			'rabbit',
			'bighorn_sheep',
			'fox',
			'coyote',
			'black_bear',
			'raccoon',
			'skunk',
			'wolf',
			'bobcat',
			'cat',
			'dog',
			'opossum',
			'bison',
			'mountain_goat',
			'mountain_lion',
		]


class data_prefetcher():
	def __init__(self, loader, label_type='float'):
		self.loader = iter(loader)
		self.stream = torch.cuda.Stream()
		self.label_type=label_type
		self.preload()

	def preload(self):
		try:
			self.next_input, self.next_target, self.next_ids = next(self.loader)
		except StopIteration:
			self.next_input = None
			self.next_target = None
			self.next_ids =None
			return
		with torch.cuda.stream(self.stream):
			self.next_input = self.next_input.cuda(non_blocking=True)
			self.next_target = self.next_target.cuda(non_blocking=True)
			#self.next_ids = self.next_ids.cuda(non_blocking=True)

			self.next_input = self.next_input.float()
			if self.label_type=='float':
				self.next_target = self.next_target.float()
			else:
				self.next_target = self.next_target.long()

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		input = self.next_input
		target = self.next_target
		ids = self.next_ids
		self.preload()
		return input, target,ids


def get_iwildcam_loader(params, mode='train'):
	if mode == 'train' or mode == 'train_val' or mode == 'train_dev':
		train_data = iWildCam(params, mode=mode)

		train_loader = torch.utils.data.DataLoader(
			train_data, batch_size=params['batch_size'], shuffle=True,
			num_workers=params['threads'], drop_last=True, pin_memory=True)

		dev_data = iWildCam(params, mode='dev')

		dev_loader = torch.utils.data.DataLoader(
			dev_data, batch_size=params['eval_batch_size'], shuffle=False,
			num_workers=params['threads'], drop_last=False, pin_memory=True)
		return train_loader, dev_loader
	elif mode == 'infer':
		test_data = iWildCam(params, mode='infer')

		test_loader = torch.utils.data.DataLoader(
			test_data, batch_size=params['batch_size'], shuffle=False,
			num_workers=params['threads'], drop_last=False, pin_memory=True)
		return test_loader
	else:
		return None


