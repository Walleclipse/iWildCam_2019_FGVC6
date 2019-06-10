
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import json
from glob import glob
from time import time
import cv2

TRAIN_DATASET={'CCT':'iWildCam_2019_CCT','iNat':'iWildCam_2019_iNat_Idaho','IDFG':'iWildCam_IDFG'} #_images_small
DATA_DIR='./data/'
ANNOTATION_DIR =DATA_DIR+ 'iWildCam_2019_Annotations/'

# CCT orig: 196157 bbox:196015
#orig_list 196086 bbox_list 196015
# iNat orig: 24813 bbox:25263
# IDFG orig: 153730 bbox:153730


def crop_image(img_names,ws,ids,img2det,bbox_detect_dir):
	print('images num:',len(img_names))
	print('detection num:', len(img2det))
	t1=time()
	miss=0
	for ii in range(len(img_names)):
		img_file = img_names[ii]
		if os.path.exists(bbox_detect_dir + 'bbox_temp/' + img_file):
			continue
		dirs = img_file.split('/')
		for jj in range(len(dirs)):
			now_dir = '/'.join(dirs[:jj])
			temp_dir = bbox_detect_dir + 'bbox_temp/' + now_dir
			if not os.path.exists(temp_dir):
				os.mkdir(temp_dir)
			crop_dir = bbox_detect_dir + 'cropped_image/' + now_dir
			if not os.path.exists(crop_dir):
				os.mkdir(crop_dir)
		img_id = ids[ii]
		image = cv2.imread(DATA_DIR+img_file)
		iBox = 0
		try:
			box = img2det[img_id][iBox]
		except KeyError as e:
			print(e)
			miss+=1
		#	with open(bbox_detect_dir+'bug_img/img_list.txt','a') as f:
		#		f.write(img_file+'\t'+img_id+'\n')
			continue

		imageWidth = image.shape[1]

		ratio = imageWidth / ws[ii]
		box_new = [x * ratio for x in box]
		buffer_scale=0.2
		ww = max(0, int((box_new[3] - box_new[1]) * buffer_scale))
		hh = max(0, int((box_new[2] - box_new[0]) * buffer_scale))

		topRel = int(max(0, box_new[0] - hh))
		leftRel = int(max(0, box_new[1] - ww))
		bottomRel = int(box_new[2] + hh)
		rightRel = int(box_new[3] + ww)

		cropped = image.copy()[leftRel:rightRel, topRel:bottomRel] #.copy()
		if len(cropped==0):
			with open(bbox_detect_dir + 'cropped_image/zero_crop.txt','a') as f:
				f.write(img_file+'\n')
		cv2.imwrite(bbox_detect_dir + 'cropped_image/' + img_file,cropped)


		#img_det = cv2.rectangle(image.copy(),(topRel, leftRel), (bottomRel, rightRel), (0, 255, 0), 3)
		#cv2.imwrite(bbox_detect_dir + 'bbox_temp/' + img_file, img_det)
		if ii%100==0:
			print('processing image',ii,(time()-t1)/100)
			t1 = time()
	print('all miss data',miss)

def detect_train_images(bbox_detect_dir='data/bbox/',CCT=True,iNat=True):
	train_file=pd.read_csv(DATA_DIR+'train_file.csv')
	dev_file = pd.read_csv(DATA_DIR + 'dev_file.csv')
	data_file=pd.concat([train_file,dev_file])

	if CCT:
		print('begin to crop CCT data')
		data_cct = data_file[data_file['dataset']=='CCT'].reset_index(drop=True)
		img_names = data_cct['file_name'].values
		ids = data_cct['id'].values
		ws = data_cct['width'].values

		img2det = {}
		with open(bbox_detect_dir+'Detection_Results/CCT_Detection_Results_1.p', 'rb') as data_file:
			temp = pickle.load(data_file, encoding='iso-8859-1')
		for img, res in zip(temp['images'], temp['detections']):
			img2det[img] = res[:10]
		with open(bbox_detect_dir+'Detection_Results/CCT_Detection_Results_2.p', 'rb') as data_file:
			temp = pickle.load(data_file, encoding='iso-8859-1')
		for img, res in zip(temp['images'], temp['detections']):
			img2det[img] = res[:10]

		crop_image(img_names, ws, ids, img2det, bbox_detect_dir)

	if iNat:
		print('begin to crop iNat data')
		data_cct = data_file[data_file['dataset']=='iNat'].reset_index(drop=True)
		img_names = data_cct['file_name'].values
		ids = data_cct['id'].values
		ws = data_cct['width'].values

		img2det = {}
		with open(bbox_detect_dir+'Detection_Results/iNat_Idaho_Detection_Results.p', 'rb') as data_file:
			temp = pickle.load(data_file, encoding='iso-8859-1')
		for img, res in zip(temp['images'], temp['detections']):
			img2det[img] = res[:10]

		crop_image(img_names, ws, ids, img2det, bbox_detect_dir)


def detect_test_images(bbox_detect_dir='data/bbox/'):
	print('detect test image')
	data_file=pd.read_csv(DATA_DIR+'raw_data/test_file_orig.csv')
	print('test_file',data_file.shape)

	img_names = data_file['file_name'].values
	ids = data_file['id'].values
	ws = data_file['height'].values

	img2det = {}
	with open(bbox_detect_dir + 'Detection_Results/IDFG_Detection_Results_1.p', 'rb') as data_file:
		temp = pickle.load(data_file, encoding='iso-8859-1')
	for img, res in zip(temp['images'], temp['detections']):
		img2det[img] = res[:10]
	with open(bbox_detect_dir + 'Detection_Results/IDFG_Detection_Results_2.p', 'rb') as data_file:
		temp = pickle.load(data_file, encoding='iso-8859-1')
	for img, res in zip(temp['images'], temp['detections']):
		img2det[img] = res[:10]

	crop_image(img_names, ws, ids, img2det, bbox_detect_dir)

def rewrite_cropped_csv():
	def check_file(df,prefix_dir='bbox/cropped_image/',name='train'):
		df=df.reset_index(drop=True)
		print(name,df.shape)
		#new_df=pd.DataFrame()
		t1=time()
		file_names=df['file_name'].values
		valid_ind=[]
		new_width,new_height=[],[]
		for ii, file in enumerate(file_names):
			new_path = DATA_DIR + prefix_dir + file
			if os.path.exists(new_path):
				try:
					img = cv2.imread(new_path)
					sh=img.shape
				except:
					continue
				new_width.append(sh[1])
				new_height.append(sh[0])
				valid_ind.append(ii)
			if ii%1000==0:
				print("datatype: %s, index: %d, time: %f min, data len %d" %(name,ii, (time()-t1)/60),len(valid_ind))

		new_df = df.iloc[valid_ind]
		new_df['new_width']=new_width
		new_df['new_height'] = new_height
		print('new_df for:', name, new_df.shape)
		new_df.to_csv(DATA_DIR+prefix_dir+name+'.csv',index=False)
		return new_df
	train = pd.read_csv(DATA_DIR+'train_file.csv')
	new_train=check_file(train,name='train_file')
	dev = pd.read_csv(DATA_DIR+'dev_file.csv')
	new_dev=check_file(dev,name='dev_file')
	test = pd.read_csv(DATA_DIR+'test_file.csv')
	new_test=check_file(test,name='test_file')



def main():
	detect_train_images(CCT=True,iNat=True)
	#merge_test_bbox()
	detect_test_images()
	#extract_miss_bbox(dataset='iNat')
	#rewrite_cropeped_csv()

if __name__=='__main__':
	main()



