
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import json
import cv2
from time import time
import threading
import math

DATASET={'CCT':'iWildCam_2019_CCT','iNat':'iWildCam_2019_iNat_Idaho','IDFG':'iWildCam_IDFG'} #_images_small
DATA_DIR='./data/'
ANNOTATION_DIR =DATA_DIR+ 'iWildCam_2019_Annotations/'



def rewrite_train_data_json(dataset='CCT'):
	json_path=ANNOTATION_DIR+DATASET[dataset]+'.json'

	json_data = json.load(open(json_path,'r'))
	images = json_data['images']
	annotations = json_data['annotations']
	csv_data={'category_id':[],'date_captured':[],'id':[],'file_name':[],
	          'rights_holder':[],'width':[],'height':[],'location':[]}
	print('len of  data:',dataset,len(images))
	for ii,(img, annot) in enumerate(zip(images,annotations)):
		if img['id'] != annot['image_id']:
			print('there are some error in',ii,img['id'],annot['image_id'])
		if 'date_captured' in img:
			date=img['date_captured']
		elif 'datetime' in img:
			date = img['datetime']
		else:
			date = json_data['info']['date_created']
		csv_data['date_captured'] += [date]
		csv_data['category_id'] += [annot['category_id']]
		csv_data['file_name'] += [img['file_name']]
		csv_data['rights_holder'] += [img['rights_holder']]
		csv_data['id'] += [img['id']]
		csv_data['width'] += [img['width']]
		csv_data['height'] += [img['height']]
		if 'location' in img:
			locat = img['location']
		else:
			locat=-1
		csv_data['location'] += [locat]


	csv_data = pd.DataFrame(csv_data)
	csv_data.to_csv(ANNOTATION_DIR+DATASET[dataset]+'.csv',index=False)


def split_train_dev(CCT=True,iNat=True):
	columns=['category_id','date_captured','id','file_name',
	          'rights_holder','width','height','location']
	train=pd.DataFrame()
	if CCT:
		temp=pd.read_csv(ANNOTATION_DIR+DATASET['CCT']+'.csv')[columns]
		temp['dataset'] = 'CCT'
		temp['file_name'] = temp['file_name'].map(lambda x:'iWildCam_2019_CCT_images_small/'+x)
		print('use CCT data',temp.shape)
		train=pd.concat([train,temp])

	if iNat:
		temp=pd.read_csv(ANNOTATION_DIR+DATASET['iNat']+'.csv')[columns]
		temp['dataset'] = 'iNat'
		temp['file_name'] = temp['file_name'].map(lambda x: 'iWildCam_2019_iNat_Idaho/' + x)
		print('use iNat data',temp.shape)
		train=pd.concat([train,temp])


	print('train shape',train.shape)
	#train=train.sample(frac=1,random_state=0).reset_index(drop=True)

	dev_file = train[train['location'] == 46]  # 46
	train_file = train[train['location'] != 46]


	train_file.to_csv(DATA_DIR+'train_file.csv',index=False)
	dev_file.to_csv(DATA_DIR+'dev_file.csv',index=False)

	print('category ratio for train data:')
	cnt = Counter(train_file['category_id'].values)
	L = len(train_file)
	for ii in range(23):
		print(ii, cnt[ii], cnt[ii] / L)

	print('category ratio for dev data:')
	cnt = Counter(dev_file['category_id'].values)
	L = len(dev_file)
	for ii in range(23):
		print(ii, cnt[ii], cnt[ii] / L)


def save_test():
	columns=['date_captured','id','file_name',
	          'rights_holder','width','height','location']
	test = pd.read_csv(DATA_DIR+'test.csv')[columns]
	test['dataset'] = 'test'
	test['category_id'] = -1
	test['file_name'] = test['file_name'].map(lambda x:'test_images/'+x)
	print('test shape',test.shape) #153730

	test.to_csv(DATA_DIR+'test_file.csv',index=False)

full_data_dir='data/raw_data/iWildCam_2019_IDFG/iWildCam_IDFG_images/'
def get_test_orig_size_split(test_file,name=0):
	name=str(name)
	print('get_test_orig_size_split for thread',name,test_file.shape)
	file_names= test_file['file_name'].values
	width,height=[],[]
	t1=time()
	for ii,fname in enumerate(file_names):
		mod_name =full_data_dir + fname.split('/')[-1]
		image = cv2.imread(mod_name)
		s = image.shape
		#imageHeight = s[0]
		#imageWidth = s[1]
		width.append(s[0])
		height.append(s[1])
		if ii%100==0:
			print('threads %s, index %d, time-cost %f min'%(name,ii,(time()-t1)/60))
		if ii % 1000 == 0:
			joblib.dump([ii,width,height],DATA_DIR+'raw_data/test_size_temp_{}.pkl'.format(name))
	test_file['width']=width
	test_file['height'] = height
	print(name,'test shape',test_file.shape) #153730

	test_file.to_csv(DATA_DIR+'raw_data/test_file_orig_{}.csv'.format(name),index=False)

def get_test_size_multi_thread(thread_num=1):
	test_file = pd.read_csv(DATA_DIR+'test_file.csv')
	test_file['small_width']=test_file['width']
	test_file['small_height'] = test_file['height']
	chunk=math.ceil(len(test_file)/thread_num)
	thread_list=[]
	for ii in range(thread_num):
		sup_file=test_file.iloc[ii*chunk:(ii+1)*chunk]
		thr=threading.Thread(target=get_test_orig_size_split,args=(sup_file,ii))
		thread_list.append(thr)
	for t in thread_list:
		t.setDaemon(True)
		t.start()
	for t in thread_list:
		t.join()

def merge_test_size_file():
	data=pd.DataFrame()
	for name in range(10):
		data_path=DATA_DIR + 'raw_data/test_file_orig_{}.csv'.format(str(name))
		temp=pd.read_csv(data_path)
		data=pd.concat([data,temp])
		print(name,data.shape)

	data.to_csv(DATA_DIR + 'raw_data/test_file.csv',index=False)

def main(CCT=True,iNat=True):
	if CCT:
		rewrite_train_data_json('CCT')
	if iNat:
		rewrite_train_data_json('iNat')

	split_train_dev(CCT=CCT,iNat=iNat)
	save_test()

if __name__=='__main__':
	main()
	#get_test_size_multi_thread(thread_num=10)
	#merge_test_size_file()
