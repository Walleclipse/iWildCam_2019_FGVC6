# coding=utf-8
from __future__ import absolute_import, print_function
import os
import torch
from time import time
import pandas as pd
import numpy as np
from glob import glob
from DataSet.dataset import get_iwildcam_loader, data_prefetcher


import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


def infer(params):

	model = torch.load(params['init_model'])
	print('load model', params['init_model'])
	model = model.to(device)
	model.eval()

	infer_loader = get_iwildcam_loader(params,mode='infer')
	infer_loader = data_prefetcher(infer_loader)
	y_preds, y_ids=[],[]
	logits_preds = []
	t1 = time()
	print('begin to infer')
	with torch.no_grad():
		inputs, labels, ids = infer_loader.next()
		i = 0
		while inputs is not None:
			output = model(inputs)
			output = torch.nn.functional.softmax(output,dim=-1)
			output = output.cpu().detach().numpy()
			logits_preds.extend(output)
			y_preds.extend(np.argmax(output, axis=1))
			y_ids.extend(ids)

			if (i+1) % params['print_step'] == 0:
				print("iter: %d,  time_cost_per_iter: %.4f s" % (i, (time() - t1)/params['print_step']))
				t1 = time()
			i += 1
			inputs, labels, ids = infer_loader.next()

	pred_df = {'Id':y_ids,'Predicted':y_preds}
	pred_df=pd.DataFrame(pred_df)
	save_path=params['save_pred_dir']+params['init_model'].split('/')[-1].split('.')[0]+'_preds.csv'
	pred_df.to_csv(save_path,index=False)

	logits_df = {'Id':y_ids,'Predicted':y_preds,'Logits':list(logits_preds)}
	logits_df=pd.DataFrame(logits_df)
	save_path=params['save_pred_dir']+params['init_model'].split('/')[-1].split('.')[0]+'_logits.csv'
	logits_df.to_csv(save_path,index=False)

	print('pred done',pred_df.shape)

	return pred_df

def get_params():
	params = {
		'mode':'infer',
		'data_dir': 'data/bbox/cropped_image/', #data/bbox/cropped_image/'
		'save_pred_dir': 'final_output/output_0/log/',
		'use_org_and_crop': True,
		'init_model': 'final_output/model_5_6827.pkl',
		'batch_size': 128,
		'num_classes': 23,
		'print_step':10,

		'clahe':True,
		'gray':False,
		'height':64,
		'width':64,
		'threads':2,

	}
	print(params)
	return params

def merge_pred_logit(pred_logit_files,save_dir):
	def get_proba(x):
		proba = [eval(num) for num in x[1:-1].split()]
		return proba

	df=pd.DataFrame()
	for ii, pf in enumerate(pred_logit_files):
		temp=pd.read_csv(pf)
		print('process file:',ii, pf, temp.shape)
		temp['Logits']=temp['Logits'].map(lambda x: get_proba(x))
		temp=temp.rename(columns={'Predicted':'Predicted'+str(ii),'Logits':'Logits'+str(ii)})
		if len(df)==0:
			df=temp
		else:
			df=pd.merge(df,temp,on=['Id'],how='inner')

	print('merged file',df.shape)
	mean_probas=[list(df['Logits'+str(ii)].values) for ii in range(len(pred_logit_files))]
	mean_probas=np.array(mean_probas)
	mean_probas=np.mean(mean_probas,axis=0)
	df['Probability'] = list(mean_probas)
	df['Predicted'] = np.argmax(mean_probas,axis=1)
	df[['Id','Predicted']].to_csv(save_dir + 'merged_submission.csv', index=False)
	df.to_csv(save_dir+'merged_proba.csv', index=False)
	print('mean proba done',df.shape)



def main():
	params = get_params()
	infer(params)

def merge_pred():
	save_dir = 'final_output/pred/'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	output_dir=[
		'output_0','output_1','output_2','output_3',
		'output_4','output_5','output_6','output_7',
		'output_8','output_9',
	]
	output_dir = ['final_output/' + x + '/log/' for x in output_dir]
	pred_logit_files =[]
	for outf in output_dir:
		logf = glob(outf+'*_logits.csv')
		pred_logit_files += logf
	print('pred_logit_files',len(pred_logit_files))
	print(pred_logit_files)
	merge_pred_logit(pred_logit_files, save_dir)

if __name__ == '__main__':
	main()






