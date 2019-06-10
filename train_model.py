# coding=utf-8
from __future__ import absolute_import, print_function
import os
import json
import torch
from time import time
import numpy as np
import pandas as pd
from sklearn import metrics
from glob import glob

from DataSet.dataset import get_iwildcam_loader, data_prefetcher
from Utils.train_utils import cross_entropy,focal_loss, get_optimizer
from Utils.train_utils import mixup_data, mixup_criterion
from Models.model_factory import create_model

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

def evaluate(model, data_loader, criterion,use_onehot=True):
	y_pred, y_true, losses=[],[],[]
	with torch.no_grad():
		inputs, labels, ids = data_loader.next()
		while inputs is not None:
			if use_onehot:
				targets = np.argmax(labels.cpu().detach().numpy(), axis=1)
			else:
				targets = labels.cpu().detach().numpy()
			y_true.extend(targets)
			output = model(inputs)
			loss = criterion(output, labels)
			y_pred.extend(np.argmax(output.cpu().detach().numpy(), axis=1))
			losses.append(loss.cpu().detach().numpy())

			inputs, labels, ids = data_loader.next()

	acc = metrics.accuracy_score(y_true, y_pred)
	f1 = metrics.f1_score(y_true, y_pred, average='macro')
	loss_val=np.mean(losses)
	return loss_val, acc, f1

def train(params):

	if params['init_model'] is not None:
		model = torch.load(params['init_model'])
		print('load model', params['init_model'])
	else:
		model = create_model(
			params['Net'],
			pretrained=params['pretrained'],
			num_classes=params['num_classes'],
			drop_rate=params['drop_rate'],
			global_pool='avg',
			bn_tf=False,
			bn_momentum=0.99,
			bn_eps=1e-3,
			checkpoint_path=params['init_model'],
			in_chans=3)

	optimizer = get_optimizer(params,model)
	param_num = sum([p.data.nelement() for p in model.parameters()])
	print("Number of model parameters: {} M".format(param_num / 1024 / 1024))
	model = model.to(device)
	model.train()

	if params['lr_schedule']:
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['lr_decay_epochs'], gamma=0.2)
	if params['loss'] =='ce' or params['loss'] =='cross_entropy':
		criterion = cross_entropy().to(device)
		label_type = 'float'
	elif params['loss'] =='focal':
		criterion = focal_loss(gamma=1.0, alpha=1.0).to(device)
		label_type='long'
	else:
		print('no exist loss',params['loss'])
	train_data_loader, dev_data_loader = get_iwildcam_loader(params,mode=params['mode'])

	train_log=[]
	dev_log=[]
	best_acc, best_f1, best_epoch=0,0,0
	t1 = time()
	print('begin to train')
	use_onehot=params['loss']!='focal'
	for epoch in range(params['epochs']):
		train_loader = data_prefetcher(train_data_loader,label_type)
		inputs, labels, ids = train_loader.next()
		i = 0
		while inputs is not None:
			mixup_now = np.random.random()<params['aug_proba']
			if params['mixup'] and mixup_now:
				inputs, labels_a, labels_b, lam = mixup_data(inputs, labels,
			                                               params['mixup_alpha'])


			optimizer.zero_grad()
			output = model(inputs)
			if params['mixup'] and mixup_now:
				loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
			else:
				loss = criterion(output, labels)
			loss.backward()
			optimizer.step()

			if i % params['print_step'] == 0:
				preds = np.argmax(output.cpu().detach().numpy(), axis=1)
				if use_onehot:
					targets = np.argmax(labels.cpu().detach().numpy(), axis=1)
				else:
					targets = labels.cpu().detach().numpy()
				acc = metrics.accuracy_score(targets, preds)
				loss_val = loss.cpu().detach().numpy()
				f1 = metrics.f1_score(targets,preds,average='macro')
				train_log.append([epoch,i, loss_val, acc, f1])
				print("epoch: %d, iter: %d, train_loss: %.4f, train_acc: %.4f, train_f1: %.4f, time_cost_per_iter: %.4f s" % (
				epoch, i, loss_val, acc, f1,(time() - t1)/params['print_step']))
				with open(params['log_dir'] + 'train.tsv', 'a') as f:
					f.write('%05d\t%05d\t%f\t%f\t%f\n' % (epoch, i, loss_val, acc, f1))
				t1 = time()

			if (i+1) % params['save_step'] == 0:
				save_model_path= os.path.join(params['save_dir'], 'model_%d_%d.pkl' % (epoch,i))
				torch.save(model,save_model_path)
				print('save model to',save_model_path)

			if (i+1) % params['eval_step'] == 0:
				t2=time()
				model.eval()
				data_loader = data_prefetcher(dev_data_loader,label_type)
				loss_val, acc, f1 = evaluate(model, data_loader, criterion,use_onehot)
				model.train()
				dev_log.append([epoch,i, loss_val, acc, f1])

				if f1 > best_f1:
					best_acc, best_f1, best_epoch = acc, f1, epoch
				print('[Evaluation] -------------------------------')
				print("epoch: %d, test acc: %.4f, f1-score: %.4f, loss: %.4f, best-f1-score: %.4f, eval_time: %.4f s" % (
					epoch, acc, f1, loss_val, best_f1,time()-t2))
				print('[Evaluation] -------------------------------')

				with open(params['log_dir'] + 'eval.tsv', 'a') as f:
					f.write('%05d\t%05d\t%f\t%f\t%f\n' % (epoch, i, loss_val, acc, f1))

			inputs, labels, ids = train_loader.next()
			i += 1

		if params['lr_schedule']:
			scheduler.step(epoch)

	return model

def get_params():
	params = {
		'mode':'train_val',
		'data_dir': 'data/bbox/cropped_image/', #['data/bbox/cropped_image/','data/']
		'CCT':True,
		'iNat':True,
		'save_dir': 'final_output/output_0/',
		'init_model': None,#'output_1/resnet_101_3_3427.pkl',
		'Net': 'tf_efficientnet_b0',  # 'resnet','wideresnet','tf_efficientnet_b0'
		'pretrained': True,
		'drop_rate':0.2,

		'batch_size': 32,
		'eval_batch_size': 32,
		'num_classes': 23,
		'epochs': 6,
		'print_per_epoch':500,
		'eval_per_epoch': 4,
		'save_per_epoch': 4,

		'loss':'ce',#['ce','focal']
		'lr_schedule': True,
		'lr': 5e-3,
		'weight_decay':1e-6,
		'optim': 'adam',
		'lr_decay_epochs':[2,4],

		'clahe':True,
		'clahe_prob': 0.2,
		'gray':True,
		'gray_prob':0.01,
		'aug_proba':0.5,
		'cut_size':8,
		'label_smooth':0.01,
		'mixup':True,
		'mixup_alpha':1,
		'height':64,#380,#224 resnet, 300
		'width':64,
		'threads':2,

	}
	params['log_dir'] = os.path.join(params['save_dir'], 'log/')
	if not os.path.exists(params['save_dir']):
		os.mkdir(params['save_dir'])
	if not os.path.exists(params['log_dir']):
		os.mkdir(params['log_dir'])
		with open(params['log_dir'] + 'eval.tsv', 'a') as f:
			f.write('Epoch\tStep\tLoss\tAccuracy\tF1-Score\n')
		with open(params['log_dir'] + 'train.tsv', 'a') as f:
			f.write('Epoch\tStep\tLoss\tAccuracy\tF1-Score\n')
	root = params['data_dir']
	params['train_data_size'] = len(pd.read_csv(root + 'train_file.csv'))
	params['dev_data_size'] = len(pd.read_csv(root + 'dev_file.csv'))
	params['step_per_epoch'] = params['train_data_size'] // params['batch_size']
	params['print_step'] = max(1,params['step_per_epoch']//params['print_per_epoch'])
	params['eval_step'] = max(1,params['step_per_epoch']//params['eval_per_epoch'])
	params['save_step'] = max(1,params['step_per_epoch']//params['save_per_epoch'])

	json.dump(obj=params, fp=open(params['log_dir'] + 'parameters.json', 'w'))
	print(params)

	return params

def load_params(save_dir):
	params_path=save_dir + 'log/parameters.json'
	print('load params form',params_path)
	params = json.load(fp=open(params_path, 'r'))
	ckpts = glob(save_dir+'*.pkl')
	if len(ckpts)>0:
		ckpts = sorted(ckpts, key=lambda x: eval(x.split('/')[-1].split('.')[0].split('_')[-1]))
		params['init_model']=ckpts[-1]
	print(params)
	return params

def main():
	params = get_params()
	train(params)

if __name__ == '__main__':
	main()






