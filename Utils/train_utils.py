import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class cross_entropy(nn.Module):
	""" Cross entropy that accepts soft targets"""

	def __init__(self, size_average=True):
		super(cross_entropy, self).__init__()
		self.size_average = size_average

	def forward(self, input, target):
		logsoftmax = nn.LogSoftmax()
		if self.size_average:
			return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
		else:
			return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


class focal_loss(nn.Module):
	def __init__(self, alpha=1., gamma=1.):
		super(focal_loss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, inputs, targets, **kwargs):
		CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
		pt = torch.exp(-CE_loss)
		F_loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss
		return F_loss.mean()


class focal_loss_3d(nn.Module):
	def __init__(self, gamma=1.0, alpha=None, size_average=True):
		super(focal_loss_3d, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
		if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, input, target):
		if input.dim() > 2:
			input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
		target = target.view(-1, 1)

		logpt = F.log_softmax(input)
		logpt = logpt.gather(1, target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type() != input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0, target.data.view(-1))
			logpt = logpt * Variable(at)

		loss = -1 * (1 - pt) ** self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()


def weights_init(m, init_f=nn.init.kaiming_normal_):
	gain = nn.init.calculate_gain('relu')
	if isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1)
		m.bias.data.zero_()
	elif isinstance(m, nn.Linear):
		init_f(m.weight.data, nonlinearity='relu')
		m.bias.data.zero_()


def adjust_learning_rate_cliff(optimizer, epoch, initial_lr):
	"""Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
	lr = initial_lr * (0.1 ** (epoch // 8))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate_linear(optimizer, epoch, initial_lr):
	lr = initial_lr * (0.9 ** (epoch // 1))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate_traingle(optimizer, epoch, max_lr, min_lr=0.0001, cycle=8):
	valid_epoch = epoch % cycle
	k = (max_lr - min_lr) / (cycle // 2)
	lr = max_lr - abs(valid_epoch - cycle // 2) * k
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate_warmup(optimizer, epoch, max_lr, min_lr=0.0001, cycle=8):
	valid_epoch = epoch % cycle
	valid_max_lr = max_lr * (0.7 ** (epoch // cycle))
	delta_lr = (valid_max_lr - min_lr)
	k = delta_lr / cycle
	if epoch < cycle:
		if epoch <= cycle // 2:
			lr = min_lr + 0.5 * delta_lr + valid_epoch * k
		else:
			lr = max_lr - (epoch - cycle // 2) * 2 * k
	else:
		lr = valid_max_lr - (valid_epoch - cycle) * k
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def get_optimizer(params, model):
	param_groups = model.parameters()

	if params['optim'] == 'adam':
		optimizer = torch.optim.Adam(param_groups, lr=params['lr'],weight_decay=params['weight_decay'])
	else:
		optimizer = torch.optim.SGD(param_groups, lr=params['lr'], momentum=0.9, nesterov=True,weight_decay=params['weight_decay'])

	return optimizer


def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
