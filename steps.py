import torch
import torch.nn as nn
from torch.nn import functional as F

def ordinary_classification_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	if net.training:
		inputs = (0.3 * torch.randn_like(inputs) + inputs).detach()
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores = net(inputs)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def pair_classification_step(net, batch, batch_idx, **kw):
	x1, x2, labels = batch['image1'], batch['image2'], batch['target']
	inputs = torch.cat((x1, x2), dim = 0)
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores = net(inputs)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def pad2_classification_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs = F.pad(inputs, [2] * 4)
	if net.training:
		inputs = (0.3 * torch.randn_like(inputs) + inputs).detach()
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores = net(inputs)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def attacked_classification_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def attacked_pair_classification_step(net, batch, batch_idx, **kw):
	x1, x2, labels = batch['image1'], batch['image2'], batch['target']
	inputs = torch.cat((x1, x2), dim = 0)
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}