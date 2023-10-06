import torch

config = {
	'dataset':'lfw',
	'batch_size':32,
	'attack':'torchattacks.PGD',
	'attack_config':{
		'eps':8/255,
		# 'loss':'ce',
		# 'n_queries':10
		# 'alpha':0.2,
		# 'steps':40,
		# 'random_start':True,
	},
	'device':'cuda' if torch.cuda.is_available() else 'cpu',
	'validation_step':'steps.pair_classification_step',
	'attacked_step':'steps.attacked_pair_classification_step'
}

import torch.nn as nn
import torch.nn.functional as F
import math

class SiameseClsEns(nn.Module):
	def __init__(self, modules, thresholds):
		super().__init__()
		self.submodules = nn.ModuleList(modules)
		self.thresholds = list(thresholds)

	def forward(self, x):
		x = x * 2 - 1
		res = []
		for m, t in zip(self.submodules, self.thresholds):
			cosine_similarity = F.cosine_similarity(*torch.split(F.normalize(m(x), p=2, dim=1), [len(x) // 2] * 2))
			angle_deg = torch.acos(cosine_similarity - 1e-7) * 180.0 / math.pi
			res.append(torch.stack([angle_deg - t, t - angle_deg], dim = 1))
		return sum(res) / (len(self.submodules) + 1e-7)

def main(config):

	from torchiteration import attack, validate

	models = [
		[torch.hub.load('cat-claws/nn', 'lightcnn_facexzoo', depth = 29, drop_ratio = 0.2, out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'lightcnn_facexzoo'), 77.52],
		[torch.hub.load('cat-claws/nn', 'inception_resnet_v1', num_classes = 8631, pretrained = 'inceptionresnetv1_vggface2'), 64.01],
		[torch.hub.load('cat-claws/nn', 'iresnet', layers = [3, 4, 14, 3], pretrained = 'iresnet50_arcface'), 76.63],
		[torch.hub.load('cat-claws/nn', 'efficientnet_facexzoo', out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'efficientnet_facexzoo'), 76.95],
		[torch.hub.load('cat-claws/nn', 'ghostnet_facexzoo', out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'ghostnet_facexzoo'), 77.78], 
		[torch.hub.load('cat-claws/nn', 'tf_nas_facexzoo',  out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'tfnas_facexzoo'), 75.86],
		[torch.hub.load('cat-claws/nn', 'attentionnet_facexzoo', stage1_modules = 1, stage2_modules = 2, stage3_modules = 3,  out_h = 7, out_w = 7, feat_dim = 512, pretrained = 'attentionnet_facexzoo'), 74],
		[torch.hub.load('cat-claws/nn', 'rexnet_facexzoo', use_se=False, pretrained = 'rexnet_facexzoo'), 76.29],
		[torch.hub.load('cat-claws/nn', 'repvgg_facexzoo', num_blocks = [2, 4, 14, 1], width_multiplier = [0.75, 0.75, 0.75, 2.5], pretrained = 'repvgg_facexzoo'), 76.61]
	]

	import steps

	m = SiameseClsEns(*zip(*models[:2])).to(config['device'])

	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter(comment = f"_{config['dataset']}_{m._get_name()}_{config['validation_step']}", flush_secs=10)

	import sys
	sys.path.insert(0, '../adversarial-attacks-pytorch/')
	import torchattacks

	for k, v in config.items():
		if k.endswith('_step'):
			config[k] = eval(v)
		elif k == 'optimizer':
			config[k] = vars(torch.optim)[v]([p for p in m.parameters() if p.requires_grad], **config[k+'_config'])
			config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['schedulere'])
		elif k == 'adversarial' or k == 'attack':
			config[k] = eval(v)(m, **config[k+'_config'])
		

	import torchvision.transforms as T
	from torchvision import datasets

	from datasets import load_dataset
	lfw = load_dataset('cat-claws/face-verification')['lfw']

	transform=T.Compose([
		T.ToTensor(),
		])

	def transform_(e):
		e['image1'] = [transform(x) for x in e['image1']]
		e['image2'] = [transform(x) for x in e['image2']]
		return e

	test_loader = torch.utils.data.DataLoader(lfw.with_transform(transform_), num_workers=4, batch_size = config['batch_size'])

	# lfw_pairs = datasets.LFWPairs('data', split = 'test', image_set = 'deepfunneled', transform = transform, download = True)
	# test_loader = torch.utils.data.DataLoader(lfw_pairs, num_workers=1, batch_size = config['batch_size'])


	for epoch, m1 in enumerate(models):
		m.submodules[0] = m1[0].to(config['device'])
		m.thresholds[0] = m1[1]

		attack(m,
			val_loader = test_loader,
			epoch = epoch,
			writer = writer,
			atk = config['attack'],
			**config
		)

		print(m)

	writer.flush()
	writer.close()

if __name__ == "__main__":
	main(config)
