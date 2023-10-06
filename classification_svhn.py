import torch

config = {
	'dataset':'svhn',
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
	'validation_step':'steps.ordinary_classification_step',
	'attacked_step':'steps.attacked_classification_step'
}

def main(config):

	from torchiteration import attack

	from pytorchcv.model_provider import get_model
	models = [
				get_model('wrn16_10_svhn', pretrained = True),
				get_model('wrn28_10_svhn', pretrained = True),
				get_model('seresnet56_svhn', pretrained = True),
				get_model('preresnet56_svhn', pretrained = True),
				get_model('rir_svhn', pretrained = True),
				get_model('seresnet20_svhn', pretrained = True),
				get_model('pyramidnet110_a84_svhn', pretrained = True),
				get_model('sepreresnet20_svhn', pretrained = True),
				get_model('resnet56_svhn', pretrained = True) ,
				get_model('pyramidnet110_a48_svhn', pretrained = True),
				get_model('resnet20_svhn', pretrained = True),
				get_model('resnet110_svhn', pretrained = True),
				get_model('preresnet20_svhn', pretrained = True),
				get_model('ror3_56_svhn', pretrained = True),
				get_model('nin_svhn', pretrained = True),
			]

	import steps

	from mnist import Ens
	m  = Ens(models[:1]).to(config['device'])

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

	transform=T.Compose([
		T.ToTensor(),
		T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		])
	dataset1 = datasets.SVHN('data', split='train', download=True, transform=transform)
	dataset2 = datasets.SVHN('data', split='test', download = True, transform=transform)

	train_loader = torch.utils.data.DataLoader(dataset1, num_workers = 4, batch_size = config['batch_size'])
	test_loader = torch.utils.data.DataLoader(dataset2, num_workers = 4, batch_size = config['batch_size'])

	for epoch, m1 in enumerate(models):
		m.submodules[0] = m1.to(config['device'])

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