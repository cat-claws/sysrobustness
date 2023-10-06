import torch

# Always start with a configuration dictionary
config = {
	'dataset':'mnist',
	'batch_size':32,
	'attack':'torchattacks.PGD',
	'attack_config':{
		'eps':25/255,
		# 'loss':'ce',
		# 'n_queries':10
		# 'alpha':0.2,
		# 'steps':40,
		# 'random_start':True,
	},
	'device':'cuda' if torch.cuda.is_available() else 'cpu',
	'validation_step':'steps.ordinary_step',
	'attacked_step':'steps.attacked_step'
}

import torch.nn as nn

class Ens(nn.Module):
    def __init__(self, modules = []):
        super().__init__()
        self.submodules = nn.ModuleList(modules)

    def forward(self, x):
        x = [m(x).softmax(-1) for m in self.submodules]
        x = sum(x)
        return x

def main(config):

	from torchiteration import attack

	models = [
			torch.hub.load('cat-claws/nn', 'simplecnn', convs = [], linears = [784, 120, 84], pretrained = 'mlp_784_120_84_GdyC'),
			torch.hub.load('cat-claws/nn', 'exampleconvnet', in_channels = 1, pretrained = 'exampleconvnet_cbyC'),
			torch.hub.load('cat-claws/nn', 'simplecnn', convs = [ (1, 10, 5), (10, 20, 5) ], linears = [320, 50], pretrained = 'simplecnn_5_10_20_50_ibyC'),
			torch.hub.load('cat-claws/nn', 'simplecnn', convs = [ (1, 16, 5), (16, 24, 5) ], linears = [24*4*4, 100], pretrained = 'simplecnn_5_16_24_100_ebyC'),
			torch.hub.load('cat-claws/nn', 'simplecnn', convs = [ (1, 32, 5, 1, 2),  (32, 64, 5, 1, 2)], linears = [64*7*7, 1024], pretrained = 'simplecnn_5_32_64_1024_dbyC')
			]

	m  = Ens(models[:2]).to(config['device'])

	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter(comment = f"_{config['dataset']}_{m._get_name()}_{config['validation_step']}", flush_secs=10)
	
	# Customise how the names are converted to variables

	import steps

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
		])
	dataset1 = datasets.MNIST('', train=True, download=True, transform=transform)
	dataset2 = datasets.MNIST('', train=False, transform=transform)

	train_loader = torch.utils.data.DataLoader(dataset1, num_workers = 4, batch_size = config['batch_size'])
	test_loader = torch.utils.data.DataLoader(dataset2, num_workers = 4, batch_size = config['batch_size'])

	epoch = 0

	for i, m1 in enumerate(models):
		for j, m2 in enumerate(models):

			m.submodules[0] = m1.to(config['device'])
			m.submodules[1] = m2.to(config['device'])

			attack(m,
				val_loader = test_loader,
				epoch = epoch,
				writer = writer,
				atk = config['attack'],
				**config
			)

			epoch += 1
		
	print(m)

	writer.flush()
	writer.close()

if __name__ == "__main__":
	main(config)
