import torch
from insightface.recognition.arcface_torch.backbones import get_model

import torch
import torchvision
import torch.nn.functional as F

lfw_pairs = torchvision.datasets.LFWPairs('', split = 'test', image_set = 'original', transform = torchvision.transforms.ToTensor(), download = True)
test_loader = torch.utils.data.DataLoader(lfw_pairs, batch_size = 4, num_workers=1)

x1, x2, y = next(iter(test_loader))

# from retinaface import retinaface_resnet50
# detect = retinaface_resnet50('retinaface/weights/retinaface_resnet50_2020-07-20.pth')
# detect = retinaface_resnet50('pretrained/retinaface/retinaface_resnet50_2020-07-20.pth')


model = get_model('r18').cuda()
model.load_state_dict(torch.load('pretrained/arcface/glint360k_cosface_r18_fp16_0.1.pth'))
model.eval()

resize = torchvision.transforms.Resize(112)
class Recognition:
    def __init__(self):
        self.training = False
        self.model = model

    def eval(self):
        pass
        
    def __call__(self, x):
        return model(resize(x) * 2 - 1)

recognition = Recognition()
# print(recognition(resize(x1)))
# print(x1.device)
# print(recognition.device)


# print('done')
# raise

import sys
sys.path.append('../adversarial-attacks-pytorch')
# sys.path.append('adversarial-attacks-pytorch')

import torchattacks
attack = torchattacks.Square_(recognition, device='cuda', norm='Linf', eps=8/255, n_queries=5000, n_restarts=2, p_init=.8, seed=0, verbose=False, loss='cos', resc_schedule=True)
# >>> adv_images = attack(images, labels)

from tqdm import tqdm
results = []
for x1, _, y in tqdm(test_loader):
    # boxes = detect(x1).data
    feat = recognition(x1.cuda())
    adv_images = attack(x1, feat.clone())
    feat_ = recognition(adv_images)

    print(feat.shape, feat_.shape)

    for s in F.cosine_similarity(feat, feat_):
        results.append(s.item())
    # for b1, b2 in zip(boxes, boxes_):
    #     results.append(torchvision.ops.box_iou(b1, b2).item())
        print(sum(results)/len(results))
    # if len(results)>40:
    #     break

# adv_images = result.adv_images
# best = result.best.tolist()

print('robustness based on Cos Sim is: ', round(sum(results)/len(results), 4))