import torch
from insightface.recognition.arcface_torch.backbones import get_model

import torch
import torchvision
import torch.nn.functional as F

lfw_pairs = torchvision.datasets.LFWPairs('', split = 'test', image_set = 'original', transform = torchvision.transforms.ToTensor(), download = True)
test_loader = torch.utils.data.DataLoader(lfw_pairs, batch_size = 4, num_workers=1)

x1, x2, y = next(iter(test_loader))

from retinaface import retinaface_resnet50
detect = retinaface_resnet50('pretrained/retinaface/retinaface_resnet50_2020-07-20.pth')
# detect = retinaface_resnet50('pretrained/retinaface/retinaface_resnet50_2020-07-20.pth')


model = get_model('r18').cuda()
model.load_state_dict(torch.load('pretrained/arcface/glint360k_cosface_r18_fp16_0.1.pth'))
model.eval()

resize = torchvision.transforms.Resize((112, 112))
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

import torch

def crop_images_with_boxes(images, boxes):
    boxes = boxes.squeeze(1)
    # print(images.shape, boxes.shape)
    # Extract the box coordinates
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Compute the crop dimensions
    batch_size, channels, h, w = images.size()
    x1_ = torch.clamp(x1, 0, w - 1)   # Ensure x1 is within the image width
    x2_ = torch.clamp(x2, 0, w - 1)   # Ensure x2 is within the image width
    x1 = torch.minimum(x1_, x2_)
    x2 = torch.maximum(x1_, x2_)

    y1_ = torch.clamp(y1, 0, h - 1)   # Ensure y1 is within the image height
    y2_ = torch.clamp(y2, 0, h - 1)   # Ensure y2 is within the image height

    y1 = torch.minimum(y1_, y2_)
    y2 = torch.maximum(y1_, y2_)

    # Crop the images using the bounding box coordinates
    # cropped_images = torch.zeros_like(images)

    ci = []

    for i in range(batch_size):
        ci.append(resize(images[i, :, y1[i]:y2[i]+1, x1[i]:x2[i]+1].unsqueeze(0)))

    cropped_images = torch.cat(ci, dim = 0)
        

    return cropped_images

# Example usage:
# Assuming you have 'images' tensor with size [batch_size, channels, h, w]
# and 'boxes' tensor with size [batch_size, 4]

# cropped_images = crop_images_with_boxes(images, boxes)


class DR:
    def __init__(self):
        self.training = False
    
    def eval(self):
        pass

    def __call__(self, x):
        boxes_ = detect(x)

        faces = crop_images_with_boxes(x, boxes_)
        return model(resize(faces) * 2 - 1)


dr = DR()

# print('done')
# raise

import sys
sys.path.append('../adversarial-attacks-pytorch')
# sys.path.append('adversarial-attacks-pytorch')

import torchattacks
attack = torchattacks.Square_(dr, device='cuda', norm='Linf', eps=8/255, n_queries=5000, n_restarts=2, p_init=.8, seed=0, verbose=False, loss='cos', resc_schedule=True)
# >>> adv_images = attack(images, labels)

from tqdm import tqdm
results = []
for x1, _, y in tqdm(test_loader):
    boxes = detect(x1).data
    faces = crop_images_with_boxes(x1, boxes)
    feat = recognition(faces.cuda())

    # adv_images = attack(x1, boxes.clone())
    # boxes_ = detect(adv_images)

    adv_images = attack(x1, feat.clone())
    feat_ = dr(adv_images)

    # print(feat.shape, feat_.shape)

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