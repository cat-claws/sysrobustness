import torch
import torchvision

lfw_pairs = torchvision.datasets.LFWPairs('', split = 'test', image_set = 'original', transform = torchvision.transforms.ToTensor(), download = True)
test_loader = torch.utils.data.DataLoader(lfw_pairs, batch_size = 128, num_workers=4)

x1, x2, y = next(iter(test_loader))

from retinaface import retinaface_resnet50
# detect = retinaface_resnet50('retinaface/weights/retinaface_resnet50_2020-07-20.pth')

import argparse
parser = argparse.ArgumentParser(description='Retinaface Testing')
parser.add_argument('--ckpt', default=None, help='checkpoint for testing')
args = parser.parse_args()

detect = retinaface_resnet50(args.ckpt)

import sys
sys.path.append('../adversarial-attacks-pytorch')
# sys.path.append('adversarial-attacks-pytorch')

import torchattacks
attack = torchattacks.Square_(detect, device='cuda', norm='Linf', eps=8/255, n_queries=5000, n_restarts=2, p_init=.8, seed=0, verbose=False, loss='iou', resc_schedule=True)
# >>> adv_images = attack(images, labels)

from tqdm import tqdm
with open(args.ckpt.replace('pth', 'txt'), 'w') as f:
    results = []
    for x1, _, y in tqdm(test_loader):
        boxes = detect(x1).data
        adv_images = attack(x1, boxes.clone())
        boxes_ = detect(adv_images)

        for b1, b2 in zip(boxes, boxes_):
            results.append(torchvision.ops.box_iou(b1, b2).item())
            f.write(str(sum(results)/len(results)))
            f.write('\n')
    # if len(results)>40:
    #     break

# adv_images = result.adv_images
# best = result.best.tolist()

    f.write('robustness based on IOU is: ', str(round(sum(results)/len(results), 4)))