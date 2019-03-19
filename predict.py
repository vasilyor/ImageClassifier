import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from torch.autograd import Variable
from collections import OrderedDict
import vutils

# arguments definition
parser = argparse.ArgumentParser(description='Required Options for predict.py')
parser.add_argument('input', action='store', type = str, help='Path to the input image', default='flowers/test/20/image_04910.jpg')
parser.add_argument('checkpoint', default='my_py_checkpoint.pth', action="store", type = str, help='Checkpoint path')
# parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='Top N classes to show')
parser.add_argument('--category_names', dest="category_names", action="store", type = str, default='cat_to_name.json', help='Path to Class names file')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available', default="gpu")

#args
args = parser.parse_args()
image = args.input
top_k = args.top_k
device = args.gpu
path = args.checkpoint
category_names = args.category_names


def main():

    # GPU Check
    device = vutils.if_gpu(is_gpu=args.gpu)
    model = vutils.load_checkpoint(path, device)
    print(model)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    top5_prob_arary, top5_classes = vutils.predict(image,  model, top_k, cat_to_name, device)
    print(top5_prob_arary, top5_classes)


if __name__== "__main__":
    main()
