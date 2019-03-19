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
parser = argparse.ArgumentParser(description='Required Options for train.py')
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="my_py_checkpoint.pth")
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available', default="gpu")
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)


#args
args = parser.parse_args()
hidden_layer1 = args.hidden_units
epochs = args.epochs
path = args.save_dir
structure = args.arch
images_dir = args.data_dir
lr = args.learning_rate
device = args.gpu


def main():

    # GPU Check
    device = vutils.if_gpu(is_gpu=args.gpu)

    # loading and testing image data
    train_image_data, valid_image_data, test_image_data, train_loader , valid_loader, test_loader = vutils.data_loading(images_dir)
    print("Loaded {} images under {}".format(len(train_image_data), 'train'))
    print("Loaded {} images under {}".format(len(valid_image_data), 'valid'))
    print("Loaded {} images under {}".format(len(test_image_data), 'test'))

    model, optimizer, criterion = vutils.network_setup(structure,hidden_layer1,lr,device)
    # print(model)

    vutils.deep_learn_func(model, optimizer, criterion, epochs, 1, device, train_loader, valid_loader)

    vutils.save_the_checkpoint(model, train_image_data, path,structure,optimizer)


if __name__ == '__main__':
	main()
