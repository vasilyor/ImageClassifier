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

def if_gpu(is_gpu):
    # GPU Check
    use_cuda = torch.cuda.is_available()

    if is_gpu and use_cuda:
        device = torch.device("cuda:0")
        print(f"Device is set to {device}")
        return device
    else:
        device = torch.device("cpu")
        print(f"Device is set to {device}")
        return device


def data_loading(images_dir):

    data_dir = images_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

	# TODO: Define your transforms for the training, validation, and testing set
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_image_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_image_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_image_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_image_data, batch_size =32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_image_data, batch_size = 32, shuffle=True)

    return train_image_data, valid_image_data, test_image_data, train_loader , valid_loader, test_loader


def network_setup(structure='vgg16', hidden_layer1 = 4096,lr = 0.001, device=0):

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_filters=25088
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
        num_filters=9216
    else:
        model = models.vgg16(pretrained=True)
        num_filters=25088
        print("Please try for vgg16 only...for now")

    #for param in model.parameters():
    #    param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_filters, hidden_layer1)),
                          ('droput1', nn.Dropout(p=0.5)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 1024)),
                          ('relu2', nn.ReLU()),
                          ('droput2', nn.Dropout(p=0.5)),
                          ('fc3', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)
    #if torch.cuda.is_available() and device == 'gpu':
    #    model.cuda()
    return model, criterion, optimizer


def deep_learn_func(model, criterion, optimizer, epochs = 1, print_every=10, device=0, train_loader=0, valid_loader=0):
    steps = 0
    print('Starting training process....')
    print('\n')

    steps = 0
    running_loss = 0
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            #if torch.cuda.is_available() and device =='gpu':
            #    inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        #if torch.cuda.is_available() and device =='gpu':
                        #    inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(train_loader))
                valid_losses.append(valid_loss/len(valid_loader))
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

    print('\n\nTraining process done.')
    print('\n')


def save_the_checkpoint(model=0, train_image_data=0, path="my_py_checkpoint.pth",structure="vgg16",optimizer=None):
    # TODO: Save the checkpoint
    model.class_to_idx = train_image_data.class_to_idx

    checkpoint = {'classifier': model.classifier,
    'arch': structure,
    'state_dict': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, device=0):
    #state_dict = torch.load(file)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image='flowers/test/20/image_04910.jpg'):
    image = Image.open(image).convert("RGB")

    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = adjustments(image)

    return image



def predict(image='flowers/test/20/image_04910.jpg', model=0, top_k=5, category_names='cat_to_name.json', device=0):

    image = process_image(image)
    image = image.unsqueeze(0)
    #output = model.forward(Variable(image.to(device), volatile=True))
    with torch.no_grad():
        output = model.forward(Variable(image))

    top5_prob, top5_labels = torch.topk(output.cpu(), top_k)
    top5_prob = top5_prob.exp()
    top5_prob_arary = top5_prob.data.numpy()[0]
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    top5_labels_data = top5_labels.data.numpy()
    top5_labels_list = top5_labels_data[0].tolist()
    top5_classes = [inv_class_to_idx[x] for x in top5_labels_list]

    reversed_classes = top5_classes[::-1]
    reversed_probs = top5_prob_arary[::-1]

    c_names = [category_names[x] for x in reversed_classes]

    return reversed_probs, c_names
