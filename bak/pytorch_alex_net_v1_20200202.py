import numpy as np
import os
import time

# Load and normalizing the CIFAR10 training and test datasets using torchvision
# Define a Convolutional Neural Network
# Define a loss function
# Train the network on the training data
# Test the network on the test data

# Load and normalizing the CIFAR10 training and test datasets using torchvision

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler

torch.__version__
datasets_root = "../datasets"
datasets_list = [
    'CIFAR10', 'MNIST'
]

# Define a Convolutional Neural Network

from torchvision.models.alexnet import AlexNet, alexnet, model_urls
from torch.hub import load_state_dict_from_url

models_root = "../models"


def alexnet(pretrained=False, model_root=None, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = torch.load(model_root)
        model.load_state_dict(state_dict)
    return model


import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# config
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
GAMMA = 0.1
lr = 0.001
MILESTONES = [20, 40]
EPOCHS = 60

import argparse

# Define loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:  # print every 2000 mini-batches
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    # validate
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            # _, pred = torch.max(outputs, 1)
            # correct += (pred == labels).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def run(args, model, device, train_loader, test_loader, scheduler, optimizer):
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        # scheduler
        scheduler.step()


def getcifar(args,scaled_size=235,tobecroped=True,croppedsize=227,Random_h_flip=True):
    args.test_batch_size
    args.batch_size
    transform = transforms.Compose([
        transforms.Resize(235),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=datasets_root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)

    transform = transforms.Compose([
        transforms.Resize(235),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root=datasets_root, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)
    return trainloader,testloader

TRAIN_PARAMETER = '''\
# TRAIN_PARAMETER
## loss
CrossEntropyLoss
## optimizer
SGD: base_lr %f momentum %f weight_decay %f
## lr_policy
MultiStepLR: milestones [%s] gamma %f epochs %d
''' % (
    lr,
    MOMENTUM,
    WEIGHT_DECAY,
    ','.join([str(v) for v in MILESTONES]),
    GAMMA,
    EPOCHS,
)


def main_bak():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    print("+++", args)

    transform = transforms.Compose([
        transforms.Resize(235),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=datasets_root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)

    transform = transforms.Compose([
        transforms.Resize(235),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root=datasets_root, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)
    net = alexnet(False, models_root, num_classes=10)
    # for param in net.parameters():
    #     param = nn.init.normal_(param)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    # Train the network on the training data
    # Test the network on the test data
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        # train
        net.train()
        for batch_idx, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # print("epoch:%d-batch:%d-"%(epoch,batch_idx),inputs.shape,type(inputs),labels.shape,type(labels))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print("--",inputs.shape,labels.shape)
            outputs = net(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_idx % args.log_interval == 0:  # print every 2000 mini-batches
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(inputs), len(trainloader.dataset),
                           100. * (batch_idx + 1) / len(trainloader), loss.item()))
                running_loss = 0.0

        # validate
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                # print("epoch:%d-batch:%d-"%(epoch,batch_idx),inputs.shape,type(inputs),labels.shape,type(labels))

                outputs = net(inputs)
                test_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                # _, pred = torch.max(outputs, 1)
                # correct += (pred == labels).sum().item()
        test_loss /= len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))

        # scheduler
        scheduler.step()

    PATH = './cifar10_alexnet_cifar10.pth'
    torch.save(net.state_dict(), PATH)
    print('Finished Training')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='load_pretrained')
    parser.add_argument('--model_path', type=str, default="", help="load model path.")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    print("+++", args)

    transform = transforms.Compose([
        transforms.Resize(235),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=datasets_root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)

    transform = transforms.Compose([
        transforms.Resize(235),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root=datasets_root, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)
    net = alexnet(args.pretrained, args.model_path, num_classes=10)
    # for param in net.parameters():
    #     param = nn.init.normal_(param)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    # Train the network on the training data
    # Test the network on the test data
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    run(args, net, device, trainloader, testloader, scheduler, optimizer)

    PATH = './cifar10_alexnet_cifar10.pth'
    torch.save(net.state_dict(), PATH)
    print('Finished Training')


if __name__ == '__main__':
    main()
