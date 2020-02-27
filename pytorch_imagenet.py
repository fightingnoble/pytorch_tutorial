import os

import torch

import model
from dataset import getcifar, NUM_TRAIN

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from helper import accuracy, AverageMeter, save_checkpoint

import time
# Load and normalizing the CIFAR10 training and test datasets using torchvision
# Define a Convolutional Neural Network
# Define a loss function
# Train the network on the training data
# Test the network on the test data
# Load and normalizing the CIFAR10 training and test datasets using torchvision

# print(torch.__version__)
# datasets_root = "../datasets"
# datasets_list = [
#     'CIFAR10', 'MNIST'
# ]
best_prec1 = 0

# Define a Convolutional Neural Network

# from torchvision.models.alexnet import AlexNet, alexnet, model_urls


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
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * labels.size(0),
                                                                           NUM_TRAIN,
                                                                           100. * (batch_idx + 1) / len(train_loader),
                                                                           loss.item()), end="\r")
    print('\n')


def test(args, model, device, test_loader):
    if test_loader.dataset.train:
        print("test on validation set\r\n")
    else:
        print("test on test set\r\n")

    # validate
    model.eval()
    # test_loss = 0
    # correct = 0
    # num_samples = 0
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            test_loss = criterion(outputs, labels).item()
            # pred = outputs.argmax(dim=1, keepdim=True)
            # correct += pred.eq(labels.view_as(pred)).sum().item()
            # # _, pred = torch.max(outputs, 1)
            # # correct += (pred == labels).sum().item()
            # num_samples += pred.size(0)
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(test_loss, labels.size(0))
            top1.update(prec1[0], labels.size(0))
            top5.update(prec5[0], labels.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: Prec@1:{}/{} ({:.2f}%) Prec@5:{}/{} ({:.2f}%)\n'.format(
        losses.avg, top1.sum // 100, top1.count, top1.avg, top5.sum // 100, top1.count, top5.avg))
    return top1.avg, top5.avg


def run(args, model, device, train_loader, test_loader, scheduler, optimizer):
    global best_prec1
    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        train(args, model, device, train_loader, optimizer, epoch)
        prec1, prec5 = test(args, model, device, test_loader)
        # scheduler
        scheduler.step()
        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if args.detail and is_best:
            save_checkpoint({
                'epoch': args.epochs + 1,
                'arch': args.model_type,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, args.checkpoint_path + '_' + args.model_type + '_' + str(args.model_structure))


# import os
# os.path.join()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # model cfg
    # parser.add_argument('--pretrained', action='store_true', default=False,
    #                     help='load pretrained model')
    parser.add_argument('--model-type', type=str, default="", help="type of the model.")
    parser.add_argument('--model-structure', type=int, default=0, metavar='N',
                        help='model structure to be trained (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint, (default: None)')
    parser.add_argument('--e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--Quantized', action='store_true', default=False,
                        help='use quantized model')
    parser.add_argument('--qbit', default='4,8', help='activation/weight qbit')

# dataset
    parser.add_argument('--dataset-root', type=str, default="../datasets", help="load dataset path.")
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    # train cfg
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful to restarts)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    # optimizer
    parser.add_argument('--optim', type=str, default="Adam", help="optim type Adam/SGD")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    # scheduler
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--decreasing-lr', default='16,30,54', help='decreasing strategy')
    # device init cfg
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # result output cfg
    parser.add_argument('--detail', action='store_true', default=False,
                        help='show log in detial')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--checkpoint-path', type=str, default="", help="save model path.")
    args = parser.parse_args()
    print("+++", args)

    # Train the network on the training data
    # Test the network on the test data
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # net = alexnet(args.pretrained, args.resume, num_classes=10, structure=args.model_structure)
    # create model
    # config
    qbit_list = list(map(int, args.qbit.split(',')))
    quantize_cfg = {'input_qbit': qbit_list[1], 'weight_qbit': qbit_list[0], 'activation_qbit': qbit_list[1]}

    if args.model_type == 'VGG16':
        net = model.VGG16()
    elif args.model_type == 'cifar10' or args.model_type == 'VGG8':
        net = model.cifar10(n_channel=128,  quantized=args.Quantized, **quantize_cfg)
    elif args.model_type == 'alexnet':
        net = model.alexnet(num_classes=10, structure=args.model_structure)
    elif args.model_type == 'resnet18':
        net = model.resnet18()
    else:
        net = model.cifar10(n_channel=128,  quantized=args.Quantized, **quantize_cfg)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # for param in net.parameters():
    #     param = nn.init.normal_(param)

    # config
    milestones = list(map(int, args.decreasing_lr.split(',')))
    print(milestones)

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY) # not good enough 68%
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.optim == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    # optionlly resume from a checkpoint
    if args.resume:
        print("=> using pre-trained model '{}'".format(args.model_type))
    else:
        print("=> creating model '{}'".format(args.model_type))

    global best_prec1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    trainloader, valloader, testloader = getcifar(args, 'pad', 4, True, 32, True, **kwargs)
    print(len(trainloader), len(valloader), len(testloader))

    t_s = time.monotonic()

    if not args.evaluate:
        print("!!train!!")
        run(args, net, device, trainloader, valloader, scheduler, optimizer)
    print("!!test!!")
    test(args, net, device, testloader)
    t_e = time.monotonic()

    m, s = divmod(t_e-t_s, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

    PATH = args.checkpoint_path + '_' + args.model_type + '_' + str(args.model_structure) + '_final.pth'
    torch.save({
        'epoch': args.epochs + 1,
        'arch': args.model_type,
        'state_dict': net.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict()
    }, PATH)
    print('Finished Training')


if __name__ == '__main__':
    main()
