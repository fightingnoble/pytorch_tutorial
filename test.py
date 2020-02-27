import os

import torch

import model
from dataset import getcifar, NUM_TRAIN

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from helper import accuracy, AverageMeter, save_checkpoint

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
            }, is_best, args.checkpoint_path + args.model_type + str(args.model_structure))


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

    parser.add_argument('--crxb-size', type=int, default=64, help='corssbar size')
    parser.add_argument('--vdd', type=float, default=3.3, help='supply voltage')
    parser.add_argument('--gwire', type=float, default=0.0357,
                        help='wire conductacne')
    parser.add_argument('--gload', type=float, default=0.25,
                        help='load conductance')
    parser.add_argument('--gmax', type=float, default=0.000333,
                        help='maximum cell conductance')
    parser.add_argument('--gmin', type=float, default=0.000000333,
                        help='minimum cell conductance')
    parser.add_argument('--ir-drop', action='store_true', default=False,
                        help='switch to turn on ir drop analysis')
    parser.add_argument('--scaler-dw', type=float, default=1,
                        help='scaler to compress the conductance')
    parser.add_argument('--test', action='store_true', default=False,
                        help='switch to turn inference mode')
    parser.add_argument('--enable_noise', action='store_true', default=False,
                        help='switch to turn on noise analysis')
    parser.add_argument('--enable_SAF', action='store_true', default=False,
                        help='switch to turn on SAF analysis')
    parser.add_argument('--enable_ec-SAF', action='store_true', default=False,
                        help='switch to turn on SAF error correction')
    parser.add_argument('--freq', type=float, default=10e6,
                        help='scaler to compress the conductance')
    parser.add_argument('--temp', type=float, default=300,
                        help='scaler to compress the conductance')

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
    crxb_cfg = {'crxb_size': args.crxb_size, 'gmax': args.gmax, 'gmin': args.gmin, 'gwire': args.gwire,
                'gload': args.gload, 'vdd': args.vdd, 'ir_drop': args.ir_drop, 'device': device, 'freq': args.freq,
                'temp': args.temp, 'enable_SAF': args.enable_SAF, 'enable_noise': args.enable_noise,
                'enable_ec_SAF': args.enable_ec_SAF, 'quantize': 64}

    if args.model_type == 'VGG16':
        net = model.VGG16()
    elif args.model_type == 'cifar10' or args.model_type == 'VGG8':
        net = model.cifar10(n_channel=128, physical=0, **crxb_cfg)
    elif args.model_type == 'alexnet':
        net = model.alexnet(num_classes=10, structure=args.model_structure)
    elif args.model_type == 'resnet18':
        net = model.resnet18()
    else:
        net = model.cifar10(n_channel=128, physical=True, **crxb_cfg)

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
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    # optionlly resume from a checkpoint
    if args.resume:
        print("=> using pre-trained model '{}'".format(args.model_type))
    else:
        print("=> creating model '{}'".format(args.model_type))

    global best_prec1
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         net.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    trainloader, valloader, testloader = getcifar(args, 'pad', 4, True, 32, True, **kwargs)
    print(len(trainloader), len(valloader), len(testloader))

    print('\r\n1!!!model_dict')
    model_dict = net.state_dict()
    print(model_dict.keys(),"\r\n2!!!model parameters")
    parm = {}
    for name, parameters in net.named_parameters():
        print(name)

    print('\r\n3!!!pretrained_dict')
    checkpoint = torch.load(args.resume)
    # print(type(checkpoint),'\r\n!!!')
    # print(checkpoint.keys(),'\r\n!!!')

    pretrained_dict = checkpoint['state_dict']
    print(pretrained_dict.keys(),'\r\n4!!!new_dict')

    import re
    new_dict={}
    for k, v in pretrained_dict.items():
        if k not in model_dict:
            bn_detect = re.match(r'module\.features\.(1|4|8|11|15|18|22)\.(running_mean|num_batches_tracked|running_var)', k)
            if bn_detect:
                k = 'module.features.{}.bn.{}'.format(bn_detect.group(1),bn_detect.group(2))
                print(k)
                new_dict[k]=v
            else:
                pass
        else:
            new_dict[k]=v
    print(new_dict.keys(),'\r\n5!!!')
    print([k for k, v in new_dict.items() if k not in model_dict],'\r\n')
    print([k for k, v in model_dict.items() if k not in new_dict])
    # print('net buffers')
    # print([n for n,v in net.named_buffers()], '\r\n !!!ideal_buffer')

    ideal_buffer = torch.load("../models/cifar10_crxb_ideal_VGG8_0_final.pth")
    # buffer_list = [k for k, v in ideal_buffer['state_dict'].items() if k not in new_dict]
    for k, v in ideal_buffer['state_dict'].items():
        if k not in new_dict:
            new_dict[k] = v
    print("\r\ncheck:", new_dict.keys() == model_dict.keys())
    model_dict.update(new_dict)
    # model_dict.update(ideal_buffer['state_dict'])
    net.load_state_dict(model_dict)
    # print('vvv')
    # print([k for k,v in ideal_buffer['state_dict'].items() if k not in model_dict])
    # net.load_state_dict(ideal_buffer['state_dict'])

    test(args, net, device, testloader)





# a = net.load_state_dict(checkpoint['state_dict'])
    # print(type(a))

if __name__ == '__main__':
    main()
import visdom
vis = visdom.Visdom(server='10.15.89.41', port=30330, use_incoming_socket=False)