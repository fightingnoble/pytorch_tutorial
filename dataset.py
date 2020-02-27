import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler

NUM_TRAIN = 49000
NUM_VAL = 1000


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def getcifar(args, mode, scaled_pad_size, RandomCrop, croppedsize, Random_h_flip=True, **kwargs):
    if mode == "rescale":
        pre1 = [transforms.Resize(scaled_pad_size)]
    elif mode == "pad":
        pre1 = [transforms.Pad(scaled_pad_size)]
    else:
        pre1 = []
    if RandomCrop:
        pre2 = [transforms.RandomCrop(croppedsize)]
    else:
        pre2 = []
    if Random_h_flip:
        pre3 = [transforms.RandomHorizontalFlip()]
    else:
        pre3 = []

    transform = transforms.Compose(pre1 + pre2 + pre3 +
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
                                              shuffle=False, sampler=ChunkSampler(NUM_TRAIN, 0), **kwargs)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    valset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=True,
                                          download=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False,
                                            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN), **kwargs)

    testset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, **kwargs)
    return trainloader, valloader, testloader
