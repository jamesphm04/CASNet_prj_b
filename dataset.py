import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as trans

## Import Data Loaders ##
from dataloader import *


def get_dataset(dataset, batch, imsize, workers):

    if dataset == 'M':
        train_dataset = dset.MNIST(root='./data', train=True, download=True,
                                   transform=trans.Compose([
                                       trans.Resize(imsize),
                                       trans.Grayscale(3),
                                       trans.ToTensor(),
                                       trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        test_dataset = dset.MNIST(root='./data', train=False, download=True,
                                  transform=transforms.Compose([
                                      trans.Resize(imsize),
                                      trans.Grayscale(3),
                                      trans.ToTensor(),
                                      trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))

    elif dataset == 'MM':
        train_dataset = MNIST_M(root='./data/mnist_m', train=True,
                                transform=transforms.Compose([
                                    trans.Resize(imsize),
                                    trans.ToTensor(),
                                    trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        test_dataset = MNIST_M(root='./data/mnist_m', train=False,
                               transform=transforms.Compose([
                                   trans.Resize(imsize),
                                   trans.ToTensor(),
                                   trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif dataset == 'SC':
        train_dataset = Synth_Dataset(root='./data/Synthetic_Can_Data', train=True,
                                     transform=transforms.Compose([
                                         trans.Resize(imsize),
                                         trans.ToTensor(),
                                         trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
        test_dataset = Synth_Dataset(root='./data/Synthetic_Can_Data', train=False,
                                      transform=transforms.Compose([
                                          trans.Resize(imsize),
                                          trans.ToTensor(),
                                          trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))

    elif dataset == 'PC':
        train_dataset = Phys_Dataset(root='./data/physical_can', train=True,
                                     transform=transforms.Compose([
                                         trans.Resize(imsize),
                                         trans.ToTensor(),
                                         trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
        test_dataset = Phys_Dataset(root='./data/physical_can', train=False,
                                     transform=transforms.Compose([
                                         trans.Resize(imsize),
                                         trans.ToTensor(),
                                         trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch,
                                                   shuffle=True, num_workers=int(workers), pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch*4,
                                                   shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader
