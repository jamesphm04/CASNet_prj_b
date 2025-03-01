import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as trans
import random 
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop

## Import Data Loaders ##
from dataloader import *

from dataloader.RealCratersDataset import RealCratersDataset
from dataloader.SyntheticCratersDataset import SyntheticCratersDataset

def get_synthetic_craters_ds_info():
    image_dir = 'data/synthetic/ground_truth_images'
    images_ellipses_dir = 'data/synthetic/ground_truth_projected_ellipses'

    image_dir_items = [i.split(".")[0] for i in os.listdir(image_dir)]
    images_ellipses_dir_items = [i.split(".")[0] for i in os.listdir(images_ellipses_dir)]
    items = list(set(image_dir_items) & set(images_ellipses_dir_items))
    items = sorted(items, key=lambda x: int(x.split('.')[0]))

    img_dict = {i: os.path.join(image_dir, f'{item}.png') for i, item in enumerate(items)}

    # Get the list of image IDs
    img_keys = list(img_dict.keys())

    # Shuffle the image IDs
    random.shuffle(img_keys)

    # Define the percentage of the images that should be used for training
    train_pct = 0.8
    val_pct = 0.2

    # Calculate the index at which to split the subset of image paths into training and validation sets
    train_split = int(len(img_keys)*train_pct)
    val_split = int(len(img_keys)*(train_pct+val_pct))

    # Split the subset of image paths into training and validation sets
    train_keys = img_keys[:train_split]
    val_keys = img_keys[train_split:]
    
    return train_keys, val_keys, images_ellipses_dir, img_dict

def get_synthetic_craters_ds_transform():
    train_sz = 1024

    # Create a RandomIoUCrop object
    iou_crop = CustomRandomIoUCrop(min_scale=0.3, 
                                max_scale=1.0, 
                                min_aspect_ratio=0.5, 
                                max_aspect_ratio=2.0, 
                                sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                                trials=400, 
                                jitter_factor=0.25)

        
    # Compose transforms for data augmentation
    data_aug_tfms = transforms.Compose([
        iou_crop,
        transforms.ColorJitter(
                brightness = (0.875, 1.125),
                contrast = (0.5, 1.5),
                saturation = (0.5, 1.5),
                hue = (-0.05, 0.05),
        ),
        transforms.RandomGrayscale(),
        transforms.RandomEqualize(),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        ],
    )

    # Compose transforms to resize and pad input images
    resize_pad_tfm = transforms.Compose([
        transforms.Resize([train_sz] * 2, antialias=True)
    ])

    # Compose transforms to sanitize bounding boxes and normalize input data
    final_tfms = transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.SanitizeBoundingBoxes(),
    ])

    # Define the transformations for training and validation datasets
    train_tfms = transforms.Compose([
        data_aug_tfms, 
        resize_pad_tfm, 
        final_tfms
    ])
    valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms])
    
    return train_tfms, valid_tfms

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
    elif dataset == 'SC1':
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
        
    elif dataset == 'RC': #real craters
        train_dataset = RealCratersDataset(root='./data/real/ground_truth_images', train=True,
                                    transform=transforms.Compose([
                                        trans.Resize(imsize),
                                        trans.ToTensor(),
                                        trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
        test_dataset = RealCratersDataset(root='./data/real/ground_truth_images', train=False,
                                    transform=transforms.Compose([
                                        trans.Resize(imsize),
                                        trans.ToTensor(),
                                        trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
    elif dataset == 'SC': #synthetic craters
        
        train_keys, val_keys, images_ellipses_dir, img_dict = get_synthetic_craters_ds_info()
        # train_tfms, valid_tfms = get_synthetic_craters_ds_transform()
        
        train_tfms = transforms.Compose([
            trans.Resize(imsize),
            trans.ToTensor(),
            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        valid_tfms = transforms.Compose([
            trans.Resize(imsize),
            trans.ToTensor(),
            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
            
        class_names = ['background']+['crater']
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        
        train_dataset = SyntheticCratersDataset(train_keys, images_ellipses_dir, img_dict, class_to_idx, train_tfms)
        test_dataset =  SyntheticCratersDataset(val_keys, images_ellipses_dir, img_dict, class_to_idx, valid_tfms)
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch,
                                                   shuffle=True, num_workers=int(workers), pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch*4,
                                                   shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader
