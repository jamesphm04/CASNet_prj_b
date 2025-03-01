import os 
import numpy as np 
import torch 
import torch.utils.data
import torchvision 
from PIL import Image
import cv2
from torchvision.tv_tensors import BoundingBoxes, Mask
from torch.utils.data import Dataset
import torchvision.transforms.v2  as transforms

class SyntheticCratersDataset(Dataset):
    def __init__(self, img_keys, anno_ellipses_dir, img_dict, class_to_idx, transforms=None):
        super(Dataset, self).__init__()
        
        self._img_keys = img_keys  # List of image keys
        self._anno_ellipses_dir = anno_ellipses_dir
        self._img_dict = img_dict  # Dictionary mapping image keys to image paths
        self._class_to_idx = class_to_idx  # Dictionary mapping class names to class indices
        self._transforms = transforms  # Image transforms to be applied
        
    def __len__(self):
        return len(self._img_keys)
        
    def __getitem__(self, index):
        # Retrieve the key for the image at the specified index
        img_path = self._img_dict[index]
        # # Get the annotations for this image
        annot_filename = img_path.split('/')[-1].split('.')[0] + '.txt'
        annot_file_path = os.path.join(self._anno_ellipses_dir, annot_filename)

        # Load the image and its target (segmentation masks, bounding boxes and labels)
        image, target = self._load_image_and_target(img_path, annot_file_path)        
        
        if self._transforms:
            image = self._transforms(image)
        return image, target
    
    @staticmethod
    def __get_mask( x_centre, y_centre, semi_major_axis, semi_minor_axis, rotation) -> np.ndarray:
        mask = np.zeros((1024, 1024), dtype=np.uint8)
     
        cv2.ellipse(mask, (int(x_centre), int(y_centre)),
                    (int(semi_major_axis), int(semi_minor_axis)),
                    angle=rotation, startAngle=0, endAngle=360,
                    color=1, thickness=-1)
        mask = Image.fromarray(mask, mode='L')
        return mask
    
    @staticmethod
    def __get_box(mask: np.ndarray):
        pos = np.nonzero(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def _load_image_and_target(self, img_path, annot_file_path):
        image = Image.open(img_path).convert('RGB')
        
        target = {
            "boxes": [],
            "masks": [],
            "labels": [],
        }
        with open(annot_file_path, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                label = 1 
                data = line.strip().split(',')
                # Extract ellipse parameters
                x_centre, y_centre, semi_major_axis, semi_minor_axis, rotation = map(float, data)
                rotation = np.degrees(rotation)
                mask = self.__get_mask(x_centre, y_centre, semi_major_axis, semi_minor_axis, rotation)
                
                box = self.__get_box(mask)
                target["masks"].append(mask)    
                target["boxes"].append(box)
                target["labels"].append(label)

        labels_tensor = torch.Tensor(target['labels']).to(dtype=torch.int64)
        masks = Mask(torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in target["masks"]]))
        bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=image.size[::-1])

        # return image, {'masks': masks,'boxes': bboxes, 'labels': labels_tensor}
        return image, labels_tensor[0]