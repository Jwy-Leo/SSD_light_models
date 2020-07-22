import numpy as np
import torch
import os
import re
from pycocotools.coco import COCO


def test():
    img_root_path = ""
    annotation_path = ""

class COCO_like_object_detection_dataset(torch.utils.data.Dataset):
    def __init__(self, img_root_path, annotation_path, img_transform=None, loc_transform=None):
        super(COCO_like_object_detection_dataset, self).__init__()
        self.img_root_path = img_root_path
        self.annotation_path = annotation_path
        self.img_transform = img_transform
        self.loc_transform = loc_transform

        
        cocotool = COCO(self.annotation_path)
        import pdb;pdb.set_trace()

    
    def __getitem__(self, index):
        pass

    def __len__(self):
        return 

if __name__ == "__init__":
    test()