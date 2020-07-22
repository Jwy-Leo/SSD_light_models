import numpy as np
import torch
import os
import re
from pycocotools.coco import COCO
import cv2


def test():
    img_root_path = "..\\..\\dataset\\NTHUSCDet\\COCO_like\\NTHUSCDet_square_light\\images\\train"
    annotation_path = "..\\..\\dataset\\NTHUSCDet\\COCO_like\\NTHUSCDet_square_light\\annotations\\instances_train.json"
    dataset = COCO_like_object_detection_dataset(img_root_path, annotation_path)
    import pdb;pdb.set_trace()
    imgs, anno = dataset[0]
    import pdb;pdb.set_trace()


class COCO_like_object_detection_dataset(torch.utils.data.Dataset):
    def __init__(self, img_root_path, annotation_path, img_transform=None, loc_transform=None):
        super(COCO_like_object_detection_dataset, self).__init__()
        self.img_root_path = img_root_path
        self.annotation_path = annotation_path
        self.img_transform = img_transform
        self.loc_transform = loc_transform
        
        cocotool = COCO(self.annotation_path)
        self.num_classes = len(cocotool.loadCats(cocotool.getCatIds())) + 1
        self._classes_name = ['Background']
        for sc, _id, name in cocotool.loadCats(cocotool.getCatIds()):
            self._classes_name.append(name)

        self.meta_files = cocotool.loadImgs(cocotool.getImgIds())
        import pdb;pdb.set_trace()
        self.anno_files = cocotool.loadAnns(cocotool.getAnnIds())

    
    def __getitem__(self, index):

        _meta_data = self.meta_files[index]
        img_path = os.path.join(self.img_root_path, _meta_data['file_name'])
        img = cv2.imread(img_path)

        gts = []
        _anno = self.anno_files[index]
        for _ann in _anno:
            import pdb;pdb.set_trace()
            boxes = _ann['bbox']
            category_id = _ann['category_id']
            boxes.append(category_id)
            gts.append(boxes, category_id)
        if self.img_transform is not None:
            img, gts = self.img_transform(img, gts)
        if self.loc_transform is not None:
            gts= self.loc_transform(gts)

        return img, gts

    def __len__(self):
        return len(self.meta_files)

if __name__ == "__main__":
    test()