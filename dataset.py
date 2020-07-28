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
    # import pdb;pdb.set_trace()
    imgs, anno, labels = dataset[0]
    for _anno in anno:
        _anno = _anno.astype(int)
        cv2.rectangle(imgs, (_anno[1],_anno[0]),( _anno[1]+ _anno[3], _anno[0]+_anno[2]), (0,0,255),1)
    cv2.imshow('aa', imgs)
    cv2.waitKey()
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
        img_ids = cocotool.getImgIds()
        self.meta_files = cocotool.loadImgs(img_ids)

        self.anno_files = []
        for img_index in img_ids:
            anno_ids = cocotool.getAnnIds(imgIds=img_index)
            temp_object_detection_infos = cocotool.loadAnns(anno_ids)
            import pdb;pdb.set_trace()
            ddd, ppp = self._Parsing_from_dict_to_np(temp_object_detection_infos)
            # import pdb;pdb.set_trace()
            self.anno_files.append((ddd, ppp))
    
    def __getitem__(self, index):

        _meta_data = self.meta_files[index]
        img_path = os.path.join(self.img_root_path, _meta_data['file_name'])
        img = cv2.imread(img_path)

        bboxes, labels = self.anno_files[index]
        # _anno = self.anno_files[index]
        # for _ann in _anno:
        #     import pdb;pdb.set_trace()
        #     boxes = _ann['bbox']
        #     category_id = _ann['category_id']
        #     boxes.append(category_id)
        #     gts.append(boxes, category_id)
        if self.img_transform is not None:
            img, gts = self.img_transform(img, gts)
        if self.loc_transform is not None:
            gts= self.loc_transform(gts)

        return img, bboxes, labels

    def _Parsing_from_dict_to_np(self, dict_files):
        '''
        Input: (dict_files, a list of dicts, using coco format)
        Output: bbox_format_anno
            (np.ndarray)
            4 x 1, cx, cy, w, h
        '''
        bbox_format_anno = []
        id_list = []
        for _df in dict_files:
            _box, _id = _df['bbox'], _df['id']
            bbox_format_anno.append(_box)
            id_list.append(_id)
        bbox_format_anno = np.array(bbox_format_anno)
        
        return bbox_format_anno, id_list

    def __len__(self):
        return len(self.meta_files)

if __name__ == "__main__":
    test()