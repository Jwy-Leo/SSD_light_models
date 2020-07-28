import torch

# Format Introduction
# Pascal Format : x1, y1, x2, y2
# COCO Format : x, y, w, h
# DarkNet Format : cx, cy, w, h


# location2center
def Pascal2COCO(bbox_pascal_format):
    
    type_error(bbox_pascal_format)
    shape_error(bbox_pascal_format)

    bbox_COCO_format = []
    bbox_COCO_format.append(bbox_pascal_format[0])
    bbox_COCO_format.append(bbox_pascal_format[1])
    bbox_COCO_format.append(bbox_pascal_format[2]-bbox_pascal_format[0]])
    bbox_COCO_format.append(bbox_pascal_format[3]-bbox_pascal_format[1]])
    
    bbox_COCO_format = torch.FloatTensor(bbox_COCO_format)
    
    return bbox_COCO_format

# center2location
def COCO2Pascal(bbox_COCO_format):

    type_error(bbox_COCO_format)
    shape_error(bbox_COCO_format)

    bbox_pascal_format = []
    bbox_pascal_format.append(bbox_COCO_format[0])
    bbox_pascal_format.append(bbox_COCO_format[1])
    bbox_pascal_format.append(bbox_COCO_format[0]+bbox_COCO_format[2]])
    bbox_pascal_format.append(bbox_COCO_format[1]+bbox_COCO_format[3]])

    bbox_pascal_format = torch.FloatTensor(bbox_pascal_format)

    return bbox_pascal_format

def COCO2DarkNet(bbox_COCO_format, input_config=None):

    type_error(bbox_COCO_format)
    shape_error(bbox_COCO_format)

    assert input_config !=None, "Doesn't given any input config, please recheck the configuration"

    bbox_darknet_format = []

    return bbox_darknet_format

# ScaleBack2Origin
# def ScaleBack2Origin():
#     pass

# Normalize2standard
# def Normalize2standard():
#     pass

def shape_error(tensor):
    assert tensor.shape[-1] == 4, 'Only support ... x 4 format, but not support {}'.format(tensor.shape[-1])

def type_error(tensor):
    assert isinstance(tensor, torch.Tensor), "Only support torch Tensor, not support {}".format(type(tensor))