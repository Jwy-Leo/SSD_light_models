import os
import re
import numpy as np
import argparse
import yaml
from dataset import COCO_like_object_detection_dataset
from model_files import VGG_SSD
import torch


def Arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_file", type=str, default="config/light_config.yaml", help="config file by yaml")
    parser.add_argument('--Epoch', type=int, default=10, help='Training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='model learning rate')

    args = parser.parse_args()

    with open(args.config_file, 'r') as F:
        args.config = yaml.load(F)
    
    args.config['dataset']['dataset_root_path'] = os.path.join(*re.split(r'\\||/',args.config['dataset']['dataset_root_path']))
    print(args)

    return args

def main(args):
    
    devices = 'cpu' if not torch.cuda.is_available() else 'cuda'

    num_classes, train_dataset, test_dataset = dataset_loader(args)
    model = model_loader(num_classes)
    optimizer, criteria = optim_and_criteria(args, model)

    model = model.to(devices)

    train_dataloader = DataLoader(train_dataset, shuffle=True, drop_last=True, num_workers=args.config['dataloader_config']['num_workers'])
    test_dataloader = DataLoader(test_dataset, shuffle=False, drop_last=False, num_workers=args.config['dataloader_config']['num_workers'])

    for epoch in range(args.Epochs):
        training_loss = Training(args, model, optim, criteria. train_dataloader)
        test_loss = Testing(args, model, criteria, test_dataloader)
        print('loss:{}'.format(loss.items()))
    pass

def dataset_loader(args):
    '''
    Input:
        args: load the dataset configuration
    Output:
        num_classes:
        train_dataset:
        test_dataset:
    '''
    imgs_train = os.path.join(args.config['dataset']['dataset_root_path'], 'images', 'train')
    anno_train = os.path.join(args.config['dataset']['dataset_root_path'], 'annotations', "instances_train.json")

    train_dataset = COCO_like_object_detection_dataset(imgs_train, anno_train)
    tr_num_classes = train_dataset.num_classes

    imgs_test = os.path.join(args.config['dataset']['dataset_root_path'], 'images', 'test')
    anno_test = os.path.join(args.config['dataset']['dataset_root_path'], 'annotations', "instances_test.json")
    
    test_dataset = COCO_like_object_detection_dataset(imgs_test, anno_test)
    te_num_classes = test_dataset.num_classes
    assert tr_num_classes == te_num_classes, \
            "training and testing task deons'y be aligned, training classes/testing classes :{}/{}".format(tr_num_classes, te_num_classes)

    return train_dataset, test_dataset, te_num_classes

def model_loader(num_classes):
    model = VGG_SSD(num_classes)
    return model

def optim_and_criteria(args, model):
    '''
    Input:
        args : argparser and yaml.config
    output:
        optimizer: Adam
        criteria:  Multi-object matching loss, with the classification and regression loss
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criteria = None

    return optimizer, criteria

def Training(args, model, optim, criteria, dataloader):

    accumulate_loss = []
    model.train()
    for step, (imgs, gts_tuple) in enumerate(dataloader):
        classify_results, regression_results = model(imgs)
        loss = criteria(classify_results, regression_results, gts_tuple)
        optim.zero_grad()
        loss.backward()
        optim.step()
        accumulate_loss.append(loss.items())

    return float(np.array(accumulate_loss).mean())

def Testing(args, model, criteria, dataloader):

    accumulate_loss = []
    model.eval()
    with torch.no_grad():
        for step, (imgs, gts_tuple) in enumerate(dataloader):
            classify_results, regression_results = model(imgs)
            loss = criteria(classify_results, regression_results, gts_tuple)
            accumulate_loss.append(loss.items())

    return float(np.array(accumulate_loss).mean())

if __name__ == "__main__":
    args = Arguments()
    main(args)