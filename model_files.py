import torch
import torch.nn as nn

def test():
    _batch_data = torch.ones((1, 3, 300, 300))
    model = VGG_SSD()
    _class, _reg = model(_batch_data)

class VGG_SSD(nn.Module):
    def __init__(self, num_classes = 7):
        '''
        VGG backbone for SSD
        The first 2 layers are belong to Backnone in feature detector
        And the other features are additional extractor to extract more large-scale objects
        '''
        super(VGG_SSD, self).__init__()
        
        self.num_classes = num_classes
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3,stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU()
            )
        ])

        self.regression_header = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ])
        self.classification_header = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        confidences, locations = [], []
        _x = None

        for index, (_FE, _reg, _classify) in enumerate(zip(self.feature_extractor, self.regression_header, self.classification_header)):
            _x = x if _x is None else _x
            _x = _FE(_x)

            _x = nn.functional.dropout2d(_x, 0.3, training = self.training)
            _location = _reg(_x)
            _location = _location.permute(0, 2, 3, 1).contiguous()
            _location = _location.view(_location.size(0), -1, 4)
            _confidence = _classify(_x)
            _confidence = _confidence.permute(0, 2, 3, 1).contiguous()
            _confidence = _confidence.view(_confidence.size(0), -1, self.num_classes)
            print(index)
            confidences.append(_confidence)
            locations.append(_location)
        
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        '''
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                        locations, self.priors, torch.FloatTensor(self.config.center_variance).cuda()[None,:], torch.FloatTensor(self.config.size_variance).cuda()[None,:]
                )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        
        else:
            return confidences, locations
        '''
        return confidences, locations

if __name__ == "__main__":
    test()