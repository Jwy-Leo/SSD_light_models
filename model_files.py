import torch
import torch.nn as nn

# Create VGG_SSD

class VGG_SSD(nn.Modules):
    def __init__(self):
        _vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'C', 512,512,512, 'M', 512,512,512 
        ]
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
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU()
                nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU()
                nn.Conv2d(128, 256, kernel_size=3,stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU()
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU()
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
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
        ])
        self.classification_header = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
        ])
        pass

    def forward(self, x):
        confidences, locations = [], []
        _x = None

        for _FE, _reg, _classify in zip(self.feature_extractor, self.regression_header, self.classifcation_header):
            _x = X if _x is None else _x
            _x = _FE(_x)

            _x = nn.functional.dropout2d(_x, 0.3, training = self.training)
            _location = _reg(_x)
            _location = _location.permute(0, 2, 3, 1).contiguous()
            _location = _location.view(location.size(0), -1, 4)
            _confidence = _classify(_x)
            _confidence = _confidence.permute(0, 2, 3, 1).contiguous()
            _confidence = _confidence.view(_confidence.size(0), -1, self.num_classes)

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

