import cv2
import torch
import numpy as np
from torchvision.datasets import VisionDataset
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2


class AbstractDataset(VisionDataset):
    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        super(AbstractDataset, self).__init__(cfg['root'], transforms=transforms,
                                              transform=transform, target_transform=target_transform)
        # fix for re-production
        np.random.seed(seed)

        self.images = list()
        self.targets = list()
        self.split = cfg['split']
        if self.transforms is None:
            self.transforms = Compose(
                [getattr(albumentations, _['name'])(**_['params']) for _ in cfg['transforms']] +
                [ToTensorV2()]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        tgt = self.targets[index]
        return path, tgt

    def load_item(self, items):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        images = list()
        for item in items:
            #print(item) tensor
            img = cv2.imread(item) #filename
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=img)['image']
            images.append(image)
        return torch.stack(images, dim=0)
