from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image


class SaltDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        super(SaltDataset, self).__init__()
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        msk = Image.open(self.masks[index])

        if self.transform:
            transformed = self.transform(image=np.array(img), mask=np.array(msk))
            img = transformed['image']
            msk = transformed['mask']
        else:
            img = torch.tensor(img, dtype=torch.float32)
            msk = torch.tensor(msk, dtype=torch.float32)

        if msk.max():
            msk = msk / msk.max()

        img = img.float() / 255.0

        return img.float(), msk.float()

