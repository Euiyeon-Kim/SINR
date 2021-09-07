from glob import glob

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0, 0.15), scale=(0.5, 1.25)),
    transforms.ToTensor()
])


class Custom(Dataset):
    def __init__(self, data_root, res, ext='jpg'):
        self.res = res
        self.paths = glob(f'{data_root}/*.{ext}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img = transform(img)
        # img = np.float32(img) / 255
        return img


if __name__ == '__main__':
    import os
    import torch
    from torchvision.utils import save_image

    os.makedirs('../test_dataloader', exist_ok=True)
    origin = Image.open('../inputs/mountains.jpg').convert('RGB')

    for i in range(10):
        img = transform(origin)
        print(img.shape, torch.min(img), torch.max(img))
        save_image(img, f'../test_dataloader/{i}.jpg')



