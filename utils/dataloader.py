from glob import glob

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0, 0.15), scale=(0.5, 1.25)),
    transforms.ToTensor()
])


class Custom(Dataset):
    def __init__(self, data_root, ext='jpg', transform=None):
        self.paths = glob(f'{data_root}/*.{ext}')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        else:
            img = np.float32(img) / 255
        return img


class PatchINR(Dataset):
    def __init__(self, path, patch_size=131):
        self.img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
        self.h, self.w, _ = self.img.shape
        self.ph, self.pw = (self.h - patch_size + 1), (self.w - patch_size + 1)
        self.num_pixel = self.ph * self.pw
        self.patch_size = patch_size

    def __len__(self):
        return self.num_pixel

    def __getitem__(self, idx):
        c_h = (idx // self.pw) + (self.patch_size // 2)
        c_w = (idx % self.pw) + (self.patch_size // 2)
        img = self.img[c_h - (self.patch_size // 2):c_h + (self.patch_size // 2 + 1),
                       c_w - (self.patch_size // 2):c_w + (self.patch_size // 2 + 1), :]
        flat = img # .flatten()
        return np.array([(c_h - (self.patch_size // 2)) / self.ph, (c_w - (self.patch_size // 2)) / self.pw], dtype=np.float32), flat


class PatchINRVal(Dataset):
    def __init__(self, path, patch_size=131):
        self.img = np.array(Image.open(path).convert('RGB')) / 255.
        self.h, self.w, _ = self.img.shape
        self.ph, self.pw = (self.h - patch_size + 1), (self.w - patch_size + 1)
        self.num_pixel = self.ph * self.pw
        self.patch_size = patch_size

    def __len__(self):
        return self.num_pixel

    def __getitem__(self, idx):
        c_h = (idx // self.pw) + (self.patch_size // 2)
        c_w = (idx % self.pw) + (self.patch_size // 2)
        return np.array([(c_h - (self.patch_size // 2)) / self.ph, (c_w - (self.patch_size // 2)) / self.pw],
                        dtype=np.float32)


class PoC(Dataset):
    def __init__(self, path, patch_size=5):
        self.img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
        self.h, self.w, _ = self.img.shape
        self.ph, self.pw = (self.h - patch_size + 1), (self.w - patch_size + 1)
        self.num_pixel = self.ph * self.pw
        self.patch_size = patch_size

    def __len__(self):
        return self.num_pixel

    def __getitem__(self, idx):
        c_h = (idx // self.pw) + (self.patch_size // 2)
        c_w = (idx % self.pw) + (self.patch_size // 2)
        img = self.img[c_h - (self.patch_size // 2):c_h + (self.patch_size // 2 + 1),
                       c_w - (self.patch_size // 2):c_w + (self.patch_size // 2 + 1), :]
        flat = img.flatten()
        return np.array([(c_h - (self.patch_size // 2)) / self.ph,
                         (c_w - (self.patch_size // 2)) / self.pw],
                        dtype=np.float32), \
               flat


class LargeINR(Dataset):
    def __init__(self, path, b):
        self.img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
        self.h, self.w, _ = self.img.shape
        self.num_pixel = self.h * self.w
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, self.w), np.linspace(0, 1, self.h))
        self.grid = np.stack([grid_y, grid_x], axis=-1)
        self.mapped_grid = np.sin((2 * np.pi * self.grid) @ b.T).astype(np.float32)

    def __len__(self):
        return self.num_pixel

    def __getitem__(self, item):
        h_idx = item // self.w
        w_idx = item % self.w
        pixel = self.img[h_idx, w_idx, :]
        coord = self.mapped_grid[h_idx, w_idx, :]
        return coord, pixel


class FLOODINR(Dataset):
    def __init__(self, path):
        self.img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
        self.h, self.w, _ = self.img.shape
        self.num_pixel = self.h * self.w

    def __len__(self):
        return self.num_pixel

    def __getitem__(self, item):
        h_idx = item // self.w
        w_idx = item % self.w
        pixel = self.img[h_idx, w_idx, :]
        coord = np.array([h_idx/self.h, w_idx/self.w], dtype=np.float32)
        origin_coord = np.array([h_idx, w_idx], dtype=np.float32)
        return coord, pixel, origin_coord


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = LargeINR('../inputs/wild_bush.jpg')
    loader = DataLoader(dataset, shuffle=False, batch_size=128)

    from PIL import Image
    origin = Image.open('../inputs/wild_bush.jpg')
    sanity = np.zeros_like(np.array(origin), dtype=np.float32)

    for data in loader:
        coord, pixel, origin_coord = data
        for o, c in zip(origin_coord, pixel):
            o = o.numpy()
            c = c.numpy()
            sanity[o[0], o[1], :] = c

    sanity = np.array(sanity*255.).astype(np.uint8)
    recon = Image.fromarray(sanity)
    recon.save('tmp.jpg')

