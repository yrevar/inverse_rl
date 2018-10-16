import glob
import numpy as np
from PIL import Image
# from skimage import io, transform

# Torch
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms

class ImageFolderDataset(Dataset):
    
    def __init__(self, pathname, img_loader=None, transform=None):
        
        self.pathname = pathname
        self.filenames = sorted(glob.glob(pathname))
        self.img_loader = img_loader if img_loader else Image.open
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        
        img = self.img_loader(self.filenames[i])
        if self.transform:
            img = self.transform(img)
        return img
    
def get_img_normalization_params(dataset):
    
    if isinstance(dataset, DataLoader):
        dataloader = dataset
    else:
        dataloader = DataLoader(dataset, batch_size=64, 
                                shuffle=False, drop_last=False)

    mean = 0.
    for i, imgs in enumerate(dataloader):
        
        if i == 0:
            if len(imgs.shape) == 3:
                mean = 0.
            elif len(imgs.shape) == 4:
                mean = torch.zeros(imgs.shape[3]).float()
            else:
                raise NotImplementedError
        
        mean = (i * mean + imgs.float().mean(0).mean(0).mean(0)) / (i+1)

    var = 0
    for i, imgs in enumerate(dataloader):
        
        if i == 0:
            if len(imgs.shape) == 3:
                var = 0.
            elif len(imgs.shape) == 4:
                var = torch.zeros(imgs.shape[3]).float()
            else:
                raise NotImplementedError
                
        var =  (i * var +  ((imgs.float() - mean)**2).mean(0).mean(0).mean(0)) / (i+1)
        
    return mean, np.sqrt(var)

def normalize_image(img, mu=0.2967633, std=0.19371898):
    return img.float().sub(mu).div(std)

def unnormalize_image(img, mu=0.2967633, std=0.19371898):
    return img.mul(std).add(mu).clamp(0,1).mul(255).byte()

def data_loader(pathname="./features/state_100x100_features/imgs_64x64/*.jpg", 
                batch_size=64, shuffle=True, num_workers=4, drop_last=True, 
                tr_va_te_split=[0.5, 0.3, 0.2], transform=transforms.Compose([
                    lambda x: np.asarray(x.convert("L"), np.uint8)/255.,
                    lambda x: torch.from_numpy(x),
                    normalize_image])):
    
    assert np.abs(sum(tr_va_te_split) - 1) < 1e-9
        
    dataset = ImageFolderDataset(pathname, img_loader=Image.open, transform=transform)
    
    n_tr = int(tr_va_te_split[0]*len(dataset))
    n_va = int(tr_va_te_split[1]*len(dataset))
    n_te = int(len(dataset) - n_tr - n_va)
    
    tr, va, te = torch.utils.data.random_split(dataset, [n_tr, n_va, n_te])
    params = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    
    return DataLoader(tr, **params), DataLoader(va, **params), DataLoader(te, **params)
