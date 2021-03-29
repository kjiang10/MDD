import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import VisionDataset
from torchvision import transforms

import os
import os.path
from PIL import Image

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def is_image_file(filename):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(data_dir, class_to_idx, is_valid_file):
    instances = []
    directory = os.path.expanduser(data_dir)
    for c in sorted(class_to_idx.keys()):
        class_idx = class_to_idx[c]
        target_dir = os.path.join(directory, c)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_idx
                    instances.append(item)
    return instances

class DatasetFolder(VisionDataset):
    def __init__(self, root, loader=None, domain=0, n_sample=None, transform=None, target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, is_valid_file)

        self.loader = loader
        self.domain = domain
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.n_sample = n_sample if n_sample else len(samples) 

    def __getitem__(self, idx):
        idx = idx % len(self.samples)
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(sample)
        return sample, target, self.domain

    def __len__(self):
        return self.n_sample
    
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageFolder(DatasetFolder):
    def __init__(self, root, domain=0, n_sample=None, transform=None, target_transform=None, loader=pil_loader, is_valid_file=is_image_file):
        super(ImageFolder, self).__init__(root, 
                                        loader,
                                        domain=domain,
                                        n_sample=n_sample,
                                        transform=transform, 
                                        target_transform=target_transform, 
                                        is_valid_file=is_valid_file)
        self.imgs = self.samples

# class CycleConcatDataset(Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets
    
#     def __getitem__(self, idx):
#         items = []
#         for dataset in self.datasets:
#             idxi = idx % len(dataset)
#             items.append(dataset[idxi])
#         return tuple(items)
    
#     def __len__(self):
#         return max(len(d) for d in self.datasets)

class CatDataloader():
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
    
    def __iter__(self):
        self.loader_iter = []
        for dl in self.dataloaders:
            self.loader_iter.append(iter(dl))
        return self
    def __next__(self):
        items = []
        for data_iter in self.loader_iter:
            items.append(next(data_iter))
        return tuple(items)
    
    def __len__(self):
        return len(self.dataloaders[0])

def get_loader(s='mnist', t='mnist_m', batch_size=1000, train=True):
    source_dir = f'domain_adaptation_images/{s}/images/'
    target_dir = f'domain_adaptation_images/{t}/images/'
    transformA = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize(size=(28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])
    transformB = transforms.Compose([transforms.Resize(size=(28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])
    if train:
        source = ImageFolder(source_dir, domain=0, transform=transformA)
        target = ImageFolder(target_dir, domain=1,  transform=transformA)
        n_samples = max(len(target), len(source))
        source = ImageFolder(source_dir, domain=0, n_sample=n_samples, transform=transformA)
        target = ImageFolder(target_dir, domain=1, n_sample=n_samples, transform=transformA)

        sd = DataLoader(source, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        td = DataLoader(target, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        dl = CatDataloader([sd, td])
        return dl
    else:
        target = ImageFolder(target_dir, domain=1, transform=transformB)
        td = DataLoader(target, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        return td

if __name__ == "__main__":
    dl = get_loader()

    # for i, (img, label, domain) in enumerate(dl):
    #     print(i)
    print(len(dl))
    # for i, (img, label, domain) in enumerate(dl):
    for i, (s, t) in enumerate(dl):
        simg, sl, sd = s
        print(i)
        # if i == 1:
        #     break
        # print(simg.shape, sl, sd)

