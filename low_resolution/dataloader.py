import os, gc, sys
import json, PIL
import torch

import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _Loss
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import glob
from PIL import Image, ImageFile
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.datasets import ImageFolder as IFolder
from torch.utils.data import ConcatDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset, Subset
import torchvision.transforms as T


def preprocess_ffhq_fn(crop_center_size, output_resolution):
    if crop_center_size is not None:
        return T.Compose(
            [
                T.CenterCrop((crop_center_size, crop_center_size)),
                T.Resize((output_resolution, output_resolution), antialias=True),
                T.ToTensor(),  # Convert image to tensor
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((output_resolution, output_resolution), antialias=True),  # Resize the image
                T.ToTensor(),  # Convert image to tensor
            ]
        )


class FFHQ(Dataset):

    def __init__(
        self,
        root_path='/home/csxpeng/code/BiDO+_low/attack_datasets/FFHQ/thumbnails128x128',
        mode='test_ood',
        crop_center_size=800,
        preprocess_resolution=224,
        output_transform=None,
    ):
        self.preprocess_transform = preprocess_ffhq_fn(
            crop_center_size, preprocess_resolution
        )

        self.dataset = IFolder(root=root_path, transform=None)

        self.transform = output_transform

        self.targets = [0] * len(self.dataset)  # Adjust according to how labels are assigned

        print("FFHQ " + mode + ": Load " + str(len(self.dataset)) + " images")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        im = self.preprocess_transform(im)
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class FFHQ64(FFHQ):

    def __init__(
        self, root_path='/home/csxpeng/code/BiDO+_low/attack_datasets/FFHQ/thumbnails128x128', mode=None, output_transform=None
    ):

        super().__init__(root_path, mode, 88, 64, output_transform)


def preprocess_facescrub_fn(crop_center, output_resolution):
    if crop_center:
        crop_size = int(54 * output_resolution / 64)
        return T.Compose(
            [
                T.Resize((output_resolution, output_resolution), antialias=True),
                T.CenterCrop((crop_size, crop_size)),
                T.Resize((output_resolution, output_resolution), antialias=True),
                T.ToTensor(),  # Convert image to tensor

            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((output_resolution, output_resolution), antialias=True),  # Resize the image
                T.ToTensor(),  # Convert image to tensor
            ]
        )


class FaceScrub(Dataset):

    def __init__(
        self,
        root_path='/home/csxpeng/code/BiDO+_low/attack_datasets/FaceScrub',
        mode='test_ood',
        crop_center=False,
        preprocess_resolution=224,
        transform=None,
    ):

        root_actors = os.path.join(root_path, 'actors/faces')
        root_actresses = os.path.join(root_path, 'actresses/faces')
        dataset_actors = IFolder(root=root_actors, transform=None)
        target_transform_actresses = lambda x: x + len(dataset_actors.classes)
        dataset_actresses = IFolder(
            root=root_actresses,
            transform=None,
            target_transform=target_transform_actresses,
        )
        dataset_actresses.class_to_idx = {
            key: value + len(dataset_actors.classes)
            for key, value in dataset_actresses.class_to_idx.items()
        }
        self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
        self.classes = dataset_actors.classes + dataset_actresses.classes
        self.class_to_idx = {
            **dataset_actors.class_to_idx,
            **dataset_actresses.class_to_idx,
        }
        self.targets = dataset_actors.targets + [
            t + len(dataset_actors.classes) for t in dataset_actresses.targets
        ]
        self.name = 'facescrub_all'

        self.preprocess_transform = preprocess_facescrub_fn(
            crop_center, preprocess_resolution
        )

        self.transform = transform

        print("FaceScrub " + mode + ": Load " + str(len(self.dataset)) + " images")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        im = self.preprocess_transform(im)
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class FaceScrub64(FaceScrub):

    def __init__(
        self,
        root_path='/home/csxpeng/code/BiDO+_low/attack_datasets/FaceScrub',
        mode='test_ood',
        output_transform=None,
    ):
        super().__init__(root_path, mode, True, 64, output_transform)


class ImageFolder(data.Dataset):
    def __init__(self, args, file_path=None, mode=None):
        self.args = args
        self.model_name = args["model"]["architecture"]
        self.mode = mode
        self.processor = self.get_processor()

        if mode == "test_ood":
            file_path = args["dataset"]["ood_test_path"]
        if mode == "aux_ood":
            file_path = args["dataset"]["aux_ood_path"]

        self.name_list, self.targets = self.get_list(file_path)
        self.num_img = len(self.name_list)
        self.n_classes = args["model"]["num_classes"]
        print("CelebA " + mode + ": Load " + str(self.num_img) + " images")
        # self.image_list = self.load_img_list()
        # self.attr = self.load_attr()
    
    def get_list(self, file_path):
        name_list, targets = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "test_ood" or self.mode == "aux_ood":
                img_path = line.strip()
            else:
                img_path, iden = line.strip().split(' ')
                targets.append(int(iden))
            
            name_list.append(img_path)

        return name_list, targets

    def load_img_list(self):
        img_list = []
        data_root = self.args["dataset"]["img_path"]
        for i, img_name in enumerate(self.name_list):
            path = os.path.join(data_root, img_name)
            img = PIL.Image.open(path)
            if self.args['dataset']['name'] == 'celeba' or self.args['dataset']['name'] == "cifar":
                img = img.convert('RGB')
            else:
                img = img.convert('L')
            img_list.append(img)
        return img_list

    def load_attr(self):
        att_path = '../attack_datasets/CelebA/Anno/list_attr_celeba.txt'
        att_list = open(att_path).readlines()[2:] # start from 2nd row
        data_label = []
        for i in range(len(att_list)):
            data_label.append(att_list[i].split())

        attr = {}
        # transform label into 0 and 1
        for m in range(len(data_label)):
            k = data_label[m][0]
            data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
            data_label[m] = [int(p) for p in data_label[m]]
            attr[k] = data_label[m]
        
        return attr

    def load_img(self, path):
        data_root = self.args["dataset"]["img_path"]
        path = os.path.join(data_root, path)
        try:
            img = PIL.Image.open(path)
        except:
            base_name, ext = os.path.splitext(path)
            new_path = base_name + ".jpg"
            img = PIL.Image.open(new_path)        
        if self.args['dataset']['name'] == 'celeba' or self.args['dataset']['name'] == "cifar":
            img = img.convert('RGB')
        else:
            img = img.convert('L')
        return img

    def get_processor(self):
        if self.args['dataset']['name'] == "FaceNet":
            re_size = 112
        elif self.args['dataset']['name'] == "cifar":
            re_size = 32
        else:
            re_size = 64

        if self.args['dataset']['name'] == 'celeba':
            crop_size = 108

            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
            crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        elif self.args['dataset']['name'] == 'facescrub':
            crop_size = 64
            offset_height = (64 - crop_size) // 2
            offset_width = (64 - crop_size) // 2
            crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        elif self.args['dataset']['name'] == "cifar":
            proc = []
            if self.mode == "train":
                proc.append(transforms.ToTensor())
                proc.append(transforms.RandomCrop(32, padding=4)),
                proc.append(transforms.ToPILImage())
                proc.append(transforms.RandomHorizontalFlip(p=0.5))
                proc.append(transforms.ToTensor())

            else:
                proc.append(transforms.ToTensor())
                proc.append(transforms.ToPILImage())
                proc.append(transforms.ToTensor())

            return transforms.Compose(proc)

        proc = []
        if self.mode == "train":
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip(p=0.5))
            proc.append(transforms.ToTensor())
        else:

            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.ToTensor())

        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.load_img(self.name_list[index]))

        if self.mode == "test_ood" or self.mode == "aux_ood":
            return img, 1000
        
        label = self.targets[index]

        return img, label

    def __len__(self):
        return self.num_img


class GrayFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.model_name = args["model"]["architecture"]
        self.mode = mode
        self.processor = self.get_processor()
        self.name_list, self.targets = self.get_list(file_path)
        self.num_img = len(self.name_list)
        self.n_classes = args["model"]["num_classes"]
        print("Load " + str(self.num_img) + " images")
    
    def get_list(self, file_path):
        name_list, targets = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "test_ood" or self.mode == "aux":
                img_path = line.strip()
            else:
                img_path, iden = line.strip().split(' ')
                targets.append(int(iden))
            
            name_list.append(img_path)

        return name_list, targets

    def load_img(self, path):
        data_root = self.args["dataset"]["img_path"]
        path = os.path.join(data_root, path)
        img = PIL.Image.open(path)

        if self.args['dataset']['name'] == 'celeba' or self.args['dataset']['name'] == "cifar":
            img = img.convert('RGB')
        else:
            img = img.convert('L')
        return img

    def get_processor(self):
        proc = []
        if self.args['dataset']['name'] == "mnist":
            if self.model_name == "scnn" or self.model_name == "mcnn":
                re_size = 32
            elif self.model_name == "LeNet":
                re_size = 28
            proc.append(transforms.Grayscale(num_output_channels=3))  # 转换为三通道图像

        else:
            re_size = 64
        
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())

        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.load_img(self.name_list[index]))

        if self.mode == "test_ood" or self.mode == "aux":
            import torchvision
            torchvision.utils.save_image(img, "test_celeba_out.png")
            return img, 1000
        
        label = self.targets[index]

        # print(img.shape, type(img), img.max(), img.min()) #torch.Size([3, 64, 64]) <class 'torch.Tensor'> tensor(0.9020) tensor(0.)
        # img_path = os.path.join('./', self.name_list[index])
        # import torchvision
        # torchvision.utils.save_image(img, "test_celeba.png")

        return img, label

    def __len__(self):
        return self.num_img


class celeba(data.Dataset):
    def __init__(self, data_path=None, label_path=None):
        self.data_path = data_path
        self.label_path = label_path

        # Data transforms
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2

        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((112, 112)))
        proc.append(transforms.ToTensor())
        proc.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        self.transform = transforms.Compose(proc)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        image_tensor = self.transform(image_set)
        image_label = torch.Tensor(self.label_path[idx])
        return image_tensor, image_label


def load_attri():
    data_path = sorted(glob.glob('../attack_datasets/CelebA/Img/*.png'))
    print(len(data_path))
    # get label
    att_path = '../attack_datasets/CelebA/Anno/list_attr_celeba.txt'
    att_list = open(att_path).readlines()[2:] # start from 2nd row
    data_label = []
    for i in range(len(att_list)):
        data_label.append(att_list[i].split())

    attr = {}
    # transform label into 0 and 1
    for m in range(len(data_label)):
        k = data_label[m][0]
        data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
        data_label[m] = [int(p) for p in data_label[m]]
        attr[k] = data_label[m]
    
    dataset = celeba(data_path, data_label)
    # split data into train, valid, test set 7:2:1
    indices = list(range(202599))
    split_train = 141819
    split_valid = 182339
    train_idx, valid_idx, test_idx = indices[:split_train], indices[split_train:split_valid], indices[split_valid:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)

    validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler)

    testloader =  torch.utils.data.DataLoader(dataset, sampler=test_sampler)

    print(len(trainloader))
    print(len(validloader))
    print(len(testloader))

    return trainloader, validloader, testloader


if __name__ == '__main__':
    load_attri()
