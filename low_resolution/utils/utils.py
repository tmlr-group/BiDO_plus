import numpy as np
import torch, random, sys, json, time, dataloader, copy, os
import torch.nn as nn
import torchvision.utils as tvls

from datetime import datetime
from torch.utils.data import sampler
from collections import defaultdict
from torchvision import transforms

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()


def save_tensor_images(images, filename, nrow=None, normalize=True):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)


def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_pretrain(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name.startswith("module.fc_layer"):
            continue
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_params(self, model):
    own_state = self.state_dict()
    for name, param in model.named_parameters():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    if dataset is not None:
        print("Dataset: %s" % dataset)
        print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        if i >= 3:
            print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')


def init_dataloader(args, file_path=None, mode="gan", iterator=False):
    tf = time.time()
    bs = args["training"]['batch_size']

    if mode != "aux_ood" and mode != "test_ood":
        if args['dataset']['name'] == "celeba" \
                or args['dataset']['name'] == "cifar":
            data_set = dataloader.ImageFolder(args, file_path, mode)

        elif args['dataset']['name'] == "mnist" or args['dataset']['name'] == "cxr":
            data_set = dataloader.GrayFolder(args, file_path, mode)

        elif args['dataset']['name'] == "facescrub":
            data_set = dataloader.FaceScrub64(mode=mode)
        
        elif args['dataset']['name'] == "ffhq":
            data_set = dataloader.FFHQ64(mode=mode)

    if args['outlier_exp']['enable_OE'] and mode == "aux_ood" or mode == "test_ood":
        if 'facescrub' in args['outlier_exp']['aux_ood_dataset']:
            data_set = dataloader.FaceScrub64(mode=mode)
        elif 'ffhq' in args['outlier_exp']['aux_ood_dataset']:
            data_set = dataloader.FFHQ64(mode=mode)
        elif 'celeba' in args['outlier_exp']['aux_ood_dataset']:
            data_set = dataloader.ImageFolder(args, mode=mode)
            
    if mode == "train":
        if args['dataset']['name'] == "celeba":
            if args['dataset']['sampler']:
                sampler = RandomIdentitySampler(data_set, bs, args['dataset']['instance'])
                data_loader = torch.utils.data.DataLoader(data_set,
                                                          sampler=sampler,
                                                          batch_size=bs,
                                                          num_workers=args['training']['num_workers'],
                                                          pin_memory=True,
                                                          drop_last=True
                                                          )
            else:
                data_loader = torch.utils.data.DataLoader(data_set,
                                                          batch_size=bs,
                                                          shuffle=True,
                                                          num_workers=args['training']['num_workers'],
                                                          pin_memory=True,
                                                          drop_last=True
                                                          )
        else:
            data_loader = torch.utils.data.DataLoader(data_set,
                                                      shuffle=True,
                                                      batch_size=bs,
                                                      num_workers=args['training']['num_workers'],
                                                      pin_memory=True,
                                                      drop_last=False
                                                      )
    else:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  shuffle=False,
                                                  batch_size=bs,
                                                  num_workers=args['training']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=False
                                                  )
        if iterator:
            data_loader = torch.utils.data.DataLoader(data_set,
                                                      batch_size=bs,
                                                      shuffle=True,
                                                      num_workers=args['training']['num_workers'],
                                                      pin_memory=True,
                                                      drop_last=False
                                                      ).__iter__()

    return data_set, data_loader


class RandomIdentitySampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.data_source = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)

            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def get_deprocessor():
    # resize 112,112
    proc = []
    proc.append(transforms.Resize((112, 112)))
    proc.append(transforms.ToTensor())
    return transforms.Compose(proc)


def low2high(img):
    # 0 and 1, 64 to 112
    bs = img.size(0)
    proc = get_deprocessor()
    img_tensor = img.detach().cpu().float()
    img = torch.zeros(bs, 3, 112, 112)
    for i in range(bs):
        img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
        img_i = proc(img_i)
        img[i, :, :, :] = img_i[:, :, :]

    img = img.cuda()
    return img


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes)[y.cpu()], dim=1)


def save_checkpoint(state, directory, filename):
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)


import torch.nn.functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
