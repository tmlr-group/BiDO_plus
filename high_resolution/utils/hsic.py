import torch
import numpy as np
from torch.autograd import Variable, grad


def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def coco_kernelmat(X, sigma, ktype='gaussian'):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)
        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))


    ## Adding linear kernel
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(H, torch.mm(Kx, H))

    return Kxc


def coco_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    K = coco_kernelmat(x, sigma=sigma)
    L = coco_kernelmat(y, sigma=sigma, ktype=ktype)

    res = torch.sqrt(torch.norm(torch.mm(K, L))) / m
    return res


def coco_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):
    coco_hx_val = coco_normalized_cca(hidden, h_data, sigma=sigma)
    coco_hy_val = coco_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)

    return coco_hx_val, coco_hy_val


def kernelmat(X, sigma, ktype='gaussian'):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        # 0.1，0.2，0.5，1.0，2.0，5.0
        # sigma = 1.5
        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))
       
        # print("Kx Min:", Kx.min().item())
        # print("Kx Max:", Kx.max().item())
        # print("Kx Mean:", Kx.mean().item()) # 在0.3 到 0.7 之间
        # print('-'* 50)
    
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1).type(torch.FloatTensor)

    Kxc = torch.mm(Kx, H)

    return Kxc


def hsic_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)
    # Pxy = torch.sum(torch.mul(Kxc, Kyc.t())) / m**2

    # print(Kxc)
    
    # return Pxy

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)

    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


def hsic_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):
    hsic_hx_val = hsic_normalized_cca(hidden, h_data, sigma=sigma)
    hsic_hy_val = hsic_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)

    return hsic_hx_val, hsic_hy_val


def imread(dataset, filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    if dataset == 'celeba' or dataset == 'cifar':
        return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]
    elif dataset == 'mnist' or dataset == 'cxr':
        tmp = np.asarray(Image.open(filename), dtype=np.uint8)[..., :1]
        return tmp


if __name__ == '__main__':
    from argparse import ArgumentParser
    import utils, os
    parser = ArgumentParser(description='train with BiDO')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cxr')
    parser.add_argument('--config_dir', default='./config', help='')
    args = parser.parse_args()

    file = os.path.join(args.config_dir, args.dataset + ".json")
    
    loaded_args = utils.load_json(json_file=file)
    # loaded_args['dataset']['sampler'] = True
    model_name = loaded_args['dataset']['model_name']
    train_file = loaded_args['dataset']['train_file']
    batch_size = loaded_args[model_name]['batch_size'] = 64
    trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")

    device = "cuda"
    # import pathlib
    # from PIL import Image

    # path_train = "../DMI/attack_res/celeba/trainset/"
    # path = pathlib.Path(path_train)
    # trainset = list(path.glob('*.png'))

    # path_test = "../DMI/attack_res/celeba/reg(87.40)/all/"
    # path_test = "../DMI/attack_res/celeba/HSIC-last/all/"
    # path_test = "../DMI/attack_res/celeba/HSIC-gaussian/all/"

    # for path_test in ["../DMI/attack_res/celeba/trainset/",
    #                   "../DMI/attack_res/celeba/reg(87.40)/all/",
    #                   "../DMI/attack_res/celeba/HSIC-last/all/",
    #                   "../DMI/attack_res/celeba/HSIC-gaussian/all/"]:
    #     path = pathlib.Path(path_test)
    #     testset = list(path.glob('*.png'))

    #     # exit()
    #     for i in range(0, len(trainset), batch_size):
    #         start = i
    #         end = i + batch_size

    #         # 1.
    #         train_images = np.array([imread('celeba', str(f)).astype(np.float32)
    #                         for f in trainset[start:end]])

    #         # Reshape to (n_images, 3, height, width)
    #         train_images = train_images.transpose((0, 3, 1, 2))
    #         train_images /= 255

    #         train_images = torch.from_numpy(train_images).type(torch.FloatTensor)
    #         train_images = train_images.cuda()

    #         # 2.
    #         test_images = np.array([imread('celeba', str(f)).astype(np.float32)
    #                         for f in testset[start:end]])

    #         # Reshape to (n_images, 3, height, width)
    #         test_images = test_images.transpose((0, 3, 1, 2))
    #         test_images /= 255

    #         test_images = torch.from_numpy(test_images).type(torch.FloatTensor)
    #         test_images = test_images.cuda()

    #         bs = test_images.size(0)
    #         train_images = train_images.reshape(bs, -1)
    #         test_images = test_images.reshape(bs, -1)

    #         hxz = hsic_normalized_cca(test_images, train_images, sigma=None)
    #         print(hxz)
            
    #         break

        
    for batch_idx, (inputs, iden) in enumerate(trainloader):
        bs = inputs.size(0)
        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        h_data = inputs.view(bs, -1)

        mu, std = 0, 10
        hidden = (mu + std*torch.randn(size=h_data.size())).to(device)

        hxz = hsic_normalized_cca(hidden, h_data, sigma=None)
        hxx = hsic_normalized_cca(h_data, h_data, sigma=None)
        hxzx = hsic_normalized_cca(h_data + hidden, h_data, sigma=None)
        print(hxz)
        print(hxx)
        print(hxzx)
        print('-'*80)
        # exit()
