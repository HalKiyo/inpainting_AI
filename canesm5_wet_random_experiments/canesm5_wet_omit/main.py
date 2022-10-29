import os
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from jmacmap import jmacmap


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, split='train'):
        super(Places2, self).__init__()

        if split == 'train':
            self.paths = f"{img_root}/canesm5_wet_omit_train.npy"
        else:
            self.paths = f"{img_root}/canesm5_wet_omit_valid.npy"

        self.maskpath = f"{mask_root}/canesm5_wet_omit_mask.npy"

    def __getitem__(self, index):
        npy_file = np.load(self.paths)
        gt_img = npy_file[index,:,:]
        gt_img = torch.from_numpy(gt_img[:,:])
        gt_img = gt_img.unsqueeze(0)
        a = gt_img[0,:,:]
        gt_img = a.repeat(3, 1, 1)

        mask_file = np.load(self.maskpath)
        N_mask = len(mask_file)
        mask = torch.from_numpy( mask_file[ random.randint(0, N_mask - 1), :, : ] )
        mask = mask.unsqueeze(0)
        b = mask[0,:,:]
        mask = b.repeat(3, 1, 1)

        return gt_img * mask, mask, gt_img

    def __len__(self):
        npy_file = np.load(self.paths)
        leng = len(npy_file[:,1,1])
        return leng

class Places3(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, split='val'):
        super(Places3, self).__init__()

        if split == 'val':
            self.paths = f"{img_root}/canesm5_wet_omit_valid.npy"
        else:
            self.paths = f"{img_root}/canesm5_wet_omit_valid.npy"

        self.maskpath = f"{mask_root}/canesm5_wet_omit_eval_mask.npy"

    def __getitem__(self, index):
        npy_file = np.load(self.paths)
        gt_img = npy_file[index,:,:]
        gt_img = torch.from_numpy(gt_img[:,:])
        gt_img = gt_img.unsqueeze(0)
        a = gt_img[0,:,:]
        gt_img = a.repeat(3, 1, 1)

        mask_file = np.load(self.maskpath)
        mask = torch.from_numpy( mask_file[ :, : ] )
        mask = mask.unsqueeze(0)
        b = mask[0,:,:]
        mask = b.repeat(3, 1, 1)

        return gt_img * mask, mask, gt_img

    def __len__(self):
        npy_file = np.load(self.paths)
        leng = len(npy_file[:,1,1])
        return leng

def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']

def show(imgs, imgs_label, filename):
    # figure layout
    nrows = 1
    ncols = len(imgs)
    pos_1 = nrows*100 + ncols*10 + 1
    pos = [i for i in range(pos_1, pos_1 + nrows*ncols)]

    # figure object
    fig = plt.figure()

    for i, num in enumerate(pos[:ncols]):
        ax = plt.subplot(num)

        tp = ax.imshow( imgs[i],
                        cmap=jmacmap(),
                        origin='lower',
                        vmin=0,
                        vmax=0.0004 )

        ax.set_title(imgs_label[i])
        fig.colorbar(tp, ax=ax, orientation='horizontal')

    plt.savefig(filename)


def grid_image(model, dataset, device, filename):
    sample_size = 1
    image, mask, gt = zip(*[dataset[i] for i in range(sample_size)])

    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    image = image.float()
    mask = mask.float()
    gt = gt.float()

    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))

    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = torch.cat((image,mask,output,output_comp,gt),dim=0)
    grid_detach = grid.to('cpu').detach().numpy()
    grid_channel = grid_detach[:,0,:,:]

    np.save(filename,grid_channel)

    return grid_channel

def valid_image(model, dataset, device, filename):
    sample_size = len(dataset)
    valid_output = np.empty([sample_size, 5, 40, 128])
    for i in range(sample_size):
        image, mask, gt = zip(*[dataset[i]])

        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)
        image = image.float()
        mask = mask.float()
        gt = gt.float()

        with torch.no_grad():
            output, _ = model(image.to(device), mask.to(device))

        output = output.to(torch.device('cpu'))
        output_comp = mask * image + (1 - mask) * output

        grid = torch.cat((image,mask,output,output_comp,gt),dim=0)
        grid_detach = grid.to('cpu').detach().numpy()

        valid_output[i,0,:,:] = grid_detach[0,0,:,:]
        valid_output[i,1,:,:] = grid_detach[1,0,:,:]
        valid_output[i,2,:,:] = grid_detach[2,0,:,:]
        valid_output[i,3,:,:] = grid_detach[3,0,:,:]
        valid_output[i,4,:,:] = grid_detach[4,0,:,:]

    np.save(filename,valid_output)


if __name__ == '__main__':
    # initial setting
    root = '/docker/home/hasegawa/docker-gpu/reconstructionAI'\
           '/canesm5_wet_random_experiments/canesm5_wet_omit/data'
    save_dir = '/docker/home/hasegawa/docker-gpu/reconstructionAI'\
               '/canesm5_wet_random_experiments/canesm5_wet_omit'
    batch_size = 16
    n_threads = 12
    lr = 2e-4
    max_iter = 50*10**5
    save_model_interval = 10**5
    vis_interval = 10**5
    resume = False
    load_model_name = 7*10**5
    size = (40, 128)
    LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
    IMAGES_LABEL = ['input', 'mask', 'output', 'output_comp', 'gt']

    # data loading
    dataset_train = Places2(root, root, 'train')
    dataset_val = Places3(root, root, 'val')

    # model setting
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    iterator_train = iter(data.DataLoader(dataset_train, batch_size=batch_size,
                          sampler=InfiniteSampler(len(dataset_train)),
                          num_workers=n_threads))
    model = PConvUNet().to(device)
    start_iter = 0
    optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

    # load check point (option)
    if resume is True:
        start_iter = load_ckpt(
            f'{save_dir}/ckpt/{load_model_name}.pth',
            [('model', model)],
            [('optimizer', optimizer)] )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)

    # locate data to gpu
    image, mask, gt = [x.to(device) for x in next(iterator_train)]
    image = image.float()
    mask = mask.float()
    gt = gt.float()

    # model training
    for i in tqdm(range(start_iter, max_iter)):
        model.train()
        output, _ = model(image, mask)
        loss_dict = criterion(image, mask, output, gt)

        loss = 1.0
        for key, coef in LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save check point
        if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
            save_ckpt( f'{save_dir}/ckpt/{i+1}.pth',
                       [('model', model)],
                       [('optimizer', optimizer)],
                       i + 1)

    # save output
        if (i + 1) % vis_interval == 0:
            model.eval()
            npyname = f"{save_dir}/data/output/output{i+1}.npy"
            jpgname = f"{save_dir}/img/img{i+1}.jpg"
            validname = f"{save_dir}/valid/valid{i+1}.npy"

            img = grid_image(model, dataset_val, device, npyname)
            show(img, IMAGES_LABEL, jpgname)
            valid_image(model, dataset_val, device, validname)
