import os
import torch
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor

import matplotlib.pyplot as plt

def unnormalize(x, tmp_std, tmp_mean):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(tmp_std) + torch.Tensor(tmp_mean)
    x = x.transpose(1, 3)
    return x

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
            self.paths = f"{img_root}/canesm5_wet_omit_anom_hist_train.npy"
        else:
            self.paths = f"{img_root}/canesm5_wet_omit_anom_hist_valid.npy"

        self.maskpath = f"{mask_root}/canesm5_wet_omit_anom_hist_mask.npy"

    def __getitem__(self, index):
        npy_file = np.load(self.paths)
        gt_img = npy_file[index,:,:]
        gt_img = torch.from_numpy(gt_img[:,:])
        gt_img = gt_img.unsqueeze(0)
        a = gt_img[0,:,:]
        gt_img = a.repeat(3, 1, 1)

        mask_file = np.load(self.maskpath)
        mask = torch.from_numpy(mask_file[:,:]).float()
        mask = mask.unsqueeze(0)
        b = mask[0,:,:]
        mask = b.repeat(3, 1, 1)

        return gt_img * mask, mask, gt_img

    def __len__(self):
        npy_file = np.load(self.paths)
        leng = len(npy_file[:,1,1])
        return leng

def show(imgs, filename):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(filename)

def grid_image(model, dataset, device, filename):
    sample_size = 1 #max=10(valid_data)
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

if __name__ == '__main__':
    #initial setting
    root = '/docker/home/hasegawa/docker-gpu/reconstructionAI/canesm5_wet_experiments/canesm5_wet_omit_anom_hist/data'
    save_dir = '/docker/home/hasegawa/docker-gpu/reconstructionAI/canesm5_wet_experiments/canesm5_wet_omit_anom_hist'
    batch_size = 16
    n_threads = 24
    lr = 2e-4
    max_iter = 10**6
    vis_interval = 10**5
    size = (40, 128)
    LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}

    #data loading
    dataset_train = Places2(root, root, 'train')
    dataset_val = Places2(root, root, 'val')

    #model setting
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
    image, mask, gt = [x.to(device) for x in next(iterator_train)]
    image = image.float()
    mask = mask.float()
    gt = gt.float()

    #model training
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

    #save_image
        if (i+1) % vis_interval == 0:
            model.eval()
            npyname = f"{save_dir}/data/output{i+1}.npy"
            jpgname = f"{save_dir}/img/img{i+1}.jpg"
            img = grid_image(model, dataset_val, device, npyname)
            show(img, jpgname)
