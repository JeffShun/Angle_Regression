
import random
import torch
import torchvision.transforms.functional as TF

"""
数据预处理工具
1、所有数据预处理函数都包含两个输入: img 、label
2、img、label的输入维度为3维[C,H,W]，第一个维度是通道数
"""

class TransformCompose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class to_tensor(object):
    def __call__(self, img, label):
        img_o = torch.from_numpy(img)
        label_o = torch.from_numpy(label)
        return img_o, label_o


class resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        img_o, label_o = img, label
        img_o = torch.nn.functional.interpolate(img[None], size=self.size, mode="bilinear")[0]
        return img_o, label_o


class normlize(object):
    def __init__(self, win_clip=None):
        self.win_clip = win_clip

    def __call__(self, img, label): 
        img_o, label_o = img, label 
        if self.win_clip is not None:
            img = torch.clip(img, self.win_clip[0], self.win_clip[1])
        img_o = self._norm(img)
        return img_o, label_o
    
    def _norm(self, img):
        ori_shape = img.shape
        img_flatten = img.reshape(ori_shape[0], -1)
        img_min = img_flatten.min(dim=-1,keepdim=True)[0]
        img_max = img_flatten.max(dim=-1,keepdim=True)[0]
        img_norm = (img_flatten - img_min)/(img_max - img_min)
        img_norm = img_norm.reshape(ori_shape)
        return img_norm



class random_gamma_transform(object):
    """
    input must be normlized before gamma transform
    """
    def __init__(self, gamma_range=[0.8, 1.2], prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob
    def __call__(self, img, label):
        img_o, label_o = img, label
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_o = img**gamma
        return img_o, label_o
    

class random_add_noise(object):
    def __init__(self, sigma_range=[0.1, 0.3], prob=0.5):
        self.sigma_range = sigma_range
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img, label 
        if random.random() < self.prob:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
            noise = torch.randn_like(img) * sigma
            noisy_image = img + noise
            img_o = torch.clip(noisy_image, 0, 1)
        return img_o, label_o


class random_rotate(object):
    def __init__(self, theta_range=[-20, 20], prob=0.5):
        self.theta_range = theta_range
        self.prob = prob

    def __call__(self, img, label):
        img_o, label_o = img, label
        if random.random() < self.prob:
            theta = random.uniform(self.theta_range[0], self.theta_range[1])
            img_o = TF.rotate(img[None], theta)[0]
            label_o = (label_o/torch.pi*180 - theta)/180*torch.pi
        return img_o, label_o  





