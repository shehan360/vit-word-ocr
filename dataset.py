import os
import math
import torch

from augmentation.warp import Curve, Distort, Stretch
from augmentation.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from augmentation.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from augmentation.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from augmentation.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from augmentation.camera import Contrast, Brightness, JpegCompression, Pixelate
from augmentation.weather import Fog, Snow, Frost, Rain, Shadow
from augmentation.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision.transforms as transforms


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index].split("_")[2])


def isless(prob=0.5):
    return np.random.uniform(0, 1) < prob


class DataAugment(object):
    '''
    Supports with and without data augmentation
    '''

    def __init__(self, opt):
        self.opt = opt

        if not opt.eval:
            self.process = [Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()]
            self.camera = [Contrast(), Brightness(), JpegCompression(), Pixelate()]

            self.pattern = [VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]

            self.noise = [GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()]
            self.blur = [GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()]
            self.weather = [Fog(), Snow(), Frost(), Rain(), Shadow()]

            self.noises = [self.blur, self.noise, self.weather]
            self.processes = [self.camera, self.process]

            self.warp = [Curve(), Distort(), Stretch()]
            self.geometry = [Rotate(), Perspective(), Shrink()]

            self.isbaseline_aug = False
            # rand augment
            if self.opt.isrand_aug:
                self.augs = [self.process, self.camera, self.noise, self.blur, self.weather, self.pattern, self.warp,
                             self.geometry]
            # semantic augment
            elif self.opt.issemantic_aug:
                self.geometry = [Rotate(), Perspective(), Shrink()]
                self.noise = [GaussianNoise()]
                self.blur = [MotionBlur()]
                self.augs = [self.noise, self.blur, self.geometry]
                self.isbaseline_aug = True
            # pp-ocr augment
            elif self.opt.islearning_aug:
                self.geometry = [Rotate(), Perspective()]
                self.noise = [GaussianNoise()]
                self.blur = [MotionBlur()]
                self.warp = [Distort()]
                self.augs = [self.warp, self.noise, self.blur, self.geometry]
                self.isbaseline_aug = True
            # scatter augment
            elif self.opt.isscatter_aug:
                self.geometry = [Shrink()]
                self.warp = [Distort()]
                self.augs = [self.warp, self.geometry]
                self.baseline_aug = True
            # rotation augment
            elif self.opt.isrotation_aug:
                self.geometry = [Rotate()]
                self.augs = [self.geometry]
                self.isbaseline_aug = True

        self.scale = False if opt.Transformer else True

    def __call__(self, img):
        '''
            Must call img.copy() if pattern, Rain or Shadow is used
        '''
        img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)

        if self.opt.eval or isless(self.opt.intact_prob):
            pass
        elif self.opt.isrand_aug or self.isbaseline_aug:
            img = self.rand_aug(img)
        # individual augment can also be selected
        elif self.opt.issel_aug:
            img = self.sel_aug(img)

        img = transforms.ToTensor()(img)
        if self.scale:
            img.sub_(0.5).div_(0.5)
        return img

    def rand_aug(self, img):
        augs = np.random.choice(self.augs, self.opt.augs_num, replace=False)
        for aug in augs:
            index = np.random.randint(0, len(aug))
            op = aug[index]
            mag = np.random.randint(0, 3) if self.opt.augs_mag is None else self.opt.augs_mag
            if type(op).__name__ == "Rain" or type(op).__name__ == "Grid":
                img = op(img.copy(), mag=mag)
            else:
                img = op(img, mag=mag)

        return img

    def sel_aug(self, img):

        prob = 1.

        if self.opt.process:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.process))
            op = self.process[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.noise:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.noise))
            op = self.noise[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.blur:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.blur))
            op = self.blur[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.weather:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.weather))
            op = self.weather[index]
            if type(op).__name__ == "Rain":  # or "Grid" in type(op).__name__ :
                img = op(img.copy(), mag=mag, prob=prob)
            else:
                img = op(img, mag=mag, prob=prob)

        if self.opt.camera:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.camera))
            op = self.camera[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.pattern:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.pattern))
            op = self.pattern[index]
            img = op(img.copy(), mag=mag, prob=prob)

        iscurve = False
        if self.opt.warp:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.warp))
            op = self.warp[index]
            if type(op).__name__ == "Curve":
                iscurve = True
            img = op(img, mag=mag, prob=prob)

        if self.opt.geometry:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.geometry))
            op = self.geometry[index]
            if type(op).__name__ == "Rotate":
                img = op(img, iscurve=iscurve, mag=mag, prob=prob)
            else:
                img = op(img, mag=mag, prob=prob)

        return img


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = DataAugment(self.opt)
            # i = 0
            # for image in images:
            #    transform(image)
            #    if i == 1:
            #        exit(0)
            #    else:
            #        i = i + 1
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        # else:
        #    transform = ResizeNormalize((self.imgW, self.imgH))
        #    image_tensors = [transform(image) for image in images]
        #    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
