"""
NeoNude model architecture, dataset, and utilities.

Based on a modified pix2pixHD GAN architecture.
"""

from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import functools
import os


# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    """Wraps a single OpenCV image into a PyTorch dataset."""

    def __init__(self):
        super().__init__()

    def initialize(self, opt, cv_img):
        self.opt = opt
        self.root = opt.dataroot
        self.A = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        self.dataset_size = 1

    def __getitem__(self, index):
        transform = get_transform(self.opt)
        a_tensor = transform(self.A.convert("RGB"))
        input_dict = {
            "label": a_tensor,
            "inst": 0,
            "image": 0,
            "feat": 0,
            "path": "",
        }
        return input_dict

    def __len__(self):
        return 1


class DataLoader:
    """Thin wrapper around torch.utils.data.DataLoader for a single image."""

    def __init__(self, opt, cv_img):
        super().__init__()
        self.dataset = Dataset()
        self.dataset.initialize(opt, cv_img)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.n_threads),
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return 1


# ---------------------------------------------------------------------------
# Generator Network
# ---------------------------------------------------------------------------

class ResnetBlock(torch.nn.Module):
    """A single residual block with reflection padding."""

    def __init__(self, dim, padding_type, norm_layer,
                 activation=torch.nn.ReLU(True), use_dropout=False):
        super().__init__()
        self.conv_block = self._build_conv_block(
            dim, padding_type, norm_layer, activation, use_dropout
        )

    def _build_conv_block(self, dim, padding_type, norm_layer,
                          activation, use_dropout):
        conv_block = []
        p = 0

        if padding_type == "reflect":
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"Padding [{padding_type}] is not implemented"
            )

        conv_block += [
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
            activation,
        ]

        if use_dropout:
            conv_block += [torch.nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"Padding [{padding_type}] is not implemented"
            )

        conv_block += [
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
        ]

        return torch.nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GlobalGenerator(torch.nn.Module):
    """Global generator network (U-Net style encoder-decoder)."""

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3,
                 n_blocks=9, norm_layer=torch.nn.BatchNorm2d,
                 padding_type="reflect"):
        assert n_blocks >= 0
        super().__init__()
        activation = torch.nn.ReLU(True)

        model = [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]

        # Encoder (downsample)
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                torch.nn.Conv2d(
                    ngf * mult, ngf * mult * 2,
                    kernel_size=3, stride=2, padding=1,
                ),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type,
                            activation=activation, norm_layer=norm_layer)
            ]

        # Decoder (upsample)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                torch.nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2),
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation,
            ]

        model += [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            torch.nn.Tanh(),
        ]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# High-level Model (load + inference)
# ---------------------------------------------------------------------------

class DeepModel:
    """Wraps a GlobalGenerator with weight loading and inference."""

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = []

        self.net_g = self._define_g(
            opt.input_nc, opt.output_nc, opt.ngf, opt.net_g,
            opt.n_downsample_global, opt.n_blocks_global,
            opt.n_local_enhancers, opt.n_blocks_local,
            opt.norm, self.gpu_ids,
        )
        self._load_network(self.net_g)

    def inference(self, label, inst):
        input_label, _, _, _ = self._encode_input(label, inst, infer=True)
        with torch.no_grad():
            return self.net_g.forward(input_label)

    # -- private helpers --

    def _load_network(self, network):
        save_path = os.path.join(self.opt.checkpoints_dir)
        network.load_state_dict(torch.load(save_path))

    def _encode_input(self, label_map, inst_map=None,
                      real_image=None, feat_map=None, infer=False):
        if len(self.gpu_ids) > 0:
            input_label = label_map.data.cuda()
        else:
            input_label = label_map.data
        return input_label, inst_map, real_image, feat_map

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _define_g(self, input_nc, output_nc, ngf, net_g,
                  n_downsample_global=3, n_blocks_global=9,
                  n_local_enhancers=1, n_blocks_local=3,
                  norm="instance", gpu_ids=[]):
        norm_layer = functools.partial(
            torch.nn.InstanceNorm2d, affine=False
        )
        net = GlobalGenerator(
            input_nc, output_nc, ngf,
            n_downsample_global, n_blocks_global, norm_layer,
        )
        if len(gpu_ids) > 0:
            net.cuda(gpu_ids[0])
        net.apply(self._weights_init)
        return net


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_transform(opt, method=Image.BICUBIC, normalize=True):
    """Build the image transform pipeline for the dataset."""
    transform_list = []
    base = float(2 ** opt.n_downsample_global)
    if opt.net_g == "local":
        base *= 2 ** opt.n_local_enhancers
    transform_list.append(
        transforms.Lambda(lambda img: _make_power_2(img, base, method))
    )
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return transforms.Compose(transform_list)


def _make_power_2(img, base, method=Image.BICUBIC):
    """Resize image so dimensions are multiples of base."""
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    """Convert a PyTorch tensor to a NumPy image array."""
    if isinstance(image_tensor, list):
        return [tensor2im(t, imtype, normalize) for t in image_tensor]

    image_numpy = image_tensor.cpu().float().numpy()

    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]

    return image_numpy.astype(imtype)
