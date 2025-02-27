import math
import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dmlpv2 import DMLP as DMLPv2
from trans4pass import tans_backbone

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    """

    def __init__(self, need_backbone=True):
        super(SegBaseModel, self).__init__()
        self.nclass = 10
        # self.aux = cfg.SOLVER.AUX
        # self.norm_layer = get_norm(cfg.MODEL.BN_TYPE)
        self.backbone = None
        self.encoder = None
        if need_backbone:
            self.get_backbone()

    def get_backbone(self):
        # self.backbone = cfg.MODEL.BACKBONE.lower()
        self.encoder = tans_backbone
        # get_segmentation_backbone(
            # self.backbone, self.norm_layer)

    def base_forward(self, x):
        """forwarding backbone network"""
        c1, c2, c3, c4 = self.encoder(x)
        return c1, c2, c3, c4

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred

    def evaluate(self, image):
        """evaluating network with inputs and targets"""
        scales = [1.0]
        flip = False
        crop_size = None
        batch, _, h, w = image.shape
        base_size = max(h, w)
        # scores = torch.zeros((batch, self.nclass, h, w)).to(image.device)
        scores = None
        for scale in scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)

            # resize image to current size
            cur_img = _resize_image(image, height, width)
            if crop_size is not None:
                assert crop_size[0] >= h and crop_size[1] >= w
                crop_size_scaled = (int(math.ceil(crop_size[0] * scale)),
                                    int(math.ceil(crop_size[1] * scale)))
                cur_img = _pad_image(cur_img, crop_size_scaled)
            outputs = self.forward(cur_img)[0][..., :height, :width]
            if flip:
                outputs += _flip_image(self.forward(_flip_image(cur_img))
                                       [0])[..., :height, :width]

            score = _resize_image(outputs, h, w)

            if scores is None:
                scores = score
            else:
                scores += score
        return scores


def _resize_image(img, h, w):
    return F.interpolate(img, size=[h, w], mode='bilinear', align_corners=True)


def _pad_image(img, crop_size):
    b, c, h, w = img.shape
    assert(c == 3)
    padh = crop_size[0] - h if h < crop_size[0] else 0
    padw = crop_size[1] - w if w < crop_size[1] else 0
    if padh == 0 and padw == 0:
        return img
    img_pad = F.pad(img, (0, padh, 0, padw))

    return img_pad


def _crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def _flip_image(img):
    assert(img.ndim == 4)
    return img.flip((3))


def _to_tuple(size):
    if isinstance(size, (list, tuple)):
        assert len(size), 'Expect eval crop size contains two element, ' \
                          'but received {}'.format(len(size))
        return tuple(size)
    elif isinstance(size, numbers.Number):
        return tuple((size, size))
    else:
        raise ValueError('Unsupport datatype: {}'.format(type(size)))




class Trans4PASS(SegBaseModel):

    def __init__(self):
        super().__init__()
        vit_params = {"embed_dim" : 256, "depth": 4, "num_heads": 8, "mlp_ratio": 3., "hid_dim": 64}
        c4_HxW = (1024 // 32) ** 2
        vit_params['decoder_feat_HxW'] = c4_HxW
        vit_params['nclass'] = 2
        vit_params['emb_chans'] = 64
        # if cfg.MODEL.DMLP == 'DMLPv1':
        #     print("Using DMLPv1")
        #     self.dede_head = DMLPv1(vit_params)
        # else:
        #     print("Using DMLPv2")
        self.dede_head = DMLPv2(vit_params)
        self.__setattr__('decoder', ['dede_head'])

    def forward(self, x):
        size = x.size()[2:]
        add_loss = {}
        feats, add_loss = self.encoder(x)
        c1, c2, c3, c4 = feats

        outputs = list()
        x = self.dede_head(c1, c2, c3, c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        # outputs.append(x)
        return x
        # return tuple(outputs), add_loss