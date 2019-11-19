from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


# pose_resnet related params
POSE_RESNET = CN()
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.PRETRAINED_LAYERS = ['*']

# pose_multi_resoluton_net related params
POSE_HIGH_RESOLUTION_NET = CN()
POSE_HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
POSE_HIGH_RESOLUTION_NET.STEM_INPLANES = 64
POSE_HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1

stage2 = {'CN()': CN(), 'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'NUM_BLOCKS': [4, 4],
          'NUM_CHANNELS': [32, 64], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'}
stage3 = {'CN()': CN(), 'NUM_MODULES': 1, 'NUM_BRANCHES': 3, 'NUM_BLOCKS': [4, 4, 4],
          'NUM_CHANNELS': [32, 64, 128], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'}
stage4 = {'CN()': CN(), 'NUM_MODULES': 1, 'NUM_BRANCHES': 4, 'NUM_BLOCKS': [4, 4, 4, 4],
          'NUM_CHANNELS': [32, 64, 128, 256], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'}


