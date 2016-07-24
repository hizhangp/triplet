import os
from numpy import *

# Training image data path
IMAGEPATH = 'data/face/'

# Path to caffe directory
CAFFEPATH = '/home/peng/caffe'

# Snapshot iteration
SNAPSHOT_ITERS = 2000

# Max training iteration
MAX_ITERS = 50000

# The number of samples in each minibatch for triplet loss
TRIPLET_BATCH_SIZE = 150

# The number of samples in each minibatch for other loss
BATCH_SIZE = 60

# If need to train tripletloss, set False when pre-train
TRIPLET_LOSS = True

# Use horizontally-flipped images during training?
FLIPPED = True

# training percentage
PERCENT = 0.8

# USE semi-hard negative mining during training?
SEMI_HARD = False
