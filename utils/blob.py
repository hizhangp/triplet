import numpy as np
import cv2


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def prep_im_for_blob(im):
    target_size = 224
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return im
