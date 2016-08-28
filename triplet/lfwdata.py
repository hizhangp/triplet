import os
import config as cfg
import numpy as np


class lfwdata():

    def __init__(self):
        self._pairs = []

        pairs = open(os.path.join(cfg.LFW_IMAGEPATH, '../pairs.txt'))
        pairs.readline()
        for pair in pairs:
            pair = pair.split()
            if len(pair) == 3:
                img1 = os.path.join(
                    pair[0], pair[0] + '_{:04d}.jpg'.format(int(pair[1])))
                img2 = os.path.join(
                    pair[0], pair[0] + '_{:04d}.jpg'.format(int(pair[2])))
                label = True
            elif len(pair) == 4:
                img1 = os.path.join(
                    pair[0], pair[0] + '_{:04d}.jpg'.format(int(pair[1])))
                img2 = os.path.join(
                    pair[2], pair[2] + '_{:04d}.jpg'.format(int(pair[3])))
                label = False
            else:
                assert False, pair
            self._pairs.append({'img': [img1, img2], 'label': label})

        print 'Number of pairs: {}'.format(len(self._pairs))

if __name__ == '__main__':

    pairs = lfwdata()
