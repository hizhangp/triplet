import caffe
import numpy as np
import yaml
from utils.timer import Timer
import config as cfg


class TripletLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the TripletLayer."""

        assert bottom[0].num == bottom[1].num, '{} != {}'.format(
            bottom[0].num, bottom[1].num)
        assert bottom[0].num == bottom[2].num, '{} != {}'.format(
            bottom[0].num, bottom[2].num)

        layer_params = yaml.load(self.param_str)
        self.margin = layer_params['margin']
        self._timer = Timer()

        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # self._timer.tic()
        anchor = np.array(bottom[0].data)
        positive = np.array(bottom[1].data)
        negative = np.array(bottom[2].data)
        aps = np.sum((anchor - positive) ** 2, axis=1)
        ans = np.sum((anchor - negative) ** 2, axis=1)
        # print 'ap' , aps
        # print 'an' , ans

        dist = self.margin + aps - ans
        dist_hinge = np.maximum(dist, 0.0)

        # add semi-hard mining
        # if cfg.SEMI_HARD:
        #     semihard = np.asarray(np.less(aps, ans), dtype=np.float)
        #     dist_hinge *= semihard

        self.residual_list = np.asarray(dist_hinge > 0.0, dtype=np.float)
        loss = np.sum(dist_hinge) / bottom[0].num

        top[0].data[...] = loss
        # print 'loss' , loss
        # self._timer.toc()
        # print 'Loss:', self._timer.average_time

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        if propagate_down[0]:
            anchor = np.array(bottom[0].data)
            positive = np.array(bottom[1].data)
            negative = np.array(bottom[2].data)

            coeff = 2.0 * top[0].diff / bottom[0].num
            bottom_a = coeff * \
                np.dot(np.diag(self.residual_list), (negative - positive))
            bottom_p = coeff * \
                np.dot(np.diag(self.residual_list), (positive - anchor))
            bottom_n = coeff * \
                np.dot(np.diag(self.residual_list), (anchor - negative))

            bottom[0].diff[...] = bottom_a
            bottom[1].diff[...] = bottom_p
            bottom[2].diff[...] = bottom_n

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
