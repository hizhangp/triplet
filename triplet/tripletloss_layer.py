import caffe
import numpy as np
import yaml
import config as cfg

class TripletLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the TripletLayer."""

        assert bottom[0].num == bottom[1].num, '{} != {}'.format(bottom[0].num, bottom[1].num)
        assert bottom[0].num == bottom[2].num, '{} != {}'.format(bottom[0].num, bottom[2].num)

        layer_params = yaml.load(self.param_str)
        self.margin = layer_params['margin']

        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        anchor = np.array(bottom[0].data)
        positive = np.array(bottom[1].data)
        negative = np.array(bottom[2].data)
        aps = np.sum((anchor - positive) ** 2, axis=1)
        ans = np.sum((anchor - negative) ** 2, axis=1)

        dist = self.margin + aps - ans
        dist_hinge = np.maximum(dist, 0.0)
        self.residual_list = np.asarray(dist_hinge > 0.0, dtype=np.float)
        loss = np.sum(dist_hinge) / bottom[0].num

        top[0].data[...] = loss


    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            anchor = np.array(bottom[0].data)
            positive = np.array(bottom[1].data)
            negative = np.array(bottom[2].data)

            # coeff = 2.0 / np.max((np.sum(self.residual_list), 1))
            coeff = 2.0 / bottom[0].num
            bottom_a = coeff * np.dot(np.diag(self.residual_list), (negative - positive))
            bottom_p = coeff * np.dot(np.diag(self.residual_list), (positive - anchor))
            bottom_n = coeff * np.dot(np.diag(self.residual_list), (anchor - negative))

            bottom[0].diff[...] = bottom_a
            bottom[1].diff[...] = bottom_p
            bottom[2].diff[...] = bottom_n

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
