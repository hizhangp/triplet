import caffe
import numpy as np
from sklearn.preprocessing import normalize


class L2NormLayer(caffe.Layer):
    """L2norm layer used for L2 normalization."""

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""

        top[0].reshape(bottom[0].num, bottom[0].data.shape[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        top[0].reshape(*bottom[0].data.shape)
        top[0].data[...] = normalize(bottom[0].data)

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        if propagate_down[0]:
            norm = np.linalg.norm(bottom[0].data, axis=1)[:, np.newaxis]
            td = top[0].diff
            bd = bottom[0].data
            bottom_n2 = td * norm - (bd * td).sum(axis=1)[:, np.newaxis] * bd / norm
            bottom_n2 /= norm ** 2

            bottom[0].diff[...] = bottom_n2

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
