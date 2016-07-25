import caffe
import numpy as np
import config as cfg
from collections import defaultdict

class TripletSampleLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the TripletSelectLayer."""
        top[0].reshape(*bottom[0].data.shape)
        top[1].reshape(*bottom[0].data.shape)
        top[2].reshape(*bottom[0].data.shape)


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        bottom_data = np.array(bottom[0].data)
        bottom_label = np.array(bottom[1].data)
        self.index_map = []

        top_anchor = bottom_data
        top_positive = []
        top_negative = []

        label_index_map = defaultdict(list)
        for i in xrange(bottom[0].num):
            label_index_map[bottom_label[i]].append(i)

        for i in xrange(bottom[0].num):
            anchor_label = bottom_label[i]

            positive_index = i
            while len(label_index_map[anchor_label]) > 1 and positive_index == i:
                positive_index = np.random.choice(label_index_map[anchor_label])
            positive = bottom_data[positive_index]

            negative_label = anchor_label
            while len(label_index_map) > 1 and negative_label == anchor_label:
                negative_label = np.random.choice(label_index_map.keys())
            negative_index = np.random.choice(label_index_map[negative_label])
            negative = bottom_data[negative_index]

            top_positive.append(positive)
            top_negative.append(negative)

            self.index_map.append([i, positive_index, negative_index])

        top[0].data[...] = np.array(top_anchor)
        top[1].data[...] = np.array(top_positive)
        top[2].data[...] = np.array(top_negative)


    def backward(self, top, propagate_down, bottom):

        bottom_diff = np.zeros(top[0].diff.shape)

        for i in xrange(top[0].num):
            bottom_diff[self.index_map[i][0]] += top[0].diff[i]
            bottom_diff[self.index_map[i][1]] += top[1].diff[i]
            bottom_diff[self.index_map[i][2]] += top[2].diff[i]

        bottom[0].diff[...] = bottom_diff


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
