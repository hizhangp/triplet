import caffe
import numpy as np
import cv2
import copy
import config as cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


class DataLayer(caffe.Layer):
    """Sample data layer used for training."""

    def _shuffle_data(self):
        sample = []
        sample_person = copy.deepcopy(self._data._sample_person)
        for i in sample_person.keys():
            np.random.shuffle(sample_person[i])
        while len(sample_person) > 0:
            person = np.random.choice(sample_person.keys())
            if len(sample_person[person]) < cfg.CUT_SIZE:
                sample_person.pop(person)
                continue
            num = 0
            while num < cfg.CUT_SIZE and len(sample_person[person]) > 0:
                sample.append(sample_person[person].pop())
                num += 1
        self._data._sample = sample


    def _get_next_minibatch(self):
        # Sample to use for each image in this batch
        # Sample the samples randomly

        if self._index + self.batch_size <= len(self._data._sample):
            sample = self._data._sample[self._index:self._index + self.batch_size]
            self._index += self.batch_size
        else:
            sample = self._data._sample[self._index:]
            if cfg.TRIPLET_LOSS:
                self._shuffle_data()
            else:
                np.random.shuffle(self._data._sample)
            self._epoch += 1
            print 'Epoch {}'.format(self._epoch)
            self._index = self.batch_size - len(sample)
            sample.extend(self._data._sample[:self._index])

        im_blob,labels_blob = self._get_image_blob(sample)
        blobs = {'data': im_blob,
             'labels': labels_blob}
        return blobs

    def _get_image_blob(self,sample):
        im_blob = []
        labels_blob = []
        for i in range(self.batch_size):
            im = cv2.imread(cfg.IMAGEPATH + sample[i]['picname'])
            if sample[i]['flipped']:
                im = im[:, ::-1, :]
            personname = sample[i]['picname'].split('/')[0]
            labels_blob.append(self._data._sample_label[personname])
            im = prep_im_for_blob(im)

            im_blob.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(im_blob)
        return blob,labels_blob

    def set_data(self, data):
        """Set the data to be used by this layer during training."""
        self._data = data
        if cfg.TRIPLET_LOSS:
            self._shuffle_data()
        else:
            np.random.shuffle(self._data._sample)

    def setup(self, bottom, top):
        """Setup the DataLayer."""

        if cfg.TRIPLET_LOSS:
            self.batch_size = cfg.TRIPLET_BATCH_SIZE
        else:
            self.batch_size = cfg.BATCH_SIZE
        self._name_to_top_map = {
            'data': 0,
            'labels': 1}

        self._index = 0
        self._epoch = 1

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(self.batch_size, 3, 224, 224)

        top[1].reshape(self.batch_size)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            top[top_ind].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
