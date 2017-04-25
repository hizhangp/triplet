import os
import config as cfg
import numpy as np


class sampledata():

    def __init__(self):
        self._sample_person = {}
        self._sample = []
        self._sample_label = {}
        self._sample_test = []
        face_path = cfg.IMAGEPATH

        for num, personname in enumerate(sorted(os.listdir(face_path))):
            person_path = face_path + personname + '/face'
            picnames = [{'picname': personname + '/face/' + i, 'flipped': False}
                        for i in sorted(os.listdir(person_path))
                        if os.path.getsize(os.path.join(person_path, i)) > 0]
            pic_train = int(len(picnames) * cfg.PERCENT)
            self._sample_person[personname] = picnames[:pic_train]
            self._sample_label[personname] = num
            self._sample.extend(picnames[:pic_train])
            self._sample_test.extend(picnames[pic_train:])

            if cfg.FLIPPED:
                picnames_flipped = [{'picname': i['picname'], 'flipped': True}
                                    for i in picnames[:pic_train]]
                self._sample_person[personname].extend(picnames_flipped)
                self._sample.extend(picnames_flipped)

        print('Number of training persons: {}'.format(len(self._sample_person)))
        print('Number of training images: {}'.format(len(self._sample)))
        print('Number of testing images: {}'.format(len(self._sample_test)))

if __name__ == '__main__':

    sample = sampledata()
