# Triplet

This is the re-implementation of triplet loss function in Google's [FaceNet paper](http://arxiv.org/abs/1503.03832v1).

## Dataset

[FaceScrub](http://vintage.winklerbros.net/facescrub.html) is used as traning dataset, you should modify `sampledata.py` to fit your dataset.

## Setup

Rebuild your caffe directory:

	cd $CAFFEROOT$
	cp Makefile.configexample Makefile.config

Remember to uncomment the line to makesure your python layers could be found:

	WITH_PYTHON_LAYER := 1

Then build caffe and pycaffe:

	make all -j8 & make pycaffe

## Usage

1. Modify `sampledata.py`, `config.py` and `train.py` to fit your dataset and working environment.

2. Pre-train your model with softmax loss.

3. Finetune triplet model based on your pre-trained model.

4. Learn to adjust parameters.
