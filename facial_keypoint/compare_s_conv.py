from facial_keypoint import convolution
import single
from facial_keypoint import helper
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as pyplot
from facial_keypoint import helper
import numpy as np
import cPickle as pickle

def main():
    net1 = single.initialize_single()
    net2 = convolution.initialize_conv()
    
    # Training this neural net is much more computationally costly than the first one we trained.
    # It takes around 15x as long to train; those 1000 epochs take more than 20 minutes on even a powerful GPU.
    
    sample1 = helper.load(test=True)[0][6:7]
    sample2 = helper.load2d(test=True)[0][6:7]
    y_pred1 = net1.predict(sample1)[0]
    y_pred2 = net2.predict(sample2)[0]
    
    fig = pyplot.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    helper.plot_sample(sample1[0], y_pred1, ax)
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    helper.plot_sample(sample1[0], y_pred2, ax)
    fig.savefig('comparison.png', dpi=100)
    pyplot.show()

if __name__ == "__main__":
    main()
