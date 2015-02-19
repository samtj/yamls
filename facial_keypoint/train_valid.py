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
    
    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    train_loss2 = np.array([i["train_loss"] for i in net2.train_history_])
    valid_loss2 = np.array([i["valid_loss"] for i in net2.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="net1 train")
    pyplot.plot(valid_loss, linewidth=3, label="net1 valid")
    pyplot.plot(train_loss2, linewidth=3, label="net2 train")
    pyplot.plot(valid_loss2, linewidth=3, label="net2 valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()


if __name__ == "__main__":
    main()