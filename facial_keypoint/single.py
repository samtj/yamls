from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as pyplot
from facial_keypoint import helper
import numpy as np
import os.path
import cPickle as pickle

#X, y = helper.load()
#print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
#    X.shape, X.min(), X.max()))
#print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
#    y.shape, y.min(), y.max()))

def initialize_single():
    # if train's already done
    if os.path.exists('net1.pickle'):
        with open(r"net1.pickle", "rb") as input_file:
            net1 = pickle.load(input_file)
        return net1

    
    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 9216),  # 96x96 input pixels per batch - defines the shape parameter of the input layer, None - variable batch size
        hidden_num_units=100,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function - output units' activations become just a linear combination of the activations in the hidden layer
        output_num_units=30,  # 30 target values
        
        # Parameters input_shape, hidden_num_units, output_nonlinearity, and output_num_units are each parameters for specific layers
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01, # learning rate
        update_momentum=0.9, # momentum
    
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=400,  # we want to train this many epochs
        verbose=1,
        )

    X, y = helper.load()
    net1.fit(X, y)

    with open('net1.pickle', 'wb') as f:
        pickle.dump(net1, f, -1)
    
    return net1

def main():
    net1 = initialize_single()
    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()
    
    X, _ = helper.load(test=True)
    y_pred = net1.predict(X)
    
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        helper.plot_sample(X[i], y_pred[i], ax)
    
    pyplot.show()

if __name__ == "__main__":
    main()
