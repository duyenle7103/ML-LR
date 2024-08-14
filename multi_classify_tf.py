"""multi_classify_tf.py
This file is for multi-class classification using TensorFlow
Author: Kien Huynh
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_util import get_iris_data
from bin_classify_np import add_one, add_poly_feature

if __name__ == "__main__":
    # Random seed is fixed so that every run is the same
    # This makes it easier to debug
    np.random.seed(2017)

    # Load data from file
    # Make sure that eclipse-data.npz is in data/
    train_x, train_y, test_x, test_y = get_iris_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]  

    # Add more features to train_x and test_x
    train_x = add_poly_feature(train_x, 2)
    test_x = add_poly_feature(test_x, 2)
    
    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x) 
    test_x = add_one(test_x)

    # TODO:[YC2.3] Create TF placeholders to feed train_x and train_y when training
    x = None
    y = None

    # TODO:[YC2.3] Create weights (W) using TF variables
    w1 = None
    w2 = None
    w3 = None

    # TODO:[YC2.3] Create a feed-forward operator
    pred1 = None
    pred2 = None
    pred3 = None

    # TODO:[YC2.3] Write the cost function
    cost1 = None
    cost2 = None
    cost3 = None

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.005    

    # TODO:[YC2.3] Implement GD
    optimizer1 = None
    optimizer2 = None
    optimizer3 = None 

    # Start training
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            # TODO:[YC2.3] Update weights here

            loss = 0 
            print("Epoch %d: loss is %.5f" % (e+1, loss))

    # TODO:[YC2.3] Compute test result using confusion matrix
    c_mat = np.zeros((3,3))
    print(c_mat)
