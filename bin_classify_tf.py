"""bin_classify_tf.py
This file is for binary classification using TensorFlow
Author: Kien Huynh
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_util import get_eclipse_data
from bin_classify_np import add_one, add_poly_feature

if __name__ == "__main__":
    # Random seed is fixed so that every run is the same
    # This makes it easier to debug
    np.random.seed(2017)

    # Load data from file
    # Make sure that eclipse-data.npz is in data/
    train_x, train_y, test_x, test_y = get_eclipse_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]  

    # Add more features to train_x and test_x
    train_x = add_poly_feature(train_x, 2)
    test_x = add_poly_feature(test_x, 2)
    
    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x) 
    test_x = add_one(test_x)
   
    # TODO:[YC1.9] Create TF placeholders to feed train_x and train_y when training
    x = None
    y = None

    # TODO:[YC1.9] Create weights (W) using TF variables
    w = None

    # TODO:[YC1.9] Create a feed-forward operator
    pred = None

    # TODO:[YC1.9] Write the cost function
    cost = None

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.005    

    # TODO:[YC1.9] Implement GD
    optimizer = None
 
    # Start training
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            # TODO:[YC1.9] Update weights here

            loss = 0 
            print("Epoch %d: loss is %.5f" % (e+1, loss))

    # TODO:[YC1.9] Compute test result (precision, recall, f1-score)
    precision = 0
    recall = 0
    f1 = 0

    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1-score: %.3f" % f1)

