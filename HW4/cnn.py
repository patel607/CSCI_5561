import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    mini_batch_x = []
    mini_batch_y = []

    perm = np.random.permutation(im_train.shape[1])
    
    for i in range(0, len(perm), batch_size):
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            if (i+j < len(perm)):
                batch_x.append(im_train[:, perm[i+j]])
                encoding = np.zeros(10)
                encoding[label_train[0, perm[i+j]]] = 1
                batch_y.append(encoding)

        mini_batch_x.append(np.asarray(batch_x))
        mini_batch_y.append(np.asarray(batch_y))

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    y = np.matmul(w, x) + b
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_dx = np.matmul(dl_dy, w)

    dl_dw = np.outer(dl_dy , x).reshape(w.shape)

    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    l = np.linalg.norm(y - y_tilde)**2

    dl_dy = ((y_tilde - y) * 2).T
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    # TO DO
    
    xrel = x - x.max()
    y_tilde = np.exp(xrel) / np.sum(np.exp(xrel))
    
    l = - np.sum(y * np.log(y_tilde+1))

    dl_dy = (y_tilde - y).T
    return l, dl_dy

epsilon = 0.9

def relu(x):
    # TO DO
    y = np.where(x>0, x, x*epsilon)
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    dl_dx = np.where(dl_dy>0, dl_dy, dl_dy*epsilon)
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    H, W, C1 = x.shape
    h, w, _, C2 = w_conv.shape

    h_pad = int(h/2)
    w_pad = int(w/2)
    x_padded = np.pad(x, ((h_pad, h_pad), (w_pad, w_pad), (0,0)))

    y = np.zeros((H, W, C2))

    for c in range(C2):
        for i in range(h_pad, x_padded.shape[0] - h_pad):
            for j in range(w_pad, x_padded.shape[1] - w_pad):
                inp = x_padded[(i - h_pad):(i + h_pad+1), (j - w_pad):(j + w_pad+1), :]
                convolved = np.sum(np.multiply(inp, w_conv[:,:,:,c])) + b_conv[c]
                y[i - h_pad, j - w_pad, c] = convolved

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    H, W, C1 = x.shape
    h, w, _, C2 = w_conv.shape

    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)

    h_pad = int(h/2)
    w_pad = int(w/2)
    x_padded = np.pad(x, ((h_pad, h_pad), (w_pad, w_pad), (0,0)))

    for c2 in range(C2):
        L = dl_dy[:, :, c2]
        dl_db[c2] = np.sum(L)
        for c1 in range(C1):
            for i in range(h):
                for j in range(w):
                    X = x_padded[i: i + H, j:j + W, c1]
                    dl_dw[i, j, c1, c2] = np.sum(np.multiply(L, X))
    dl_db = dl_db.reshape((1,3))
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    H, W, C = x.shape
    y = np.zeros((int(H/2), int(W/2), C))

    H = int(H/2) * 2
    W = int(W/2) * 2

    for c in range(C):
        pool = x[:,:,c]
        for h in range(0,H,2):
            for w in range(0,W,2):
                y[int(h/2), int(w/2), c] = np.amax(pool[h:h+2, w:w+2])
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    H, W, C = x.shape
    dl_dx = np.zeros(x.shape)

    H = int(H/2) * 2
    W = int(W/2) * 2

    for c in range(C):
        pool = x[:,:,c]
        for h in range(0,H,2):
            for w in range(0,W,2):
                max_ind = np.argmax(pool[h:h+2, w:w+2])
                dl_dx[h+int(max_ind/2), w+max_ind%2, c] = dl_dy[int(h/2), int(w/2), c]
    return dl_dx


def flattening(x):
    # TO DO
    # y = x.reshape((np.size(x), 1))
    y = x.reshape(-1, order='F')
    y = y.reshape((y.shape[0], 1))
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy.reshape(x.shape, order='F')
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    # set rates
    gamma = 0.001
    lamda = 0.9

    m = 196
    n = 10

    # initialize weights with random noise
    w = np.random.rand(n, m)
    b = np.random.rand(n, 1)

    k = 0
    nIters = 2000
    loss = []

    for iIter in range(nIters):
        if (iIter + 1) % 1000 == 0:
            gamma = lamda * gamma
        
        dL_dw = np.zeros(w.shape)
        dL_db = np.zeros(n)
        mbl = 0

        batch_size = len(mini_batch_x[k])

        for j in range(batch_size):
            # label prediction
            x = mini_batch_x[k][j].reshape((m,1))
            y_tilde = fc(x, w, b);
            y = mini_batch_y[k][j].reshape((n,1))
            # loss computation
            l, dl_dy = loss_euclidean(y_tilde, y)
            mbl += np.abs(l)
            # gradient back-propagation
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            dL_dw = dL_dw + dl_dw
            dL_db = dL_db + dl_db

        k += 1
        if (k == len(mini_batch_x)):
            k = 0

        loss.append(mbl)
        w -= gamma * dL_dw
        b -= gamma * dL_db.reshape(b.shape)


    # plt.xlabel('iterations', fontsize=18)
    # plt.ylabel('training loss', fontsize=16)
    # plt.plot(loss)
    # plt.show()

    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    # set rates
    gamma = 0.01
    lamda = 0.9

    m = 196
    n = 10

    # initialize weights with random noise
    w = np.random.rand(n, m)
    b = np.random.rand(n, 1)

    k = 0
    nIters = 2000
    loss = []

    for iIter in range(nIters):
        if (iIter + 1) % 1000 == 0:
            gamma = lamda * gamma
        
        dL_dw = np.zeros(w.shape)
        dL_db = np.zeros(n)
        mbl = 0

        batch_size = len(mini_batch_x[k])

        for j in range(batch_size):
            # label prediction
            x = mini_batch_x[k][j].reshape((m,1))
            y_tilde = fc(x, w, b);
            y = mini_batch_y[k][j].reshape((n,1))
            # loss computation
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            mbl += np.abs(l)
            # gradient back-propagation
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            dL_dw = dL_dw + dl_dw
            dL_db = dL_db + dl_db

        k += 1
        if (k == len(mini_batch_x)):
            k = 0

        loss.append(mbl)
        w -= gamma * dL_dw
        b -= gamma * dL_db.reshape(b.shape)


    # plt.xlabel('iterations', fontsize=18)
    # plt.ylabel('training loss', fontsize=16)
    # plt.plot(loss)
    # plt.show()

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    # set rates
    gamma = 0.01
    lamda = 0.9

    m = 196
    n = 10
    p = 30

    # initialize weights with random noise
    w1 = np.random.rand(p, m)
    b1 = np.random.rand(p, 1)    
    w2 = np.random.rand(n, p)
    b2 = np.random.rand(n, 1)

    k = 0
    nIters = 20000
    loss = []

    for iIter in range(nIters):
        if (iIter + 1) % 500 == 0:
            gamma = lamda * gamma
        
        dL_dw1 = np.zeros(w1.shape)
        dL_db1 = np.zeros(p)
        dL_dw2 = np.zeros(w2.shape)
        dL_db2 = np.zeros(n)
        mbl = 0

        batch_size = len(mini_batch_x[k])

        for j in range(batch_size):
            # label prediction
            x = mini_batch_x[k][j].reshape((m,1))
            y = mini_batch_y[k][j].reshape((n,1))

            fc_1 = fc(x, w1, b1);
            # ReLU 1
            relu_1 = relu(fc_1)
            # FC 2
            fc_2 = fc(relu_1, w2, b2)
            # loss computation
            l, dl_dy = loss_cross_entropy_softmax(fc_2, y)
            mbl += np.abs(l)
            # gradient back-propagation
            dl_dx, dl_dw2, dl_db2 = fc_backward(dl_dy, relu_1, w2, b2, fc_2)

            relu_b = relu_backward(dl_dx, fc_1, relu_1)

            dl_dx, dl_dw1, dl_db1 = fc_backward(relu_b, x, w1, b1, fc_1)
            
            dL_dw1 = dL_dw1 + dl_dw1
            dL_db1 = dL_db1 + dl_db1
            dL_dw2 = dL_dw2 + dl_dw2
            dL_db2 = dL_db2 + dl_db2

        k += 1
        if (k == len(mini_batch_x)):
            k = 0

        loss.append(mbl)
        w1 -= gamma * dL_dw1
        b1 -= gamma * dL_db1.reshape(b1.shape)
        w2 -= gamma * dL_dw2
        b2 -= gamma * dL_db2.reshape(b2.shape)


    # plt.xlabel('iterations', fontsize=18)
    # plt.ylabel('training loss', fontsize=16)
    # plt.plot(loss)
    # plt.show()
    
    return w1, b1, w2, b2

def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    # set rates
    gamma = 0.01
    lamda = 0.9

    m = 196
    n = 10
    p = 30

    # initialize weights with random noise
    w_conv = np.random.rand(3,3,1,3)
    b_conv = np.random.rand(3)
    w_fc = np.random.rand(10, 147)
    b_fc = np.random.rand(10, 1)

    k = 0
    nIters = 10000
    loss = []

    for iIter in range(nIters):
        if (iIter + 1) % 500 == 0:
            gamma = lamda * gamma
        print(iIter)
        dL_dw_conv = np.zeros(w_conv.shape)
        dL_db_conv = np.zeros(3)
        dL_dw_fc = np.zeros(w_fc.shape)
        dL_db_fc = np.zeros(10)
        mbl = 0

        batch_size = len(mini_batch_x[k])

        for j in range(batch_size):
            # label prediction
            x = mini_batch_x[k][j].reshape((14,14,1), order='F')
            y = mini_batch_y[k][j].reshape((n,1))

            conv_1 = conv(x, w_conv, b_conv)
            # print(conv_1.shape)
            # ReLU 1
            relu_1 = relu(conv_1)
            # print(relu_1.shape)
            # FC 2
            pool_1 = pool2x2(relu_1)
            # print(pool_1.shape)
            flatten_1 = flattening(pool_1)
            # print(flatten_1.shape)
            fc_1 = fc(flatten_1, w_fc, b_fc)
            # print(fc_1.shape)
            # loss computation
            l, dl_dy = loss_cross_entropy_softmax(fc_1, y)
            mbl += np.abs(l)
            # print(dl_dy.shape)
            # gradient back-propagation
            dl_dx, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, flatten_1, w_fc, b_fc, fc_1)
            # print(dl_dx.shape)
            flatten_b = flattening_backward(dl_dx, pool_1, flatten_1)

            pool_b = pool2x2_backward(flatten_b, relu_1, pool_1)

            relu_b = relu_backward(pool_b, conv_1, relu_1)

            dl_dw_conv, dl_db_conv = conv_backward(relu_b, x, w_conv, b_conv, conv_1)
            
            dL_dw_conv = dL_dw_conv + dl_dw_conv
            # dL_db_conv = dL_db_conv + dl_db_conv
            dL_dw_fc = dL_dw_fc + dl_dw_fc
            dL_db_fc = dL_db_fc + dl_db_fc

        k += 1
        if (k == len(mini_batch_x)):
            k = 0

        loss.append(mbl)
        w_conv -= gamma * dL_dw_conv
        # b_conv -= gamma * dL_db_conv.reshape(b_conv.shape)
        w_fc -= gamma * dL_dw_fc
        b_fc -= gamma * dL_db_fc.reshape(b_fc.shape)


    # plt.xlabel('iterations', fontsize=18)
    # plt.ylabel('training loss', fontsize=16)
    # plt.plot(loss)
    # plt.show()
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    # main.main_slp_linear()
    # main.main_slp()
    # main.main_mlp()
    main.main_cnn()




    mnist_train = sio.loadmat('./mnist_train.mat')
    mnist_test = sio.loadmat('./mnist_test.mat')
    im_train, label_train = mnist_train['im_train'], mnist_train['label_train']
    im_test, label_test = mnist_test['im_test'], mnist_test['label_test']
    batch_size = 32
    im_train, im_test = im_train / 255.0, im_test / 255.0
    mini_batch_x, mini_batch_y = get_mini_batch(im_train, label_train, batch_size)



