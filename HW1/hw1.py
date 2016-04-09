import os 
import struct
from array import array as pyarray
from pylab import *
from numpy import *
from numpy.linalg import eig, norm
from numpy.matlib import repmat

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels


def hw1_find_eigendigits(A):
    # get mean vector by averaging the rows
    A = A.astype(float)

    m = mean(A, axis=1)
    
    # replace X with X - E(X)
    B = repmat(m, A.shape[1], 1)

    A = A - transpose(B)
    
    # get vec and val
    print "Before Eig"
    val, vec = eig(dot(transpose(A), A))
    print "After Eig"

    vec = dot(A, vec)
    
    # sort e-vec and e-val
    idx = val.argsort()[::-1]   
    val = val[idx]
    vec = vec[:,idx]
    
    # normalize columns
    V = vec / norm(vec, axis=0)
    V = nan_to_num(V)
    V = around(V, decimals=8).real

    return (m, V)


def hw1_classify(num_train_images, test_start_index, test_end_index, M):
    train_images, train_labels = load_mnist('training', digits=range(0, 10))
    test_images, test_labels = load_mnist('testing', digits=range(0, 10))

    print "Done loading images and labels"

    A = train_images[0].flatten()

    for i in range(1, num_train_images):
        img = train_images[i]
        img = img.flatten()
        A = column_stack((A, img))

    print "Done making the A matrix"

    m, V = hw1_find_eigendigits(A)

    print "Done finding eigen digits"

    V_ = V[:,0:M]
    invV_ = transpose(V_)
    
    train_eigens = train_images[0].flatten()
    train_eigens = train_eigens.astype(float) - m
    train_eigens = dot(invV_, train_eigens)
    
    for i in range(1, num_train_images):
        I = train_images[i].flatten()
        I = I.astype(float) - m
        I = dot(invV_, I)

        train_eigens = column_stack((train_eigens, I))

    print "Done finding training eigens"

    test_eigens = test_images[test_start_index].flatten()
    test_eigens = test_eigens.astype(float) - m
    test_eigens = dot(invV_, test_eigens)
    
    for i in range(test_start_index + 1, test_end_index + 1):
        I = test_images[i].flatten()
        I = I.astype(float) - m
        I = dot(invV_, I)

        test_eigens = column_stack((test_eigens, I))

    print "Done finding testing eigens"

    predictions = k_nearest_neighbors(15, train_eigens, train_labels, test_eigens)

    print "Done making predictions"

    num_right = 0
    num_wrong = 0

    for i, p in enumerate(predictions):
        if p == test_labels[i]:
            num_right += 1
        else:
            num_wrong += 1

    accuracy = float(num_right) / float(num_right + num_wrong)

    print "ACCURACY: " + str(accuracy)


def k_nearest_neighbors(k, train_eigens, train_labels, test_eigens):
    predictions = []

    for i in range(test_eigens.shape[1]):
        test_data = test_eigens[:, i]

        B = repmat(test_data, train_eigens.shape[1], 1)

        dist = train_eigens - transpose(B)
        dist = sum(dist**2, axis=1)

        idx = dist.argsort()   
        dist = dist[idx]

        # find k-nearest labels
        nrstLabels = train_labels[idx]
        nrstLabels = reshape(nrstLabels[0:k], k)

        # choose the most frequent one
        prediction = bincount(nrstLabels).argmax()

        predictions.append(prediction)

    return predictions


hw1_classify(60000, 0, 10000, 25)


