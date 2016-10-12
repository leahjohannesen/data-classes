import numpy as np
import os
import sys
import cPickle

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding="bytes")
        # decode utf8
        for k, v in d.items():
            del(d[k])
            d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def cifar10():
    path = '/data/cifar-10/'

    nb_train_samples = 50000

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        X_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    X_train = X_train.transpose(0, 2, 3, 1)
    X_test = X_test.transpose(0, 2, 3, 1)
    
    train_path = path + 'data/train'
    test_path = path + 'data/test'

    np.save(train_path + '_x', X_train)
    np.save(train_path + '_y', y_train)
    np.save(test_path + '_x', X_test)
    np.save(test_path + '_y', y_test)

def cifar100():
    path = '/data/cifar-100/'
    label_mode = 'fine'
    
    fpath = os.path.join(path, 'train')
    X_train, y_train = load_batch(fpath, label_key=label_mode+'_labels')

    fpath = os.path.join(path, 'test')
    X_test, y_test = load_batch(fpath, label_key=label_mode+'_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    X_train = X_train.transpose(0, 2, 3, 1)
    X_test = X_test.transpose(0, 2, 3, 1)

    train_path = path + 'data/train'
    test_path = path + 'data/test'

    np.save(train_path + '_x', X_train)
    np.save(train_path + '_y', y_train)
    np.save(test_path + '_x', X_test)
    np.save(test_path + '_y', y_test)


