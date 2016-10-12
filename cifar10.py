import numpy as np

class cifar10():
    
    def __init__(self):
        self.fp = '/data/cifar-10/data/'
        self.x_trn = None
        self.x_tst = None
        self.y_trn = None
        self.y_tst = None
        self.max_trn = None
        self.max_trn = None

        self._read_data()

    def _read_data(self):
        x_trn = np.load(self.fp + 'train_x.npy')
        y_trn = np.load(self.fp + 'train_y.npy')
        x_tst = np.load(self.fp + 'test_x.npy')
        y_tst = np.load(self.fp + 'test_y.npy')

        x_trn = x_trn / 255.
        x_tst = x_tst / 255.
        y_trn = y_trn.ravel()
        y_tst = y_tst.ravel()

        self.x_trn = x_trn
        self.x_tst = x_tst
        self.y_trn = self._one_hot(y_trn)
        self.y_tst = self._one_hot(y_tst)

        self.trn_max = len(x_trn)
        self.tst_max = len(x_tst)
        self.trn_curr = 0
        self.tst_curr = 0
        self.trn_idxs = np.arange(self.trn_max)

    def next_trn(self, n):
       if self.trn_curr + n > self.trn_max:
            self.trn_curr = 0
            np.random.shuffle(self.trn_idxs)
            return False
       else:
            start = self.trn_curr
            end = self.trn_curr + n
            idx = self.trn_idxs[start:end]
            self.trn_curr = end
            return self.x_trn[idx], self.y_trn[idx]

    def next_tst(self, n):
        if self.tst_curr + n > self.tst_max:
            self.tst_curr = 0
            return False
        else:
            start = self.tst_curr
            end = self.tst_curr + n
            self.tst_curr = end
            return self.x_tst[start:end], self.y_tst[start:end]

    def _one_hot(self, arr):
        n = len(arr)
        n_class = 10
        one = np.zeros((n, n_class))
        one[np.arange(n), arr] = 1
        return one

if __name__ == '__main__':
    test = cifar10()
