import cPickle

class cifar10():

'''
My cifar 10 class, code mostly ripped from Keras.
Building as a class so I can read batches.
'''
    
    def __init__(self):
        self.fp = '/data/cifar-10/'
        self.x_trn = None
        self.x_val = None
        self.x_tst = None
        self.y_trn = None
        self.y_val = None
        self.y_tst = None

