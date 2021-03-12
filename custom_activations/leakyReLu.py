from keras.layers import *
class LeakyRELU(LeakyReLu):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyRELU"
        super(LeakyRELU, self).__init__(**kwargs)