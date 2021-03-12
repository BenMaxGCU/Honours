from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

#def swish(x, beta = 1):
    #return (x * K.sigmoid(beta * x))

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})