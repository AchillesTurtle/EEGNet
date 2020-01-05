from keras.models import Model
from keras.layers.core import Dense, Activation, Permute, SpatialDropout2D, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l1_l2
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Flatten, merge, Reshape
import keras.backend as K


def square(x):
    return K.square(x)

def log(x):
    return K.log(x)    
    
def EEGNet(nb_classes, Chans = 64, Samples = 128, regRate = 0.001, #0.001
           dropoutRate = 0.25, kernels = [(2, 32), (8, 4)]):
    """ Keras implementation of EEGNet (arXiv 1611.08024)
    This model is only defined for 128Hz signals. For any other sampling rate
    you'll need to scale the length of the kernels at layer conv2 and conv3
    appropriately (double the length for 256Hz, half the length for 64Hz, etc.)
    
    This also implements a slight variant of the original EEGNet article, where
    we use striding instead of maxpooling. The performance of the network 
    is about the same, although execution time is a bit faster. 
    
    @params 
    nb_classes: total number of final categories
    Chans: number of EEG channels
    Samples: number of EEG sample points per trial
    regRate: regularization rate for L1 and L2 regularizations
    dropoutRate: dropout fraction
    kernels: the 2nd and 3rd layer kernel dimensions (default is the
    [2, 32] x [8, 4]).
    
    """

    # start the model
    input_main   = Input((1, Chans, Samples))
    layer1       = Conv2D(16, (Chans, 1), input_shape=(1, Chans, Samples),
                                 kernel_regularizer = l1_l2(l1=regRate, l2=regRate))(input_main)
    layer1       = BatchNormalization(axis=1)(layer1)
    layer1       = ELU()(layer1)
    layer1       = Dropout(dropoutRate)(layer1)
    
    permute_dims = 2, 1, 3
    permute1     = Permute(permute_dims)(layer1)
    
    layer2       = Conv2D(4, kernels[0], padding = 'same', 
                            kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                            strides = (2, 4))(permute1)#2,4
    layer2       = BatchNormalization(axis=1)(layer2)
    layer2       = ELU()(layer2)
    layer2       = Dropout(dropoutRate)(layer2)
    
    layer3       = Conv2D(4, kernels[1], padding = 'same',
                            kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                            strides = (2, 4))(layer2)#2,4
    layer3       = BatchNormalization(axis=1)(layer3)
    layer3       = ELU()(layer3)
    layer3       = Dropout(dropoutRate)(layer3)
    
    flatten      = Flatten()(layer3)


    dense = Dense(nb_classes)(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


def EEGNetOrg(nb_classes, Chans=64, Samples=128, regRate=0.001,  # 0.001
           dropoutRate=0.25, kernels=[(2, 32), (8, 4)]):

    # start the model
    input_main = Input((1, Chans, Samples))
    layer1 = Conv2D(16, (Chans, 1), input_shape=(1, Chans, Samples),
                    kernel_regularizer=l1_l2(l1=regRate, l2=regRate))(input_main)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = ELU()(layer1)
    layer1 = Dropout(dropoutRate)(layer1)

    permute_dims = 2, 1, 3
    permute1 = Permute(permute_dims)(layer1)

    layer2 = Conv2D(4, kernels[0], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                    strides=(1, 1))(permute1)  # 2,4
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = ELU()(layer2)
    layer2 = MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(layer2)
    layer2 = Dropout(dropoutRate)(layer2)

    layer3 = Conv2D(4, kernels[1], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                    strides=(1, 1))(layer2)  # 2,4
    layer3 = BatchNormalization(axis=1)(layer3)
    layer3 = ELU()(layer3)
    layer3 = MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(layer3)
    layer3 = Dropout(dropoutRate)(layer3)

    flatten = Flatten()(layer3)

    dense = Dense(nb_classes)(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)
    
    
def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Dense Convolutional Network as described in
    Schirrmeister et. al. (2017), arXiv 1703.0505
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 10 for EEG 
    signals sampled at 250Hz. This explains why we're using length 5 
    convolutions for 128Hz sampled data (approximately half). We keep the 
    maxpool at (1, 3) with (1, 3) strides. 
    """

    # start the model
    input_main   = Input((1, Chans, Samples))
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(1, Chans, Samples))(input_main)
    block1       = Conv2D(25, (Chans, 1))(block1)
    block1       = BatchNormalization(axis=1)(block1)
    block1       = ELU()(block1)
    block1       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5))(block1)
    block2       = BatchNormalization(axis=1)(block2)
    block2       = ELU()(block2)
    block2       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5))(block2)
    block3       = BatchNormalization(axis=1)(block3)
    block3       = ELU()(block3)
    block3       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5))(block3)
    block4       = BatchNormalization(axis=1)(block4)
    block4       = ELU()(block4)
    block4       = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes)(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

    
def ShallowConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), arXiv 1703.0505
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
                ours        original paper
    pool_size   1, 35       1, 75
    strides     1, 7        1, 15
    """

    # start the model
    input_main   = Input((1, Chans, Samples))
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(1, Chans, Samples))(input_main)
    block1       = Conv2D(40, (Chans, 1))(block1)
    block1       = BatchNormalization(axis=1)(block1)
    block1       = Activation(square)(block1)
    block1       = Dropout(dropoutRate)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
  
    flatten      = Flatten()(block1)
    
    dense        = Dense(nb_classes)(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)
