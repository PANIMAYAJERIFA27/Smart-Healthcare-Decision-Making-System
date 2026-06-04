from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def lstm_layer(lstm_dt, layer):
    lstm_struct = Dense(layer,activation='relu')
    return lstm_struct

