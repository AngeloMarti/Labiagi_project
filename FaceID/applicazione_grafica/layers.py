#importo le librerie
import tensorflow as tf
from keras.layers import Layer

#ricreo la classe della distant layer del siamese model
class L1Dist(Layer):
    
    # metodo init
    def __init__(self, **kwargs):
        super().__init__()
       
    # La funzione call prende in input un immagine da validare a cui poi sottrae un'immagine che puo' essere sia positive sia negative e ritornera' il valore assoluto della sottrazione
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)