import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import TextVectorization
import numpy as np


class NlpPredictionPipeline():
    def __init__(self) -> None:
        self.CLASS_NAMES = ['Negative Review', 'Positive Review']

        # Loading model
        self.model = load_model('./Deep learning model/Bi_LSTM_model.h5')
        
        # Building Tensorflow TextVectorization Layer
        max_vocab_size = 20000
        max_length = 25

        self.text_vectorization_layer = TextVectorization(max_tokens=max_vocab_size, # how many words in the vocabulary
                                            output_sequence_length=max_length,
                                            output_mode='int')
        self.text_vectorization_weights = np.load("text_vectorization_weights.npy", allow_pickle=True)
        # Set the loaded weights to the new TextVectorization layer
        self.text_vectorization_layer.set_weights(self.text_vectorization_weights)
    
    def predict_review_sentiment(self, text):
        input_tensor = self.text_vectorization_layer([text])
        y_probs = self.model.predict(input_tensor)
        return tf.round(y_probs), y_probs
        
