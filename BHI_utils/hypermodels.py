import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
import tensorflow as tf
import numpy as np

class InitialModel(kt.HyperModel):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super().__init__(**kwargs)

    def build(self, hp):
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))

        # Define dropout rate and number of layers
        dropout_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        num_layers = hp.Int('num_layers', min_value=1, max_value=5, step=1)

        # Add layers with dynamically chosen units
        for i in range(num_layers):
            units = hp.Choice(f'units_{i}', values=[32 * (2 ** j) for j in range(int(np.log2(self.input_dim // 32)) + 1)])
            model.add(Dense(units=units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(rate=dropout_rate))

        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(
                          learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
                      loss='binary_crossentropy',
                      metrics=['accuracy',
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc'),
                               tf.keras.metrics.AUC(curve='PR', name='auprc')])
        return model