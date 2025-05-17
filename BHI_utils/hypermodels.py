import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

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

class ErmHyperModel(kt.HyperModel):
    def __init__(self, model, **kwargs):
        self.model = model
        self.original_weights = model.get_weights()
        super().__init__(**kwargs)

    def build(self, hp):
        a = hp.Float("alpha", min_value=0.1, max_value=0.9, step=0.1)
        self.model.set_weights(self.original_weights)
        return self.model

    def fit(self, hp, mdl, *args, **kwargs):
        # Combine source study data with 10% of target study data
        x_target, _, y_target, _ = train_test_split(
            args[0][kwargs['studies'] == kwargs['target_study']],
            args[1][kwargs['studies'] == kwargs['target_study']], test_size=0.8, random_state=kwargs['kseed']
        )
        x_target, x_target_val, y_target, y_target_val = train_test_split(x_target, y_target, test_size=0.5,
                                                                          random_state=kwargs['kseed'])

        x_source = args[0][kwargs['studies'] == kwargs['source_study']]
        y_source = args[1][kwargs['studies'] == kwargs['source_study']]

        x_adapt = np.concatenate([x_source, x_target])
        y_adapt = np.concatenate([y_source, y_target])

        a = hp.get("alpha")
        kwargs['sample_weight'] = np.concatenate([np.full(len(x_source), 1-a),  # Weight for source samples
                                                  np.full(len(x_target), a)])  # Weight for target samples
        del kwargs['kseed'], kwargs['studies'], kwargs['source_study'], kwargs['target_study']
        return mdl.fit(x_adapt, y_adapt, **kwargs)