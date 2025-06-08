import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from BHI_utils.utils import monte_carlo_dropout_predictions, calculate_uncertainty, find_best_threshold
from sklearn.utils.class_weight import compute_class_weight


class InitialModel(kt.HyperModel):
    def __init__(self, input_dim, output_dim=1, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, hp):
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))

        # Define dropout rate and number of layers
        dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        l2_reg = hp.Float('weight_decay', min_value=1e-3, max_value=1e-1, sampling='log')
        num_layers = hp.Int('num_layers', min_value=1, max_value=5, step=1)

        # Add layers with dynamically chosen units
        for i in range(num_layers):
            units = hp.Choice(f'units_{i}', values=[32 * (2 ** j) for j in range(int(np.log2(self.input_dim // 32)) + 1)])
            model.add(Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(rate=dropout_rate))

        # Output layer
        model.add(Dense(self.output_dim, activation='sigmoid' if self.output_dim == 1 else 'softmax'))
        model.summary()

        loss = hp.Choice('loss', [('binary_' if self.output_dim == 1 else 'categorical_') + 'crossentropy',
                                  ('binary_' if self.output_dim == 1 else 'categorical_') + 'focal_crossentropy'])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(
                          learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
                      loss=loss,
                      metrics=['accuracy',
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc'),
                               tf.keras.metrics.AUC(curve='PR', name='auprc')])
        return model

    def fit(self, hp, model, *args, **kwargs):
        del kwargs['kseed']  # Remove kseed from kwargs
        return model.fit(*args, **kwargs)

class ErmHyperModel(kt.HyperModel):
    def __init__(self, model_fn, **kwargs):
        self.model_fn = model_fn
        super().__init__(**kwargs)

    def build(self, hp):
        tf.keras.backend.clear_session()
        model = self.model_fn()
        return model

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

        a = hp.Float("alpha", min_value=0.1, max_value=0.9, step=0.1)
        kwargs['sample_weight'] = np.concatenate([np.full(len(x_source), 1-a),  # Weight for source samples
                                                  np.full(len(x_target), a)])  # Weight for target samples

        del kwargs['kseed'], kwargs['studies'], kwargs['source_study'], kwargs['target_study']
        return mdl.fit(x_adapt, y_adapt, validation_data=(x_target_val, y_target_val), **kwargs)

class ActiveLearningHyperModel(kt.HyperModel):
    def __init__(self, model_fn, select_fn, uncertainty_threshold=None, **kwargs):
        self.model_fn = model_fn
        self.select = select_fn
        self.uncertainty_threshold = uncertainty_threshold
        super().__init__(**kwargs)

    def build(self, hp):
        if self.uncertainty_threshold:
            hp.Fixed('uncertainty_threshold', self.uncertainty_threshold)
        tf.keras.backend.clear_session()
        model = self.model_fn()
        return model

    def fit(self, hp, mdl, X, y, studies, target_study, **kwargs):
        # Rank the informativeness of samples in the target study based on Monte Carlo Dropout.
        # Select the 20% most uncertain samples to annotate.
        # Use a hyperparameter threshold to distinguish certain and uncertain samples.
        # Use certain samples with pseudo-labels for training.

        np.random.seed(kwargs['kseed'])

        # Split target study data
        x_target, y_target = X[studies == target_study], y[studies == target_study]

        x_selected_train, x_selected_val, x_pseudo_train, x_pseudo_val, y_selected_train, y_selected_val, y_pseudo_train, y_pseudo_val = self.select(model=mdl, x=x_target, y=y_target, hp=hp, **kwargs)

        # Prepare training data
        x_train = np.concatenate([x_selected_train, x_pseudo_train])
        y_train = np.concatenate([y_selected_train, y_pseudo_train])

        if ('pseudo_weights' in kwargs) and (kwargs['pseudo_weights']):
            alpha = hp.Float("pseudo_weight", min_value=0.1, max_value=0.5, step=0.2)
            kwargs['sample_weight'] = np.concatenate([np.full(len(x_selected_train), 1-alpha),
                                                      np.full(len(x_pseudo_train), alpha)])
            del kwargs['pseudo_weights']

        x_val = np.concatenate([x_selected_val, x_pseudo_val])
        y_val = np.concatenate([y_selected_val, y_pseudo_val])

        classes = np.array([0, 1])
        # print(np.isin(classes, np.unique(y_selected_train)))
        # print(np.isin(classes, np.unique(y_train)))
        # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_selected_train)
        if np.all(np.isin(classes, np.unique(y_selected_train))):
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_selected_train)
        elif np.all(np.isin(classes, np.unique(y_train))):
            print(30*'Only one class in selected, two in pseudo.\n')
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        else:
            print(30*'Only one class.\n')
            class_weights = np.array([0.2 if np.isin(i, np.unique(y_selected_train)) else 0.8 for i in classes])

        class_weight_dict = dict(zip(classes, class_weights))

        if 'sample_weight' in kwargs:
            class_weights = np.array([class_weight_dict[y] for y in y_train])
            kwargs['sample_weight'] = kwargs['sample_weight'] + class_weights
        else:
            kwargs['class_weight'] = class_weight_dict

        # Remove used kwargs
        del kwargs['kseed'], kwargs['source_study']

        return mdl.fit(x_train, y_train, validation_data=(x_val, y_val), **kwargs)

class ActiveLearningSourceAwareHyperModel(ActiveLearningHyperModel):
    def fit(self, hp, mdl, X, y, source_study, **kwargs):
        # Rank the informativeness of samples in the target study based on Monte Carlo Dropout.
        # Select the 20% most uncertain samples to annotate.
        # Use a hyperparameter threshold to distinguish certain and uncertain samples.
        # Use certain samples with pseudo-labels for training.

        np.random.seed(kwargs['kseed'])
        target_study = kwargs['target_study']
        studies = kwargs['studies']

        # Split target study data
        x_target, y_target = X[studies == target_study], y[studies == target_study]
        # Split source study data
        x_source, y_source = X[studies == source_study], y[studies == source_study]

        x_selected_train, x_selected_val, x_pseudo_train, x_pseudo_val, y_selected_train, y_selected_val, y_pseudo_train, y_pseudo_val = self.select(
            model=mdl, x=x_target, y=y_target, hp=hp, **kwargs)

        # Prepare training data
        x_train = np.concatenate([x_source, x_selected_train, x_pseudo_train])
        y_train = np.concatenate([y_source, y_selected_train, y_pseudo_train])

        if ('pseudo_weights' in kwargs) and (kwargs['pseudo_weights']):
            alpha = hp.Float("pseudo_weight", min_value=0.1, max_value=0.5, step=0.2)
            kwargs['sample_weight'] = np.concatenate([np.full(len(x_source), 1 - alpha),
                                                      np.full(len(x_selected_train), 1),
                                                      np.full(len(x_pseudo_train), alpha)])
            del kwargs['pseudo_weights']

        x_val = np.concatenate([x_selected_val, x_pseudo_val])
        y_val = np.concatenate([y_selected_val, y_pseudo_val])

        classes = np.array([0, 1])
        # print(np.isin(classes, np.unique(y_selected_train)))
        # print(np.isin(classes, np.unique(y_train)))
        # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_selected_train)
        if np.all(np.isin(classes, np.unique(y_selected_train))):
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_selected_train)
        elif np.all(np.isin(classes, np.unique(y_train))):
            print(30*'Only one class in selected, two in pseudo.\n')
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        else:
            print(30*'Only one class.\n')
            class_weights = np.array([0.2 if np.isin(i, np.unique(y_selected_train)) else 0.8 for i in classes])

        class_weight_dict = dict(zip(classes, class_weights))

        if 'sample_weight' in kwargs:
            class_weights = np.array([class_weight_dict[y] for y in y_train])
            kwargs['sample_weight'] = kwargs['sample_weight'] + class_weights
        else:
            kwargs['class_weight'] = class_weight_dict

        # Remove used kwargs
        del kwargs['kseed']
        for layer in mdl.layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'kernel'):
                layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
            if hasattr(layer, 'bias_initializer') and hasattr(layer, 'bias'):
                layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))
            # For layers like BatchNormalization
            if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
                layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))
                layer.moving_variance.assign(tf.ones_like(layer.moving_variance))

        return mdl.fit(x_train, y_train, validation_data=(x_val, y_val), **kwargs)

class MCDropoutUncertaintyHyperModel(kt.HyperModel):
    def __init__(self, model_fn, **kwargs):
        self.model_fn = model_fn
        super().__init__(**kwargs)

    def build(self, hp):
        tf.keras.backend.clear_session()
        return self.model_fn()

    def fit(self, hp, model, *args, **kwargs):
        num_samples = hp.Int('num_samples', 10, 100, step=10)
        metric_name = hp.Choice('uncertainty_metric', ['total_variance', 'std', 'entropy'])

        # MC Dropout predictions
        mc_preds = monte_carlo_dropout_predictions(model, args[0], num_samples=num_samples)
        uncertainty = calculate_uncertainty(mc_preds, metric_name)

        y_pred = (mc_preds.mean(axis=0) > 0.5).astype(int)
        incorrect = (y_pred != args[1].flatten()).astype(int) # 1 for incorrect, 0 for correct
        auc = roc_auc_score(incorrect, uncertainty) if len(np.unique(incorrect))==2 else accuracy_score(incorrect, (uncertainty>find_best_threshold(incorrect, uncertainty))*1)

        hp.Fixed('uncertainty_threshold', find_best_threshold(incorrect, uncertainty))

        # Set the trial score
        model.history = type('', (), {})()  # Dummy history
        model.history.history = {'val_auc': [auc]}
        return model.history.history