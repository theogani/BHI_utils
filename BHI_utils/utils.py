from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback
import keras_tuner as kt
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_best_trial(tuner):
    # Get all completed trials
    all_trials = [t for t in tuner.oracle.trials.values() if t.status == 'COMPLETED']

    # Find the best score
    best_score = np.max([t.score for t in all_trials])

    # Filter trials with the best score
    best_trials = [t for t in all_trials if t.score == best_score]

    # Pick a deterministic one (e.g., by trial_id)
    return sorted(best_trials, key=lambda t: t.trial_id)[0]


def load(tuner, model_path=None):
    def fun():
        # Path to the best trial's directory
        best_trial = get_best_trial(tuner)

        # Build model with best parameters
        model = tuner.hypermodel.build(best_trial.hyperparameters)

        if model_path is not None:
            # Load weights of best trial
            model.load_weights(model_path / f'trial_{best_trial.trial_id}' / 'checkpoint.weights.h5')
        return model
    return fun


def fine_tune(X_trn, y_trn, return_model_and_tuner=False, scaler=None, hyper_model=None, project_dir=None, fold=None,
              kseed=None):
    if scaler:
        X_trn = scaler.fit_transform(X_trn)

    tuner = kt.RandomSearch(
        hyper_model,
        max_trials=100,
        objective=kt.Objective("val_auc", direction="max"),
        executions_per_trial=1,
        directory=project_dir,
        project_name=f'fold_{fold}',
        seed=kseed
    )
    tuner.search(X_trn, y_trn, epochs=500, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_auc',
                                                                                          patience=20,
                                                                                          mode='max',
                                                                                          verbose=1,
                                                                                          restore_best_weights=True),
                                                                            TensorBoard(log_dir=project_dir / f'fold_{fold}' / 'logs'),
                                                                            LambdaCallback(on_train_end=tf.keras.backend.clear_session)])

    # Path to the best trial's directory
    best_trial = get_best_trial(tuner)

    # Build model with best parameters
    model = tuner.hypermodel.build(best_trial.hyperparameters)

    # Load weights of best trial
    model.load_weights(project_dir / f'fold_{fold}' / f'trial_{best_trial.trial_id}' / 'checkpoint.weights.h5')

    if return_model_and_tuner:
        return model, tuner
    return model

def model_evaluation(mdl, x, y, sensitive_attributes=None):
    y_pred = mdl.predict(x, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Standard metrics
    accuracy = accuracy_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary, zero_division=np.nan)
    recall = recall_score(y, y_pred_binary, zero_division=np.nan)
    f1 = f1_score(y, y_pred_binary, zero_division=np.nan)
    auc = roc_auc_score(y, y_pred)

    metrics = {'Accuracy': accuracy,
               'Precision': precision,
               'Recall': recall,
               'F1 Score': f1,
               'AUC': auc}

    # Fairness metrics
    if sensitive_attributes is not None:
        for col in sensitive_attributes.columns:
            accuracy_rates = {}
            f1_rates = {}
            positive_rates = {}
            true_positive_rates = {}
            false_positive_rates = {}

            for group in sensitive_attributes[col].unique():
                group_mask = (sensitive_attributes[col] == group)
                y_true_group = y[group_mask]
                y_pred_group = y_pred_binary[group_mask]

                accuracy_rates[group] = accuracy_score(y_true_group, y_pred_group)
                f1_rates[group] = f1_score(y_true_group, y_pred_group, zero_division=np.nan)
                positive_rates[group] = y_pred_group.mean()
                true_positive_rates[group] = recall_score(y_true_group, y_pred_group, zero_division=np.nan)

                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
                false_positive_rates[group] = fp / (fp + tn) if (fp + tn) > 0 else 0

            metrics[f'{col} Demographic Parity'] = (min(positive_rates.values()) / max(positive_rates.values())) if max(positive_rates.values()) else np.nan
            metrics[f'{col} Equal Opportunity'] = (min(true_positive_rates.values()) / max(true_positive_rates.values())) if max(true_positive_rates.values()) else np.nan
            metrics[f'{col} Equalized Odds'] = np.nanmean(
                [(min(true_positive_rates.values()) / max(true_positive_rates.values())) if max(true_positive_rates.values()) else np.nan,
                 (min(false_positive_rates.values()) / max(false_positive_rates.values())) if max(false_positive_rates.values()) else np.nan]
            )
            metrics[f'{col} Accuracy Parity'] = (min(accuracy_rates.values()) / max(accuracy_rates.values())) if max(accuracy_rates.values()) else np.nan
            metrics[f'{col} F1-score Parity'] = (min(f1_rates.values()) / max(f1_rates.values())) if max(f1_rates.values()) else np.nan

        # # Intersectional group metrics
        # sensitive_groups = sensitive_attributes.apply(tuple, axis=1)
        # unique_groups = sensitive_groups.unique()

        # positive_rates = {}
        # true_positive_rates = {}
        # false_positive_rates = {}

        # for group in unique_groups:
        #     group_mask = (sensitive_groups == group)
        #     y_true_group = y[group_mask]
        #     y_pred_group = y_pred_binary[group_mask]

        #     positive_rates[group] = y_pred_group.mean()
        #     true_positive_rates[group] = recall_score(y_true_group, y_pred_group)

        #     tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
        #     false_positive_rates[group] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # metrics['Intersectional Demographic Parity'] = max(positive_rates.values()) - min(positive_rates.values())
        # metrics['Intersectional Equal Opportunity'] = max(true_positive_rates.values()) - min(true_positive_rates.values())
        # metrics['Intersectional Disparate Impact'] = min(positive_rates.values()) / max(positive_rates.values())
        # metrics['Intersectional Equalized Odds'] = max(
        #     max(true_positive_rates.values()) - min(true_positive_rates.values()),
        #     max(false_positive_rates.values()) - min(false_positive_rates.values())
        # )

    return metrics

def evavulate_per_study(mdl, x, y, s, sensitive_attr, source_study=None, adapt=None, **kwargs):
    # Evaluate the model on data from other studies
    if adapt is not None:
        metrics = {}
        for study in np.unique(s):
            if study == source_study:
                continue
            model = adapt(mdl, target_study=study, source_study=source_study, **kwargs)
            metrics[study] = model_evaluation(model, x[s == study], y[s == study], sensitive_attr[s == study])
        return metrics
    else:
        return {study: model_evaluation(mdl, x[s == study], y[s == study], sensitive_attr[s == study]) for study in np.unique(s)}

def plot_colored_matrix(df, cmap="plasma", **kwargs):
    plt.figure(figsize=(24, 8))
    if df.values.min()<0:
        cmap = 'PiYG'
        if 'vmin' not in kwargs:
            kwargs['vmin'] = -np.abs(df.values).max()
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.abs(df.values).max()
        sns.heatmap(df, annot=True, fmt=".3f", cmap=cmap, cbar=True, **kwargs)
    sns.heatmap(df, annot=True, fmt=".3f", cmap=cmap, cbar=True, **kwargs)

def monte_carlo_dropout_predictions(model, X, num_samples=50):
    """
    Perform Monte Carlo Dropout predictions.
    """
    f_model = lambda x: np.stack([model(x, training=True).numpy() for _ in range(num_samples)], axis=0)
    return f_model(X)

def calculate_uncertainty(predictions, fun=np.var):
    """
    Calculate uncertainty as the variance of predictions.
    """
    return fun(predictions, axis=0)

def evaluate_and_plot(model, X_test, y_test, metric='accuracy', num_samples=50):
    """
    Apply MC Dropout, calculate metric sorted by uncertainty, and plot.
    """
    # Get MC Dropout predictions
    mc_predictions = monte_carlo_dropout_predictions(model, X_test, num_samples).squeeze(axis=-1)
    mean_predictions = mc_predictions.mean(axis=0)
    uncertainty = calculate_uncertainty(mc_predictions)

    # Sort by uncertainty
    sorted_indices = np.argsort(uncertainty)
    sorted_uncertainty = uncertainty[sorted_indices]
    sorted_predictions = mean_predictions[sorted_indices]
    sorted_labels = y_test[sorted_indices]

    # Calculate metric cumulatively
    metrics = []
    for i in range(1, len(sorted_labels) + 1):
        if metric == 'accuracy':
            metrics.append(accuracy_score(sorted_labels[:i], np.round(sorted_predictions[:i])))
        elif metric == 'auc':
            metrics.append(roc_auc_score(sorted_labels[:i], sorted_predictions[:i]) if i > 1 else 0)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_uncertainty, metrics, label=f'Cumulative {metric.upper()}')
    plt.xlabel('Uncertainty')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} vs. Uncertainty')
    plt.legend()
    plt.grid()
    plt.show()


def adapt_to_target(hyper_model, x_trn, y_trn, target_study, model_path, kseed=None, suffix='', **kwargs):
    tuner = kt.RandomSearch(
        hyper_model,
        objective=kt.Objective("val_auc", direction="max"),
        executions_per_trial=1,
        max_trials=100,
        directory=model_path,
        project_name=target_study.replace('/', '') + suffix,
        seed=kseed
    )

    tuner.search(x_trn, y_trn, epochs=500, kseed=kseed, target_study=target_study,
                 callbacks=[EarlyStopping(monitor='val_auc',
                                          patience=20,
                                          mode='max',
                                          verbose=1,
                                          restore_best_weights=True),
                            LambdaCallback(on_train_end=tf.keras.backend.clear_session)], **kwargs)

    # Path to the best trial's directory
    best_trial = get_best_trial(tuner)

    # Build model with best parameters
    model = tuner.hypermodel.build(best_trial.hyperparameters)

    # Load weights of best trial
    model.load_weights(model_path / (target_study.replace('/', '') + suffix) / f'trial_{best_trial.trial_id}' / 'checkpoint.weights.h5')
    return model

def MonteCarloSelection(model, x, y, hp, num_samples=50, **kwargs):
    """
    Perform Monte Carlo Dropout predictions and select samples based on uncertainty.
    """
    kseed = kwargs.get('kseed', 42)
    np.random.seed(kseed)

    # Get MC Dropout predictions and calculate uncertainty
    mc_predictions = monte_carlo_dropout_predictions(model, x, num_samples).squeeze(axis=-1)
    mean_predictions, uncertainty = mc_predictions.mean(axis=0), calculate_uncertainty(mc_predictions)

    # Select top 20% most uncertain for annotation
    uncertain_idx = np.argsort(uncertainty)[-int(0.2 * len(x)):]

    # Use hyperparameter for pseudo-labeling
    pseudo_idx = np.argsort(uncertainty)[:int(hp.Float("pseudo %", min_value=0., max_value=0.5, step=0.05) * len(x))]

    # Split uncertain samples for training and validation
    x_uncertain_train, x_uncertain_val, y_uncertain_train, y_uncertain_val = train_test_split(
        x[uncertain_idx],
        y[uncertain_idx],
        test_size=0.5, random_state=kseed)

    return (
        x_uncertain_train,
        x_uncertain_val,
        x[pseudo_idx],
        np.empty((0, *x.shape[1:]), dtype=x.dtype),
        y_uncertain_train,
        y_uncertain_val,
        (mean_predictions[pseudo_idx] > 0.5).astype(int),
        np.empty((0, *y.shape[1:]), dtype=y.dtype)
    )