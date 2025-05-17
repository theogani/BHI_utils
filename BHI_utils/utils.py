from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback
import keras_tuner as kt
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def fine_tune(X_trn, y_trn, retrain=True, scaler=None, hyper_model=None, project_dir=None, fold=None, kseed=None):
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

    if retrain:
        # Path to the best trial's directory
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

        # Build model with best parameters
        model = tuner.hypermodel.build(best_trial.hyperparameters)

        # Load weights of best trial
        model.load_weights(project_dir / f'fold_{fold}' / f'trial_{best_trial.trial_id}' / 'checkpoint.weights.h5')
        return model
    return None

def model_evaluation(mdl, x, y, sensitive_attributes=None):
    y_pred = mdl.predict(x)
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

def evavulate_per_study(mdl, x, y, s, sensitive_attr, adapt=None, **kwargs):
    # Evaluate the model on data from other studies
    if adapt is not None:
        metrics = {}
        original_weights = mdl.get_weights()
        for study in np.unique(s):
            mdl.set_weights(original_weights)
            model = adapt(mdl, target_study=study, **kwargs)
            metrics[study] = model_evaluation(model, x[s == study], y[s == study], sensitive_attr[s == study])
        return metrics
    else:
        return {study: model_evaluation(mdl, x[s == study], y[s == study], sensitive_attr[s == study]) for study in np.unique(s)}

def plot_colored_matrix(df):
    plt.figure(figsize=(24, 8))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="plasma", cbar=True)

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