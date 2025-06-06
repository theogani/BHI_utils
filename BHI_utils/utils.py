from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback
import keras_tuner as kt
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def get_best_trial(tuner):
    # Get all completed trials
    all_trials = [t for t in tuner.oracle.trials.values() if t.status == 'COMPLETED']

    # Find the best score
    best_score = np.max([t.score for t in all_trials])

    # Filter trials with the best score
    best_trials = [t for t in all_trials if t.score == best_score]

    # Pick a deterministic one (e.g., by trial_id)
    return sorted(best_trials, key=lambda t: t.trial_id)[0]


def load_best_model(tuner, model_path=None):
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


def fine_tune(X_trn, y_trn, return_model_and_tuner=False, scaler=None, hyper_model=None, project_dir=None,
              project_name=None, restore_best_weights=True, objective="val_auc", **kwargs):
    if scaler:
        X_trn = scaler.fit_transform(X_trn)

    tuner = kt.RandomSearch(
        hyper_model,
        max_trials=100,
        objective=kt.Objective(objective, direction="max"),
        executions_per_trial=1,
        directory=project_dir,
        project_name=project_name,
        seed=kwargs['kseed']
    )
    tuner.search(X_trn, y_trn, epochs=500, **kwargs)

    # Path to the best trial's directory
    best_trial = get_best_trial(tuner)

    # Build model with best parameters
    model = tuner.hypermodel.build(best_trial.hyperparameters)

    if restore_best_weights:
        # Load weights of best trial
        model.load_weights(project_dir / project_name / f'trial_{best_trial.trial_id}' / 'checkpoint.weights.h5')

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

            metrics[f'{col} Demog Par'] = (min(positive_rates.values()) / max(positive_rates.values())) if max(positive_rates.values()) else np.nan
            metrics[f'{col} Equal Opp'] = (min(true_positive_rates.values()) / max(true_positive_rates.values())) if max(true_positive_rates.values()) else np.nan
            metrics[f'{col} Equal Odds'] = np.nanmean(
                [(min(true_positive_rates.values()) / max(true_positive_rates.values())) if max(true_positive_rates.values()) else np.nan,
                 (min(false_positive_rates.values()) / max(false_positive_rates.values())) if max(false_positive_rates.values()) else np.nan]
            )
            metrics[f'{col} Acc Par'] = (min(accuracy_rates.values()) / max(accuracy_rates.values())) if max(accuracy_rates.values()) else np.nan
            metrics[f'{col} F1 Par'] = (min(f1_rates.values()) / max(f1_rates.values())) if max(f1_rates.values()) else np.nan
            metrics[f'{col} fair perf t-off'] = (2*metrics[f'{col} Equal Odds']*metrics['F1 Score'])/(metrics[f'{col} Equal Odds'] + metrics['F1 Score'])

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
    if np.nanmin(df.values)<0:
        cmap = sns.diverging_palette(10, 240, as_cmap=True)
        if 'vmin' not in kwargs:
            kwargs['vmin'] = -np.nanmax(np.abs(df.values))
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.nanmax(np.abs(df.values))
        ax = sns.heatmap(df, annot=True, fmt=".3f", cmap=cmap, cbar=True, **kwargs)
        plt.xticks(rotation=45)
        plt.show()
        return ax
    ax = sns.heatmap(df, annot=True, fmt=".3f", cmap=cmap, cbar=True, **kwargs)
    plt.xticks(rotation=45)
    plt.show()
    return ax

def monte_carlo_dropout_predictions(model, X, num_samples=50):
    """
    Perform Monte Carlo Dropout predictions.
    """
    return np.stack([model(X, training=True).numpy().squeeze() for _ in range(num_samples)], axis=0)

def calculate_uncertainty(predictions, metric_name):
    """
    Calculate uncertainty as the variance of predictions.
    """
    if metric_name == 'var':
        return np.var(predictions, axis=0)
    elif metric_name == 'std':
        return np.std(predictions, axis=0)
    elif metric_name == 'entropy':
        mean_preds = predictions.mean(axis=0)
        return -mean_preds * np.log(mean_preds + 1e-8) - (1 - mean_preds) * np.log(1 - mean_preds + 1e-8)
    elif metric_name == 'total_variance':
        mean = predictions.mean(axis=0)
        # Compute the squared differences from the mean
        squared_diffs = (predictions - mean) ** 2
        return np.sum(squared_diffs, axis=0) / len(predictions)
    else:
        raise ValueError("Invalid uncertainty metric")

def evaluate_and_plot(model, X_test, y_test, metric='accuracy', num_samples=50):
    """
    Apply MC Dropout, calculate metric sorted by uncertainty, and plot.
    """
    # Get MC Dropout predictions
    mc_predictions = monte_carlo_dropout_predictions(model, X_test, num_samples)
    mean_predictions = mc_predictions.mean(axis=0)
    uncertainty = calculate_uncertainty(mc_predictions, 'var')

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


def adapt_to_target(hyper_model, x_trn, y_trn, target_study, model_path, suffix='',
                    fine_tune_on_source=None, **kwargs):
    if fine_tune_on_source is not None:
        # Get the source study data
        source_ids = kwargs['studies'] == kwargs['source_study']
        x_source, y_source = x_trn[source_ids], y_trn[source_ids]

        hyper_model = fine_tune_on_source(hyper_model, x_source, y_source, model_path, **kwargs)


    return fine_tune(x_trn, y_trn, hyper_model=hyper_model, project_dir=model_path,
                     project_name=target_study.replace('/', '') + suffix, target_study=target_study,
                     callbacks=[EarlyStopping(monitor='val_auc',
                                              patience=20,
                                              mode='max',
                                              verbose=1,
                                              restore_best_weights=True),
                                LambdaCallback(on_train_end=tf.keras.backend.clear_session)], **kwargs)

def MonteCarloSelection(model, x, y, hp, num_samples, uncertainty_metric, **kwargs):
    """
    Perform Monte Carlo Dropout predictions and select samples based on uncertainty.
    """
    np.random.seed(kwargs['kseed'])

    # Split uncertain samples for training and validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=int(0.1 * len(x)), random_state=kwargs['kseed'])

    # Get MC Dropout predictions and calculate uncertainty
    mc_predictions = monte_carlo_dropout_predictions(model, x_train, num_samples)
    mean_predictions, uncertainty = mc_predictions.mean(axis=0), calculate_uncertainty(mc_predictions, uncertainty_metric)

    # Select top 20% most uncertain for annotation
    ascending_uncertainty_idx = np.argsort(uncertainty)

    uncertain_idx = ascending_uncertainty_idx[-int(0.1 * len(x)):]

    # Use hyperparameter for pseudo-labeling
    thrs_source = hp.Choice('threshold_source', ['source', 'validation'])
    if thrs_source=='source':
        pseudo_idx = np.where(uncertainty < hp.get('uncertainty_threshold'))[0]
    elif thrs_source=='validation':
        mc_preds = monte_carlo_dropout_predictions(model, x_val, num_samples=num_samples)
        incorrect = ((mc_preds.mean(axis=0) > 0.5).astype(int) != y_val).astype(int)

        thrs = find_best_threshold(incorrect, calculate_uncertainty(mc_preds, uncertainty_metric))
        pseudo_idx = np.where(uncertainty < thrs)[0]
        hp.Fixed('uncertainty_threshold', thrs)
    pseudo_idx = np.setdiff1d(pseudo_idx, uncertain_idx)
    # uncertain_idx = np.where(uncertainty < hp.get('uncertainty_threshold'))[0]
    hp.Fixed('number_of_selected', len(uncertain_idx)/ len(x))

    return (
        x_train[uncertain_idx],
        x_val,
        x_train[pseudo_idx],
        np.empty((0, *x.shape[1:]), dtype=x.dtype),
        y_train[uncertain_idx],
        y_val,
        (mean_predictions[pseudo_idx] > 0.5).astype(int),
        np.empty((0, *y.shape[1:]), dtype=y.dtype)
    )

def fine_tune_mc_dropout(next_hyper_model, sel):
    def fun(hyper_model, x_source, y_source, model_path, **kwargs):
        """
        Fine-tune the model using Monte Carlo Dropout.
        """
        np.random.seed(kwargs['kseed'])

        _, tuner = fine_tune(x_source, y_source, project_dir=model_path, project_name="mc_dropout_fine_tune",
                             hyper_model=hyper_model, restore_best_weights=False, return_model_and_tuner=True,
                             kseed=kwargs['kseed'])

        def select_fn(model, x, y, hp, **kwargs):
            return sel(model, x, y, hp, **get_best_trial(tuner).hyperparameters.values, **kwargs)

        return next_hyper_model(hyper_model.model_fn, select_fn, uncertainty_threshold=get_best_trial(tuner).hyperparameters.values['uncertainty_threshold'])
    return fun

def find_best_threshold(incorrect, uncertainty):
    fpr, tpr, thresholds = roc_curve(incorrect, uncertainty)
    youden_index = tpr - fpr
    return float(thresholds[np.argmax(youden_index)])

def MonteCarlo_Representation_Selection(model, x, y, hp, num_samples, uncertainty_metric, **kwargs):
    """
    Perform Monte Carlo Dropout predictions and select samples based on uncertainty and representativeness.
    """
    np.random.seed(kwargs['kseed'])

    # Split uncertain samples for training and validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=int(0.1 * len(x)), random_state=kwargs['kseed'])

    # Get MC Dropout predictions and calculate uncertainty
    mc_predictions = monte_carlo_dropout_predictions(model, x_train, num_samples)
    mean_predictions, uncertainty = mc_predictions.mean(axis=0), calculate_uncertainty(mc_predictions, uncertainty_metric)

    # Use hyperparameter for pseudo-labeling
    thrs_source = hp.Choice('threshold_source', ['source', 'validation'])
    if thrs_source=='source':
        pseudo_idx = np.where(uncertainty < hp.get('uncertainty_threshold'))[0]
    elif thrs_source=='validation':
        mc_preds = monte_carlo_dropout_predictions(model, x_val, num_samples=num_samples)
        incorrect = ((mc_preds.mean(axis=0) > 0.5).astype(int) != y_val).astype(int)

        thrs = find_best_threshold(incorrect, calculate_uncertainty(mc_preds, uncertainty_metric))
        pseudo_idx = np.where(uncertainty < thrs)[0]
        hp.Fixed('uncertainty_threshold', thrs)

    classes = np.array([0, 1])
    if np.all(np.isin(classes, np.unique((mean_predictions[pseudo_idx] > 0.5).astype(int)))):
        x_cluster, y_cluster = x_train[pseudo_idx], (mean_predictions[pseudo_idx] > 0.5).astype(int)
    elif np.all(np.isin(classes, np.unique((mean_predictions > 0.5).astype(int)))):
        x_cluster, y_cluster = x_train, (mean_predictions > 0.5).astype(int)

    # Per-class clustering
    cluster_centers = []
    n_clusters = hp.Int('num_representatives_per_class', min_value=1, max_value=3, step=1)
    for cls in classes:
        cls_idx = np.where(y_cluster == cls)[0]
        kmeans = KMeans(n_clusters=min(n_clusters, len(cls_idx)), random_state=kwargs['kseed'])
        kmeans.fit(x_cluster[cls_idx])
        cluster_centers.extend(kmeans.cluster_centers_)
    cluster_centers = np.array(cluster_centers)

    # Score representativeness: for each x_train, compute min distance to its predicted class cluster centers
    representativeness = np.zeros(len(x_train))
    for i, sample in enumerate(x_train):
        dists = cdist([sample], cluster_centers)
        representativeness[i] = np.min(dists)

    # Normalize scores to [0, 1]
    uncertainty_norm = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-8)
    representativeness_norm = (representativeness - np.min(representativeness)) / (np.max(representativeness) - np.min(representativeness) + 1e-8)

    # Combine with a weight alpha
    alpha = hp.Float('uncertainty_weight', min_value=0.4, max_value=0.6, step=0.1, default=0.5)
    score = alpha * uncertainty_norm + (1 - alpha) * (1 - representativeness_norm)  # (1 - rep) if lower is better

    # Select top 10% most uncertain for annotation (as before)
    ascending_score_idx = np.argsort(score)
    uncertain_idx = ascending_score_idx[:int(0.1 * len(x))]
    pseudo_idx = np.setdiff1d(pseudo_idx, uncertain_idx)

    return (x_train[uncertain_idx],
            x_val,
            x_train[pseudo_idx],
            np.empty((0, *x.shape[1:]), dtype=x.dtype),
            y_train[uncertain_idx],
            y_val,
            (mean_predictions[pseudo_idx] > 0.5).astype(int),
            np.empty((0, *y.shape[1:]), dtype=y.dtype))

def Representation_Selection(model, x, y, hp, **kwargs):
    """
    Perform representation selection based on uncertainty and clustering.
    """
    np.random.seed(kwargs['kseed'])

    # Split uncertain samples for training and validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=int(0.1 * len(x)), random_state=kwargs['kseed'])

    y_pred = model.predict(x_train, verbose=0)

    x_cluster, y_cluster = x_train, (y_pred > 0.5).astype(int)

    # Per-class clustering
    cluster_centers = []
    n_clusters = hp.Int('num_representatives_per_class', min_value=1, max_value=3, step=1)
    classes = np.array([0, 1])
    for cls in classes:
        cls_idx = np.where(y_cluster == cls)[0]
        kmeans = KMeans(n_clusters=min(n_clusters, len(cls_idx)), random_state=kwargs['kseed'])
        kmeans.fit(x_cluster[cls_idx])
        cluster_centers.extend(kmeans.cluster_centers_)
    cluster_centers = np.array(cluster_centers)

    # Score representativeness: for each x_train, compute min distance to its predicted class cluster centers
    representativeness = np.zeros(len(x_train))
    for i, sample in enumerate(x_train):
        dists = cdist([sample], cluster_centers)
        representativeness[i] = np.min(dists)

    # Normalize scores to [0, 1]
    representativeness_norm = (representativeness - np.min(representativeness)) / (np.max(representativeness) - np.min(representativeness) + 1e-8)

    score = (1 - representativeness_norm)

    # Select top 10% most uncertain for annotation (as before)
    ascending_score_idx = np.argsort(score)
    select_idx = ascending_score_idx[:int(0.1 * len(x))]

    return (x_train[select_idx],
            x_val,
            np.empty((0, *x.shape[1:]), dtype=x.dtype),
            np.empty((0, *x.shape[1:]), dtype=x.dtype),
            y_train[select_idx],
            y_val,
            np.empty((0, *y.shape[1:]), dtype=x.dtype),
            np.empty((0, *y.shape[1:]), dtype=y.dtype))

def mmd(X, Y, kernel='rbf', gamma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two samples: X and Y.

    Args:
        X (np.ndarray): Samples from distribution P, shape (n_samples_X, n_features)
        Y (np.ndarray): Samples from distribution Q, shape (n_samples_Y, n_features)
        kernel (str): Kernel type for pairwise_kernels (default: 'rbf')
        gamma (float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'

    Returns:
        float: MMD statistic
    """
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)

    mmd_stat = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd_stat