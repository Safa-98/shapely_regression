import numpy as np
import pickle

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from core.models.regression import ChoquisticRegression


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "aucpr": average_precision_score(y_true, y_proba),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


def model_size_mb(model):
    return len(pickle.dumps(model)) / (1024 ** 2)


def estimate_flops(model, X):
    n_samples, n_features = X.shape
    clf = model.named_steps["clf"]

    if isinstance(clf, LogisticRegression):
        per_sample = 2 * n_features

    elif isinstance(clf, ChoquisticRegression):
        per_sample = n_features ** 2 if clf.k_add == 2 else 2 ** min(n_features, 20)

    elif isinstance(clf, RandomForestClassifier):
        per_sample = len(clf.estimators_) * np.mean(
            [t.tree_.max_depth for t in clf.estimators_]
        )

    elif isinstance(clf, XGBClassifier):
        per_sample = clf.get_booster().num_boosted_rounds() * clf.max_depth

    elif isinstance(clf, MLPClassifier):
        per_sample = sum(
            a * b for a, b in zip(
                (n_features,) + clf.hidden_layer_sizes,
                clf.hidden_layer_sizes + (1,),
            )
        )
    else:
        per_sample = 3 * n_features

    return {
        "flops_per_sample": int(per_sample),
        "flops_total": int(per_sample * n_samples),
    }
