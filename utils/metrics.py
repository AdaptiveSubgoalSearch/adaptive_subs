import numpy as np


def compute_accuracy(target_labels, pred_labels):
    return np.sum(pred_labels == target_labels) / target_labels.size


def compute_accuracy_ignore_padding(target_labels, pred_labels, padding_label):
    not_ignored = pred_labels != padding_label
    return (
            np.sum((pred_labels == target_labels) * not_ignored) /
            max(np.sum(not_ignored), 1)
    )


def compute_perfect_sequence(target_labels, pred_labels):
    return np.sum(
        np.all(target_labels == pred_labels, axis=-1)
    ) / target_labels.shape[0]
