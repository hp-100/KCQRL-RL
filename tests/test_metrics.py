import math
from evaluation.metrics import auc_score, accuracy_score, nll_score, brier_score

def test_auc_single_class_nan():
    assert math.isnan(auc_score([1,1,1], [0.2,0.4,0.9]))

def test_basic_metrics():
    assert accuracy_score([0,1], [0.1,0.9]) == 1.0
    assert nll_score([0,1], [0.1,0.9]) < 0.2
    assert brier_score([0,1], [0.1,0.9]) < 0.02
