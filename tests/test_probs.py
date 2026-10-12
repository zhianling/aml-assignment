import numpy as np

from birdcls.probs import log_probs, nll, softmax, tta_probs


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1000.0, 0.0], [0.0, 0.0]]))  # large logits must not overflow
    np.testing.assert_allclose(p.sum(1), 1.0)
    np.testing.assert_allclose(p[1], [0.5, 0.5])


def test_softmax_keeps_float32():
    assert softmax(np.zeros((2, 3), dtype=np.float32)).dtype == np.float32


def test_tta_probs_averages_the_views_probabilities():
    center, hflip = np.array([[2.0, 0.0]]), np.array([[0.0, 2.0]])
    np.testing.assert_allclose(tta_probs(center, hflip), (softmax(center) + softmax(hflip)) / 2)
    np.testing.assert_allclose(tta_probs(center), softmax(center))


def test_log_probs_stays_finite_at_zero():
    assert np.isfinite(log_probs(np.array([[0.0, 1.0]]))).all()


def test_nll_is_the_mean_log_loss_of_the_true_labels():
    probs = np.array([[0.5, 0.5], [0.25, 0.75]])
    assert np.isclose(nll(probs, np.array([0, 1])), -(np.log(0.5) + np.log(0.75)) / 2)
