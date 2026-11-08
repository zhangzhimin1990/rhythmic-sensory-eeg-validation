import numpy as np

from scripts.derive_dryad_hbm_spatial_subject import channel_condition_metrics


def test_channel_metrics_detect_phase_locked_channel():
    rng = np.random.default_rng(17)
    sfreq = 128.0
    seconds = 4.0
    time = np.arange(int(sfreq * seconds)) / sfreq
    target = 8.0
    trials = []
    for _ in range(30):
        locked = np.sin(2 * np.pi * target * time) + rng.normal(0, 0.35, len(time))
        random_phase = np.sin(2 * np.pi * target * time + rng.uniform(-np.pi, np.pi))
        random_phase += rng.normal(0, 0.35, len(time))
        trials.append(np.stack([locked, random_phase]))
    epochs = np.asarray(trials)
    itpc, snr = channel_condition_metrics(epochs, sfreq, target, (128, 384))
    assert itpc.shape == (2,)
    assert snr.shape == (2,)
    assert itpc[0] > itpc[1] + 0.3
    assert snr[0] > snr[1]


def test_channel_metrics_reject_invalid_shape():
    with np.testing.assert_raises(ValueError):
        channel_condition_metrics(np.zeros((3, 100)), 100.0, 8.0, (10, 80))
