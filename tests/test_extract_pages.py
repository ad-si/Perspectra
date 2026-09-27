import numpy

from perspectra.page_extractor import SAMPLE_RATE, detect_clicks


rng = numpy.random.default_rng(42)


def noise(duration, amplitude):
    return amplitude * rng.standard_normal(int(duration * SAMPLE_RATE))


def click(amplitude=0.5):
    """Abrupt onset followed by a fast exponential decay"""
    t = numpy.arange(int(0.1 * SAMPLE_RATE)) / SAMPLE_RATE
    return noise(0.1, amplitude) * numpy.exp(-t / 0.008)


def rustle(duration=0.6, amplitude=0.1):
    """Noise which builds up and fades out gradually"""
    envelope = numpy.sin(numpy.linspace(0, numpy.pi, int(duration * SAMPLE_RATE)))
    return noise(duration, amplitude) * envelope


def build_signal(events, duration=10.0, background=None):
    signal = noise(duration, 0.001) if background is None else background
    for start, event in events:
        index = int(start * SAMPLE_RATE)
        signal[index : index + len(event)] += event
    return signal.astype(numpy.float32)


def assert_times_match(detected, expected):
    assert len(detected) == len(expected), detected
    for detected_time, expected_time in zip(detected, expected):
        assert abs(detected_time - expected_time) < 0.005


def test_detects_isolated_clicks():
    click_times = [1.0, 3.5, 7.2]
    signal = build_signal([(time, click()) for time in click_times])
    assert_times_match(detect_clicks(signal), click_times)


def test_ignores_rustling():
    signal = build_signal(
        [(1.0, click()), (2.0, rustle()), (3.0, rustle(amplitude=0.3)), (5.0, click())]
    )
    assert_times_match(detect_clicks(signal), [1.0, 5.0])


def test_ignores_abrupt_sustained_noise():
    signal = build_signal([(2.0, noise(1.0, 0.3)), (5.0, click())])
    assert_times_match(detect_clicks(signal), [5.0])


def test_ignores_click_during_rustling():
    signal = build_signal(
        [(2.0, rustle(duration=1.0, amplitude=0.1)), (2.5, click()), (5.0, click())]
    )
    assert_times_match(detect_clicks(signal), [5.0])


def test_ignores_quiet_ticks():
    signal = build_signal(
        [(1.0, click()), (2.0, click(amplitude=0.02)), (5.0, click()), (8.0, click())]
    )
    assert_times_match(detect_clicks(signal), [1.0, 5.0, 8.0])


def test_merges_clicks_closer_than_min_gap():
    signal = build_signal([(2.0, click()), (2.2, click(amplitude=0.3))])
    assert_times_match(detect_clicks(signal), [2.0])


def test_handles_short_signals():
    assert detect_clicks(numpy.zeros(100, dtype=numpy.float32)) == []


click_times = [1.0, 3.5, 6.0, 8.5]
rustle_times = [2.2, 4.7, 7.2]


def clicks_and_rustles(background):
    return build_signal(
        [(time, click()) for time in click_times]
        + [(time, rustle()) for time in rustle_times],
        background=background,
    )


def test_works_with_loud_background_noise():
    for amplitude in [0.001, 0.01, 0.03]:
        signal = clicks_and_rustles(noise(10.0, amplitude))
        assert_times_match(detect_clicks(signal), click_times)


def test_works_with_sudden_change_of_background_noise():
    background = noise(10.0, 1.0) * numpy.where(
        numpy.arange(int(10.0 * SAMPLE_RATE)) < 5.0 * SAMPLE_RATE, 0.001, 0.02
    )
    assert_times_match(detect_clicks(clicks_and_rustles(background)), click_times)


def test_works_with_drifting_background_noise():
    # Increases from -60 dB to -30 dB
    envelope = numpy.logspace(-3, -1.5, int(10.0 * SAMPLE_RATE))
    background = noise(10.0, 1.0) * envelope
    assert_times_match(detect_clicks(clicks_and_rustles(background)), click_times)
