"""
Extract photos of book pages from a video of someone flipping through a book.

Whenever a page is held in focus, the person makes a short clicking sound
(e.g. a tongue pop). These sounds are detected in the audio track
and the (sharpest) frame at each of these moments is saved as an image.

A click is distinguished from the rustling noise of flipping pages by:

- Quiet before: It starts abruptly out of relative silence
    (the page is held still), while rustling builds up gradually
    and is usually surrounded by more rustling.
- Decay: It dies away within a few dozen milliseconds,
    while rustling is sustained.
- Signal-to-noise ratio: It is clearly louder than the background noise.
- Level: It is about as loud as the other clicks in the recording.

To work independently of the level of the background noise,
all features are calculated on the band-passed signal
relative to the locally estimated power of the background noise.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio
import numpy
from scipy.ndimage import percentile_filter
from scipy.signal import butter, find_peaks, sosfiltfilt
from skimage.filters import laplace


SAMPLE_RATE = 48_000
HOP_DURATION = 0.001  # Resolution of the detection in seconds
ONSET_DURATION = 0.010
BEFORE_DURATION = 0.250
TAIL_START = 0.030
TAIL_DURATION = 0.050

# Frequency band containing most of the energy of clicks
# while excluding hum, rumble, and hiss
BAND = (500, 10_000)

# The background noise is estimated as the power of the quieter parts
# of the surrounding time window. A short window lets the estimate
# follow changes of the background noise.
NOISE_FRAME_DURATION = 0.050
NOISE_WINDOW_DURATION = 1.0
NOISE_PERCENTILE = 30

MAX_DB = 99.0


def read_audio(video_path, sample_rate=SAMPLE_RATE):
    """Decode the first audio stream of the video as mono float samples."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", str(video_path),
            "-map", "0:a:0",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return numpy.frombuffer(result.stdout, dtype=numpy.float32)


def power_ratio_db(numerator, denominator):
    """
    Ratio in dB, where non-positive values (e.g. after subtracting the noise)
    are treated as a negligibly small power.
    """
    tiny = 1e-20
    ratio = 10 * numpy.log10(
        numpy.maximum(numerator, tiny) / numpy.maximum(denominator, tiny)
    )
    return numpy.clip(ratio, -MAX_DB, MAX_DB)


def estimate_noise_power(samples, sample_rate):
    """
    Estimate the power of the background noise over time
    as a low percentile of the power in the surrounding time window.
    Returns the times of the estimates and the estimates.
    """
    frame_length = int(NOISE_FRAME_DURATION * sample_rate)
    num_frames = len(samples) // frame_length
    frame_power = (
        (samples[: num_frames * frame_length] ** 2)
        .reshape(num_frames, frame_length)
        .mean(axis=1)
    )
    window_frames = min(
        round(NOISE_WINDOW_DURATION / NOISE_FRAME_DURATION), num_frames
    )
    noise_power = percentile_filter(
        frame_power, NOISE_PERCENTILE, size=window_frames, mode="nearest"
    )
    frame_times = (numpy.arange(num_frames) + 0.5) * NOISE_FRAME_DURATION
    # Silence would lead to divisions by zero
    return frame_times, numpy.maximum(noise_power, 1e-20)


def analyze_audio(samples, sample_rate=SAMPLE_RATE):
    """
    Calculate the click features for every hop of the audio signal.
    Returns the times (start of the onset window) and the features in dB.
    """
    samples = numpy.asarray(samples, dtype=numpy.float64)
    hop = int(HOP_DURATION * sample_rate)
    before_length = int(BEFORE_DURATION * sample_rate)
    tail_end_length = int((TAIL_START + TAIL_DURATION) * sample_rate)
    noise_frame_length = int(NOISE_FRAME_DURATION * sample_rate)

    if len(samples) < max(
        before_length + tail_end_length, noise_frame_length, 64
    ):
        return numpy.empty(0), {}

    band = butter(4, BAND, btype="bandpass", fs=sample_rate, output="sos")
    samples = sosfiltfilt(band, samples)
    energy_sum = numpy.concatenate(([0.0], numpy.cumsum(samples ** 2)))

    def power(starts, duration):
        length = int(duration * sample_rate)
        return (energy_sum[starts + length] - energy_sum[starts]) / length

    # Only analyze positions where all windows lie completely inside the signal
    positions = numpy.arange(
        before_length,
        len(samples) - tail_end_length + 1,
        hop,
    )
    times = positions / sample_rate

    noise_times, noise_power = estimate_noise_power(samples, sample_rate)
    noise = numpy.interp(times, noise_times, noise_power)

    onset = power(positions, ONSET_DURATION)
    before = power(positions - before_length, BEFORE_DURATION)
    tail = power(positions + int(TAIL_START * sample_rate), TAIL_DURATION)

    signal_onset = onset - noise

    return times, {
        "level": power_ratio_db(onset, 1.0),
        "snr": power_ratio_db(signal_onset, noise),
        # Level of the signal before the click relative to the click
        "contrast": power_ratio_db(signal_onset, before - noise),
        # Level of the time before the click relative to the noise
        "before": power_ratio_db(before, noise),
        "decay": power_ratio_db(signal_onset, tail - noise),
    }


def detect_clicks(
    samples,
    sample_rate=SAMPLE_RATE,
    min_contrast=28.0,
    min_decay=12.0,
    min_snr=15.0,
    max_level_spread=15.0,
    min_gap=0.5,
    debug=False,
):
    """
    Return the times (in seconds) of all clicking sounds.
    """
    times, features = analyze_audio(samples, sample_rate)
    if len(times) == 0:
        return []

    level = features["level"]
    distance = max(1, round(min_gap / HOP_DURATION))

    if debug:
        # Show all loud sounds to help with tuning the thresholds
        peaks, _ = find_peaks(
            numpy.where(features["snr"] >= 10, level, -numpy.inf),
            height=-MAX_DB,
            distance=distance,
        )
        print("time      level   snr  contrast  before  decay")
        onset_hops = round(0.02 / HOP_DURATION)
        for loudest in peaks:
            # Show the features at the onset (first position close to the maximum)
            start = max(0, loudest - onset_hops)
            peak = start + numpy.argmax(level[start : loudest + 1] >= level[loudest] - 6)
            print(
                f"{times[peak]:7.3f}s"
                f"  {level[peak]:5.1f}"
                f"  {features['snr'][peak]:4.1f}"
                f"  {features['contrast'][peak]:8.1f}"
                f"  {features['before'][peak]:6.1f}"
                f"  {features['decay'][peak]:5.1f}"
            )
        print()

    # The time before a click is quiet if it's much quieter than the click
    # or if it doesn't contain anything louder than the background noise
    is_quiet_before = (features["contrast"] >= min_contrast) | (
        features["before"] <= 0
    )

    # Search for the loudest position of each click which fulfills all criteria.
    # Masking out the other positions ensures that they can't suppress
    # real clicks in the peak search.
    is_click = (
        is_quiet_before
        & (features["decay"] >= min_decay)
        & (features["snr"] >= min_snr)
    )
    peaks, _ = find_peaks(
        numpy.where(is_click, level, -numpy.inf),
        height=-MAX_DB,
        distance=distance,
    )

    # Discard sounds which are much quieter than the typical click
    if len(peaks) > 0:
        median_level = numpy.median(level[peaks])
        peaks = peaks[level[peaks] >= median_level - max_level_spread]

    return [float(times[peak]) for peak in peaks]


def sharpness(image):
    """Variance of the Laplacian. Higher values mean less blur."""
    return laplace(image.mean(axis=2)).var()


def extract_sharpest_frame(video_path, time, window, tmp_dir):
    """
    Extract all frames in the window centered around the time
    and return the path of the sharpest one.
    """
    start = max(0.0, time - window / 2)
    frame_limit = ["-t", str(window)] if window > 0 else ["-frames:v", "1"]
    subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            *frame_limit,
            "-map", "0:v:0",
            "-pix_fmt", "rgb24",
            str(tmp_dir / "frame-%04d.png"),
        ],
        check=True,
    )
    frame_paths = sorted(tmp_dir.glob("frame-*.png"))
    if not frame_paths:
        raise RuntimeError(f"Could not extract a frame at {time:.3f} s")

    return max(frame_paths, key=lambda path: sharpness(imageio.imread(path)))


def extract_pages(
    input_video_path,
    output_directory=None,
    frame_window=0.2,
    min_contrast=28.0,
    min_decay=12.0,
    min_snr=15.0,
    max_level_spread=15.0,
    min_gap=0.5,
    debug=False,
    **kwargs,
):
    video_path = Path(input_video_path)
    output_dir = (
        Path(output_directory)
        if output_directory
        else video_path.with_name(f"{video_path.stem}_pages")
    )

    click_times = detect_clicks(
        read_audio(video_path),
        min_contrast=min_contrast,
        min_decay=min_decay,
        min_snr=min_snr,
        max_level_spread=max_level_spread,
        min_gap=min_gap,
        debug=debug,
    )

    if not click_times:
        print("No clicks detected")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    num_digits = max(3, len(str(len(click_times))))
    print(output_dir.resolve())

    for index, time in enumerate(click_times, start=1):
        output_path = output_dir / f"page-{index:0{num_digits}d}.png"
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = extract_sharpest_frame(
                video_path, time, frame_window, Path(tmp_dir)
            )
            shutil.move(frame_path, output_path)
        print(f"{time:8.3f} s -> {output_path.name}")
