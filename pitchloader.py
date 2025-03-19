import os
import csv
import shutil
import hashlib
from pathlib import Path

import crepe
import oyaml
import numpy as np
from fastdtw import fastdtw
from scipy.io import wavfile
from scipy.stats import zscore
from yamlcore import CoreLoader
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from platformdirs import user_cache_dir
from librosa import midi_to_hz, hz_to_midi
from scipy.ndimage import gaussian_filter1d


APP_NAME = "PitchLoader"
APP_AUTHOR = "newcomer00"
APP_VERSION = "0.1.0"
CACHE_DIR = user_cache_dir(APP_NAME, APP_AUTHOR, version=APP_VERSION)


def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file.

    This is useful for caching, ensuring that identical files are recognized.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: SHA-256 hash of the file contents.
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def clear_cache():
    """Clear the cache directory.

    Removes all cached pitch extraction data.
    """
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)


def extract_wavfile_frequency(file_path, use_cache=True):
    """Extract pitch frequency from a WAV file using Crepe.

    This function processes an audio file to extract pitch information using the
    CREPE algorithm. It supports caching to improve performance when processing
    the same file multiple times.

    Args:
        file_path (str): Path to the WAV file.
        use_cache (bool, optional): Whether to use cached data if available. Defaults to True.

    Returns:
        tuple: (time, frequency, confidence), where:
            - time (list of float): Time points in seconds.
            - frequency (list of float): Detected pitch frequencies in Hz.
            - confidence (list of float): Confidence values for the detected pitches.
    """
    time = []
    frequency = []
    confidence = []
    # Try reading data from cache
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        wav_hash = calculate_file_hash(file_path)

        cache_path = Path(CACHE_DIR) / f'{wav_hash}.csv'
        if cache_path.is_file():
            print(f"Using cache file `{cache_path}`")
            with open(cache_path, "r", newline='') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    time.append(float(row[0]))
                    frequency.append(float(row[1]))
                    confidence.append(float(row[2]))

    # If cache is unavailable
    if not all([time, frequency, confidence]):
        sr, audio = wavfile.read(file_path)
        time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=True)

        # Save data to cache
        if use_cache:
            with open(cache_path, mode='w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Time (s)', 'Frequency (Hz)', 'Confidence'])
                for t, f, c in zip(time, frequency, confidence):
                    writer.writerow([t, f, c])
            print(f"Saved to cache file `{cache_path}`")

    return time, frequency, confidence


def time_to_ticks(time, tempo, ppqn=480):
    """Convert time in seconds to MIDI ticks.

    Args:
        time (float or array-like): Time values in seconds.
        tempo (float): Tempo in beats per minute (BPM).
        ppqn (int, optional): Pulses per quarter note (MIDI resolution). Defaults to 480.

    Returns:
        numpy.ndarray: Corresponding MIDI tick values.
    """
    return (np.array(time) * tempo * ppqn) / 60


def ticks_to_time(ticks, tempo, ppqn=480):
    """Convert MIDI ticks to time in seconds.

    Args:
        ticks (int or array-like): MIDI tick values.
        tempo (float): Tempo in beats per minute (BPM).
        ppqn (int, optional): Pulses per quarter note. Defaults to 480.

    Returns:
        numpy.ndarray: Corresponding time values in seconds.
    """
    return (np.array(ticks) * 60) / (tempo * ppqn)


def load_ustx(ustx_path):
    """Load a USTX (Vocal Synth format) file as a dictionary.

    Uses YAML parsing to extract the structure of a USTX file.

    Args:
        ustx_path (str): Path to the USTX file.

    Returns:
        dict: Parsed USTX data.
    """
    with open(ustx_path, 'r', encoding='utf-8-sig') as u:
        ustx_str = u.read()
    # Use yamlcore.CoreLoader to support YAML1.2
    ustx_dict = oyaml.load(ustx_str, CoreLoader)
    return ustx_dict


def save_ustx(ustx_dict, ustx_path):
    """Save a USTX dictionary to a file, preserving order.

    Args:
        ustx_dict (dict): USTX data to save.
        ustx_path (str): Path to save the USTX file.
    """
    # Use oyaml to keep original order of USTX items
    output_str = oyaml.dump(ustx_dict, Dumper=oyaml.Dumper)
    with open(ustx_path, 'w+', encoding='utf-8-sig') as o:
        o.write(output_str)


def edit_ustx_pitch(ustx_dict, vocal_track_number, tick_seq, pitch_seq):
    """Edit pitch data in a USTX file for a given vocal track.

    This modifies the pitch curve data within a specified vocal track.

    Args:
        ustx_dict (dict): USTX data structure.
        vocal_track_number (int): Index of the vocal track to edit (1-based).
        tick_seq (numpy.ndarray): Sequence of MIDI tick values.
        pitch_seq (numpy.ndarray): Sequence of pitch values (Hz).
    """
    track_idx = vocal_track_number - 1  # track index starts from 0
    track = ustx_dict['voice_parts'][track_idx]
    if 'curves' not in track.keys():
        track['curves'] = []

    curves = track['curves']
    pitd = None
    for c in curves:
        if c['abbr'] == 'pitd':
            pitd = c
            break
    if pitd is None:
        curves.append({'xs': [], 'ys': [], 'abbr': 'pitd'})
        pitd = curves[-1]

    mask = ~np.isnan(pitch_seq)
    pitd['xs'] = tick_seq[mask].tolist()
    pitd['ys'] = np.round(pitch_seq[mask]).astype(int).tolist()


def get_wav_pitch_tick(wav_path, tempo=120, ppqn=480,
                       confidence_threshold=0.8, confidence_filter_size=9):
    """Extract pitch from a WAV file and convert it to MIDI tick values.

    This function filters out low-confidence pitch detections and converts the
    detected pitch frequencies into MIDI tick values.

    Args:
        wav_path (str): Path to the WAV file.
        tempo (float): Tempo in beats per minute (BPM).
        ppqn (int, optional): Pulses per quarter note (MIDI resolution). Defaults to 480.
        confidence_threshold (float, optional): Minimum confidence level for pitch detection. Defaults to 0.8.
        confidence_filter_size (int, optional): Size of the median filter for confidence values. Defaults to 9.

    Returns:
        tuple: (wav_pitch, wav_tick), where:
            - wav_pitch (numpy.ndarray): Extracted pitch frequencies in Hz.
            - wav_tick (numpy.ndarray): Corresponding MIDI tick values.
    """
    time, frequency, confidence = extract_wavfile_frequency(wav_path, True)

    # filter out low confidence frequency
    mask = medfilt(np.array(confidence),
                   kernel_size=confidence_filter_size) < confidence_threshold
    (wav_pitch := np.array(frequency))[mask] = np.nan
    wav_tick = np.round(time_to_ticks(np.array(time), tempo, ppqn)).astype(int)
    # each tick stamp should be unique
    assert len(wav_tick) == len(np.unique(wav_tick))
    return wav_pitch, wav_tick


def gaussian_filter1d_with_nan(seq, sigma, **kwargs):
    """Apply a 1D Gaussian filter to a sequence while handling NaN values.

    This function applies Gaussian smoothing to a sequence, ignoring NaN values to prevent distortion.

    Args:
        seq (numpy.ndarray): Input sequence with possible NaN values.
        sigma (float): Standard deviation for Gaussian kernel.
        **kwargs: Additional arguments for scipy.ndimage.gaussian_filter1d.

    Returns:
        numpy.ndarray: Smoothed sequence with NaN handling.
    """
    # https://stackoverflow.com/a/36307291
    (v := seq.copy())[np.isnan(seq)] = 0
    vv = gaussian_filter1d(v, sigma, **kwargs)
    (w := np.ones(len(seq)))[np.isnan(seq)] = 0
    ww = gaussian_filter1d(w, sigma, **kwargs)
    with np.errstate(invalid='ignore'):
        return np.divide(vv, ww)


def get_vocal_pitch_tick(ustx_dict, vocal_track_number=1,
                         smoothness=3):
    """Extract the expected pitch sequence from a USTX file.

    This function retrieves the pitch curve based on note information in the USTX format.
    The extracted pitch sequence can be optionally smoothed using a Gaussian filter.

    Args:
        ustx_dict (dict): Parsed USTX data.
        vocal_track_number (int, optional): The track number to extract pitch from (1-based index). Defaults to 1.
        smoothness (int, optional): Smoothing factor for the extracted pitch. Defaults to 3.

    Returns:
        tuple: (vocal_pitch, vocal_tick), where:
            - vocal_pitch (numpy.ndarray): Extracted pitch frequencies in Hz.
            - vocal_tick (numpy.ndarray): Corresponding MIDI tick values.
    """
    track_idx = vocal_track_number - 1  # track index starts from 0
    notes = ustx_dict['voice_parts'][track_idx]['notes']

    vocal_tick_start = notes[0]['position']
    vocal_tick_end = notes[-1]['position'] + notes[-1]['duration']

    vocal_tick = np.arange(vocal_tick_start, vocal_tick_end)
    vocal_pitch = np.full(len(vocal_tick), np.nan)
    for note in notes:
        start_idx = note['position'] - vocal_tick_start
        end_idx = start_idx + note['duration']
        frq = midi_to_hz(note['tone'])
        vocal_pitch[start_idx:end_idx] = frq

    vocal_pitch = gaussian_filter1d_with_nan(vocal_pitch, sigma=smoothness)
    return vocal_pitch, vocal_tick


def sequence_interval_intersection(a, b):
    """Find the intersection of two sequences while preserving order.

    This function merges two numerical sequences and extracts values that are common
    to both, ensuring the order is preserved.

    Args:
        a (numpy.ndarray or list): First sequence of numerical values.
        b (numpy.ndarray or list): Second sequence of numerical values.

    Returns:
        list: A sorted list containing the intersection of `a` and `b`.
    """
    # a = [0, 1, 2, 3]; b = [1, 1.1, 2, 4]
    # result = [1, 1.1, 2, 3]
    min_val = max(min(a), min(b))  # Highest lower bound
    max_val = min(max(a), max(b))  # Lowest upper bound
    return [x for x in np.unique(np.concatenate((a, b)))
            if min_val <= x <= max_val]


def unify_sequence_time(seq1_time, seq1_val, seq2_time, seq2_val):
    """Unify time sequences of two pitch series via interpolation.

    Ensures that both sequences are aligned to the same time points by interpolating missing values.

    Args:
        seq1_time (numpy.ndarray): Time values for the first sequence.
        seq1_val (numpy.ndarray): Pitch values for the first sequence.
        seq2_time (numpy.ndarray): Time values for the second sequence.
        seq2_val (numpy.ndarray): Pitch values for the second sequence.

    Returns:
        tuple: (unified_time, unified_seq1, unified_seq2), where:
            - unified_time (numpy.ndarray): Unified time values.
            - unified_seq1 (numpy.ndarray): Interpolated pitch values for sequence 1.
            - unified_seq2 (numpy.ndarray): Interpolated pitch values for sequence 2.
    """
    unified_time = np.array(
        sequence_interval_intersection(seq1_time, seq2_time))
    interp_seq1 = interp1d(seq1_time, seq1_val, kind='linear',
                           fill_value="extrapolate")
    interp_seq2 = interp1d(seq2_time, seq2_val, kind='linear',
                           fill_value="extrapolate")
    return unified_time, interp_seq1(unified_time), interp_seq2(unified_time)


def align_sequence_time(time, query, reference, align_radius=1):
    """Align the time axis of a query sequence to match a reference sequence using FastDTW.

    This function performs dynamic time warping (DTW) to align the timing of a
    query pitch sequence with a reference pitch sequence.

    Args:
        time (numpy.ndarray): Original time values for the query sequence.
        query (numpy.ndarray): Pitch values of the query sequence.
        reference (numpy.ndarray): Pitch values of the reference sequence.
        align_radius (int, optional): Search radius for FastDTW. Defaults to 1.

    Returns:
        numpy.ndarray: Interpolated pitch sequence aligned to the reference time axis.
    """
    seq_nonan = np.nan_to_num(zscore(query, nan_policy='omit'))
    pattern_nonan = np.nan_to_num(zscore(reference, nan_policy='omit'))
    distance, path = fastdtw(seq_nonan, pattern_nonan, radius=align_radius)

    path = np.array(path)
    aligned_time = np.interp(path[:, 1], np.arange(len(time)), time)
    aligned_seq = np.interp(path[:, 0], np.arange(len(query)), query)

    interp_seq = interp1d(aligned_time, aligned_seq, kind='linear',
                          fill_value="extrapolate")
    return interp_seq(time)


def align_sequence_pitch(query, reference, semitone_shift=None, smoothness=0):
    """Align pitch sequences by shifting in semitones and applying smoothing.

    This function adjusts the pitch sequence to match the reference pitch, allowing for optional smoothing.

    Args:
        query (numpy.ndarray): Pitch values to be aligned.
        reference (numpy.ndarray): Target reference pitch values.
        semitone_shift (int, optional): Number of semitones to shift the query pitch. If None, it is calculated automatically.
        smoothness (int, optional): Smoothing factor for the aligned pitch. Defaults to 0 (no smoothing).

    Returns:
        tuple: (pitch_aligned_query, semitone_shift), where:
            - pitch_aligned_query (numpy.ndarray): Aligned pitch values.
            - semitone_shift (int): Applied semitone shift.
    """
    if semitone_shift is None:
        base_pitch_wav = np.nanmedian(query)
        base_pitch_vocal = np.nanmedian(reference)
        semitone_shift = np.round(hz_to_midi(base_pitch_vocal)) - \
            np.round(hz_to_midi(base_pitch_wav))

    pitch_aligned_query = query * np.exp2(semitone_shift/12)

    if smoothness > 0:
        pitch_aligned_query = gaussian_filter1d_with_nan(
            pitch_aligned_query, sigma=smoothness)

    return pitch_aligned_query, semitone_shift


def pitch_correction(query, reference,
                     window_size=200, max_freq_diff=500, smoothness=20):
    """Apply pitch correction to align a query pitch sequence with a reference sequence.

    This function estimates the frequency shift needed to match the reference pitch
    by minimizing mean squared error over windowed segments. A Gaussian filter is used
    to smooth the estimated frequency shifts.

    Args:
        query (numpy.ndarray): The pitch sequence to be corrected.
        reference (numpy.ndarray): The target reference pitch sequence.
        window_size (int, optional): Number of samples per window for frequency shift estimation. Defaults to 200.
        max_freq_diff (int, optional): Maximum frequency shift allowed in Hz. Defaults to 500.
        smoothness (int, optional): Smoothing factor for the corrected pitch. Defaults to 20.

    Returns:
        numpy.ndarray: Corrected pitch sequence.
    """
    num_windows = len(query) // window_size
    freq_shifts = []
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size

        query_window = query[start_idx:end_idx]
        reference_window = reference[start_idx:end_idx]

        mses = [np.nanmean((query_window + freq_diff - reference_window)**2)
                for freq_diff in range(-max_freq_diff, max_freq_diff)]
        if any(np.isnan(mses)):
            freq_shift = np.nan
        else:
            freq_shift = np.argmin(mses) - max_freq_diff
        freq_shifts.append(freq_shift)

    expanded_shifts = np.full(len(query), np.nan)
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        expanded_shifts[start_idx:end_idx] = freq_shifts[i]

    smoothed_shifts = gaussian_filter1d_with_nan(
        expanded_shifts, sigma=smoothness)
    return query + smoothed_shifts


def get_pitch_delta(query, reference, scaler=2.5):
    """Calculate the difference between two pitch sequences.

    The delta represents the pitch correction needed to align the query sequence with the reference sequence.

    Args:
        query (numpy.ndarray): Pitch values from the query sequence.
        reference (numpy.ndarray): Pitch values from the reference sequence.
        scaler (float, optional): Scaling factor for the pitch difference. Defaults to 2.5.

    Returns:
        numpy.ndarray: Scaled pitch difference values.
    """
    return scaler * (query - reference)


if __name__ == '__main__':
    clear_cache()

    ustx_dict = load_ustx(ustx_path="samples/little_stars.ustx")
    tempo = ustx_dict['tempos'][0]['bpm']

    confidence_threshold = 0.8
    vocal_pitch, vocal_tick = get_wav_pitch_tick(
        wav_path="samples/little_stars_openutau.wav",
        tempo=tempo,
        confidence_threshold=0.8
    )

    wav_pitch, wav_tick = get_wav_pitch_tick(
        wav_path="samples/little_stars.wav",
        tempo=tempo,
        confidence_threshold=0.8
    )

    unified_tick, unified_wav_pitch, unified_vocal_pitch = unify_sequence_time(
        wav_tick,
        wav_pitch,
        vocal_tick,
        vocal_pitch
    )

    time_aligned_wav_pitch = align_sequence_time(
        unified_tick,
        unified_wav_pitch,
        unified_vocal_pitch,
        align_radius=1
    )

    time_pitch_aligned_wav_pitch, _ = align_sequence_pitch(
        time_aligned_wav_pitch,
        unified_vocal_pitch,
        semitone_shift=None,
        smoothness=1
    )

    delta_pitch = get_pitch_delta(
        time_pitch_aligned_wav_pitch,
        unified_vocal_pitch,
        scaler=2.5
    )

    edit_ustx_pitch(
        ustx_dict,
        unified_tick,
        delta_pitch,
        vocal_track_number=1
    )

    save_ustx(ustx_dict, ustx_path="output.ustx")
