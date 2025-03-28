import os
import csv
import shutil
import hashlib
from pathlib import Path
from itertools import accumulate

import crepe
import oyaml
import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.io import wavfile
from scipy.stats import zscore
from yamlcore import CoreLoader
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from platformdirs import user_cache_dir
from librosa import midi_to_hz, hz_to_midi
from scipy.ndimage import gaussian_filter1d


APP_NAME = "PitchLoader"
APP_AUTHOR = "newcomer00"
APP_VERSION = "0.1.0"
CACHE_DIR = user_cache_dir(APP_NAME, APP_AUTHOR, version=APP_VERSION)


def align_sequence_pitch(query, reference, semitone_shift=None, smoothness=0):
    """Align pitch sequences by shifting in semitones and applying smoothing.

    This function adjusts the pitch sequence to match the reference pitch, allowing for optional smoothing.

    Args:
        query (numpy.ndarray): Pitch values to be aligned. Shape: (n_time_points).
        reference (numpy.ndarray): Target reference pitch values. Shape: (n_time_points).
        semitone_shift (int, optional): Number of semitones to shift the query pitch. If None, it is calculated automatically.
        smoothness (int, optional): Smoothing factor for the aligned pitch. Defaults to 0 (no smoothing).

    Returns:
        tuple: (pitch_aligned_query, semitone_shift), where:
            - pitch_aligned_query (numpy.ndarray): Aligned pitch values. Shape: (n_time_points).
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


def extract_wav_frequency(file_path, use_cache=True):
    """Extract pitch frequency from a WAV file using Crepe.

    This function processes an audio file to extract pitch information using the
    CREPE algorithm. It supports caching to improve performance when processing
    the same file multiple times.

    Args:
        file_path (str): Path to the WAV file.
        use_cache (bool, optional): Whether to use cached data if available. Defaults to True.

    Returns:
        tuple: (time, frequency, confidence), where:
            - time (list of float): Time points in seconds. Shape: (n_time_points).
            - frequency (list of float): Detected pitch frequencies in Hz. Shape: (n_time_points).
            - confidence (list of float): Confidence values for the detected pitches. Shape: (n_time_points).
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


def extract_pitch_dynamics_trends(pitch, n_order=3):
    # Extract dynamic features
    pitch_grads = list(accumulate([pitch] * n_order, lambda x, _: np.gradient(x)))
    pitch_grads = np.vstack(pitch_grads)
    
    # Extract trend features
    pitch_trends = list(accumulate([pitch] * n_order, lambda x, _: np.cumsum(x)))
    pitch_trends = np.vstack(pitch_trends)
    return np.vstack([pitch_grads, pitch_trends])


def edit_ustx_pitch(ustx_dict, utau_track_number, tick_seq, pitch_seq):
    """Edit pitch data in a USTX file for a given vocal track.

    This modifies the pitch curve data within a specified vocal track.

    Args:
        ustx_dict (dict): USTX data structure.
        utau_track_number (int): Index of the vocal track to edit (1-based).
        tick_seq (numpy.ndarray): Sequence of MIDI tick values. Shape: (n_time_points).
        pitch_seq (numpy.ndarray): Sequence of pitch values (Hz). Shape: (n_time_points).
    """
    track_idx = utau_track_number - 1  # track index starts from 0
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


def get_ustx_pitch_tick(ustx_dict, utau_track_number=1,
                         smoothness=3):
    """Extract the expected pitch sequence from a USTX file.

    This function retrieves the pitch curve based on note information in the USTX format.
    The extracted pitch sequence can be optionally smoothed using a Gaussian filter.

    Args:
        ustx_dict (dict): Parsed USTX data.
        utau_track_number (int, optional): The track number to extract pitch from (1-based index). Defaults to 1.
        smoothness (int, optional): Smoothing factor for the extracted pitch. Defaults to 3.

    Returns:
        tuple: (utau_pitch, utau_tick), where:
            - utau_tick (numpy.ndarray): Corresponding MIDI tick values. Shape: (n_time_points).
            - utau_pitch (numpy.ndarray): Extracted pitch frequencies in Hz. Shape: (n_time_points).
    """
    track_idx = utau_track_number - 1  # track index starts from 0
    notes = ustx_dict['voice_parts'][track_idx]['notes']

    utau_tick_start = notes[0]['position']
    utau_tick_end = notes[-1]['position'] + notes[-1]['duration']

    utau_tick = np.arange(utau_tick_start, utau_tick_end)
    utau_pitch = np.full(len(utau_tick), np.nan)
    for note in notes:
        start_idx = note['position'] - utau_tick_start
        end_idx = start_idx + note['duration']
        frq = midi_to_hz(note['tone'])
        utau_pitch[start_idx:end_idx] = frq

    utau_pitch = gaussian_filter1d_with_nan(utau_pitch, sigma=smoothness)
    return utau_tick, utau_pitch


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


def sequence_interval_intersection(seqs):
    """Find the intersection of multiple sequences.

    This function finds the intersection of multiple sequences, returning a sorted list of unique values.
    
    Args:
        seqs (list of list): List of sequences to intersect.

    Returns:
        list: Sorted list of unique values in the intersection.

    Example:
        seqs = [[0, 1, 2, 3], [1., 1.1, 2., 4., 5.]]
        result = [1., 1.1, 2., 3.]
    """
    min_val = max(min(s) for s in seqs)  # Highest lower bound
    max_val = min(max(s) for s in seqs)  # Lowest upper bound
    return [x for x in np.unique(np.concatenate(seqs))
            if min_val <= x <= max_val]


def unify_sequence_time(seq_times, seq_vals):
    """Unify multiple sequences to a common time base.

    This function aligns multiple sequences to a common time base by interpolating values.

    Args:
        seq_times (list of array-like): List of time sequences. Shape: (n_sequences, n_time_points).
        seq_vals (list of array-like): List of value sequences. Shape: (n_sequences, n_time_points).

    Returns:
        tuple: (unified_seq_time, unified_seqs_val), where:
            - unified_seq_time (numpy.ndarray): Unified time base. Shape: (n_time_points).
            - unified_seqs_val (tuple): Unified value sequences. Shape: (n_sequences, n_time_points).
    """
    unified_seq_time = np.array(
        sequence_interval_intersection(seq_times))
    unified_seq_time = np.unique(np.round(unified_seq_time).astype(int))
    unified_seqs_val = [interp1d(st, sv, fill_value="extrapolate")(unified_seq_time) \
            for (st, sv) in zip(seq_times, seq_vals)]
    return unified_seq_time, tuple(unified_seqs_val)


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


def extract_wav_mfcc(wav_path, n_feat=6, n_mfcc=13):
    """Extract MFCC features from a WAV file.

    This function extracts Mel-frequency cepstral coefficients (MFCC) from a WAV file.

    Args:
        wav_path (str): Path to the WAV file.
        n_feat (int, optional): Number of features to extract. Defaults to 6.
        n_mfcc (int, optional): Number of MFCC coefficients to extract. Defaults to 13.

    Returns:
        tuple: (mfcc_time, mfcc), where:
            - mfcc_time (numpy.ndarray): Time points for the MFCC features. Shape: (n_time_points).
            - mfcc (numpy.ndarray): Extracted MFCC features. Shape: (n_features, n_time_points).
    """
    sr = librosa.get_samplerate(wav_path)
    y, _ = librosa.load(wav_path, sr=sr)
    
    # Extract MFCC features
    _mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_time = librosa.times_like(_mfcc, sr=sr)
    
    # Add dynamic features into the MFCC
    delta_mfcc = librosa.feature.delta(_mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(_mfcc, order=2)
    mfcc = np.vstack([_mfcc, delta_mfcc, delta2_mfcc])
    
    # PCA to reduce dimensionality
    pca = PCA(n_components=n_feat)
    mfcc = pca.fit_transform(mfcc.T).T
    return mfcc_time, mfcc


# TODO: Deal with different tempo or ppqn within the same USTX file
def get_wav_features(wav_path, tempo=120, ppqn=480,
                       confidence_threshold=0.8, confidence_filter_size=9):
    """Extract features from a WAV file.

    This function extracts pitch and MFCC features from a WAV file, aligning them to a common time base.

    Args:
        wav_path (str): Path to the WAV file.
        tempo (float, optional): Tempo in beats per minute (BPM). Defaults to 120.
        ppqn (int, optional): Pulses per quarter note (MIDI resolution). Defaults to 480.
        confidence_threshold (float, optional): Confidence threshold for pitch detection. Defaults to 0.8.
        confidence_filter_size (int, optional): Size of the median filter for confidence. Defaults to 9.

    Returns:
        tuple: (wav_tick, wav_pitch, wav_features), where:
            - wav_tick (numpy.ndarray): MIDI ticks for the extracted features. Shape: (n_time_points).
            - wav_pitch (numpy.ndarray): Extracted pitch values in Hz. Shape: (n_time_points).
            - wav_features (tuple): Extracted feature sequences. Shape: (n_features, n_time_points).
    """
    feature_times = []  # List of time sequences(list of lists)
    feature_vals = []   # List of feature sequences(list of lists)

    # Extract features from WAV file
    time, frequency, confidence = extract_wav_frequency(wav_path, True)
    mask = medfilt(np.array(confidence),
                   kernel_size=confidence_filter_size) < confidence_threshold
    (pitch := np.array(frequency))[mask] = np.nan

    pitch_tick = time_to_ticks(time, tempo, ppqn)
    feature_times += [pitch_tick]
    feature_vals += [pitch]

    # Extract pitch dynamics trends
    pitch_features = extract_pitch_dynamics_trends(pitch)
    feature_times += [pitch_tick] * len(pitch_features)
    feature_vals += list(pitch_features)
    
    # Extract MFCC features
    mfcc_time, mfcc = extract_wav_mfcc(wav_path)
    mfcc_tick = time_to_ticks(mfcc_time, tempo, ppqn)
    feature_times += [mfcc_tick] * len(mfcc)
    feature_vals += list(mfcc)

    # Unified time and features
    wav_tick, (wav_pitch, *wav_features) = \
        unify_sequence_time(
            seq_times=feature_times, 
            seq_vals=feature_vals
        )
    return wav_tick, wav_pitch, wav_features


def align_sequence_time(query_time, queries, reference_time, references, align_radius=1):
    """Align sequences to a common time base using dynamic time warping.

    This function aligns multiple sequences to a common time base using dynamic time warping.

    Args:
        query_time (numpy.ndarray): Time values for the query sequences. Shape: (n_time_points).
        queries (tuple): Query sequences to align. Shape: (n_sequences, n_time_points).
        reference_time (numpy.ndarray): Time values for the reference sequences. Shape: (n_time_points).
        references (tuple): Reference sequences to align. Shape: (n_sequences, n_time_points).
        align_radius (int, optional): Radius for dynamic time warping. Defaults to 1.

    Returns:
        tuple: (unified_time, aligned_queries, unified_references), where:
            - unified_time (numpy.ndarray): Unified time base. Shape: (n_time_points).
            - aligned_queries (tuple): Aligned query sequences. Shape: (n_sequences, n_time_points).
            - unified_references (tuple): Unified reference sequences. Shape: (n_sequences, n_time_points).
    """
    query_times = [query_time] * len(queries)
    reference_times = [reference_time] * len(references)
    
    # Unify time and sequences
    unified_time, seqs = unify_sequence_time(
        (*query_times, *reference_times), (*queries, *references))
    unified_queries = list(seqs)[:len(queries)]
    unified_references = list(seqs)[len(queries):]

    # Align sequences using dynamic time warping
    qs_nonan = np.nan_to_num(zscore(unified_queries, axis=1, nan_policy='omit'))
    rs_nonan = np.nan_to_num(zscore(unified_references, axis=1, nan_policy='omit'))
    distance, path = fastdtw(
        list(map(tuple, zip(*qs_nonan))), list(map(tuple, zip(*rs_nonan))),
        radius=align_radius)

    # Align queries to reference time
    path = np.array(path)
    aligned_queries = []
    for q in unified_queries:
        aligned_time = np.interp(path[:, 1], np.arange(len(unified_time)), unified_time)
        aligned_seq = np.interp(path[:, 0], np.arange(len(q)), q)
        interp_seq = interp1d(aligned_time, aligned_seq, fill_value="extrapolate")
        aligned_queries.append(interp_seq(unified_time))

    return unified_time, tuple(aligned_queries), tuple(unified_references)


if __name__ == '__main__':
    # clear_cache()

    # Load USTX file and extract tempo
    ustx_dict = load_ustx(ustx_path="examples/Прекрасное Далеко/project.ustx")
    tempo = ustx_dict['tempos'][0]['bpm']

    # Extract pitch features from WAV files
    input_wav = "examples/Прекрасное Далеко/utau.wav"
    utau_tick, utau_pitch, utau_features = get_wav_features(
        wav_path=input_wav,
        tempo=tempo,
        confidence_threshold=0.6
    )

    # Extract pitch features from reference WAV file
    reference_wav = "examples/Прекрасное Далеко/reference.wav"
    ref_tick, ref_pitch, ref_features = get_wav_features(
        wav_path=reference_wav,
        tempo=tempo,
        confidence_threshold=0.8
    )

    # Align all features to a common time base
    # NOTICE: features from UTAU WAV are the reference, and those from Ref. WAV are the query
    unified_tick, (time_aligned_ref_pitch, *_), (unified_utau_pitch, *_) = \
        align_sequence_time(
            query_time=ref_tick,
            queries=(ref_pitch, *ref_features),
            reference_time=utau_tick,
            references=(utau_pitch, *utau_features),
            align_radius=1
        )

    # Align pitch sequences in pitch axis
    time_pitch_aligned_ref_pitch, _ = align_sequence_pitch(
        time_aligned_ref_pitch,
        unified_utau_pitch,
        semitone_shift=None,
        smoothness=2
    )

    # Calculate pitch delta for USTX pitch editing
    delta_pitch = get_pitch_delta(
        time_pitch_aligned_ref_pitch,
        unified_utau_pitch,
        scaler=2
    )

    # Edit USTX pitch data
    edit_ustx_pitch(
        ustx_dict,
        1,
        unified_tick,
        delta_pitch
    )

    # Save USTX file
    save_ustx(ustx_dict, ustx_path="examples/Прекрасное Далеко/test-output.ustx")
