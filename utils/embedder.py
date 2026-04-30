from __future__ import annotations

import abc
from math import gcd
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.signal import resample_poly
from scipy.ndimage import gaussian_filter1d

from utils.i18n import _


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(abc.ABC):
    """Common interface for all audio embedders.

    Sub-classes must set :attr:`NAME`, :attr:`SAMPLE_RATE`, :attr:`HOP_SIZE`,
    implement :meth:`_load_model` and :meth:`__call__`, and optionally
    override :meth:`download`.

    The inference backend is intentionally unconstrained — sub-classes may
    use ONNX Runtime, PyTorch, TensorFlow, or anything else; only the public
    ``__call__`` signature is enforced.
    """

    #: Short identifier used for factory registration and the download CLI.
    #: Must be a non-empty string in every concrete sub-class.
    NAME: ClassVar[str] = ""

    #: SPDX license identifier for the model weights, e.g. ``"MIT"``,
    #: ``"Apache-2.0"``, ``"CC-BY-4.0"``.  Empty string means unspecified.
    LICENSE: ClassVar[str] = ""

    #: Target sample rate in Hz.
    SAMPLE_RATE: ClassVar[int] = 16_000

    #: Hop size in samples between successive embedding frames.
    HOP_SIZE: ClassVar[int] = 320

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self._model_path = Path(model_path)
        self._device = device
        self._load_model(self._model_path, device)

    @abc.abstractmethod
    def _load_model(self, model_path: Path, device: str) -> None:
        """Initialise the inference backend from *model_path*.

        Called once from ``__init__``.  Load whatever framework the
        sub-class uses (ONNX Runtime, PyTorch, …) here.
        """

    @abc.abstractmethod
    def __call__(
        self, wav_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings from an audio file.

        Args:
            wav_path: Path to a WAV (or any soundfile-readable) file.

        Returns:
            embeddings:  ``(T, D)`` float32 array.
            frame_times: ``(T,)`` array of frame centre times in seconds.
        """

    @classmethod
    def download(
        cls,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> Path:
        """Download the model weights and return the local path.

        The base implementation raises :exc:`NotImplementedError`.
        Sub-classes that fetch weights from a remote source should override
        this.  Extra *kwargs* allow variant/precision selection without
        breaking the shared CLI signature.

        Args:
            cache_dir: Where to store the downloaded file.

        Returns:
            Local path to the ready-to-use model file.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement download()."
        )

    # ------------------------------------------------------------------
    # Shared audio helpers
    # ------------------------------------------------------------------

    def _load_audio(self, wav_path: str | Path) -> np.ndarray:
        """Read *wav_path*, convert to mono float32, resample to SAMPLE_RATE."""
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), always_2d=False)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sr != self.SAMPLE_RATE:
            g = gcd(sr, self.SAMPLE_RATE)
            audio = resample_poly(audio, self.SAMPLE_RATE // g, sr // g)

        return audio.astype(np.float32)

    def _frame_times(self, n_frames: int) -> np.ndarray:
        """Return centre times (seconds) for *n_frames* frames."""
        return (np.arange(n_frames) + 0.5) * (self.HOP_SIZE / self.SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Factory / registry
# ---------------------------------------------------------------------------

class EmbedderFactory:
    """Registry of :class:`BaseEmbedder` sub-classes, keyed by :attr:`~BaseEmbedder.NAME`.

    Sub-classes are registered by passing them to :meth:`register`, which
    doubles as a plain class decorator::

        @EmbedderFactory.register
        class MyEmbedder(BaseEmbedder):
            NAME = "my_model"
            ...

        # or after the fact:
        EmbedderFactory.register(MyEmbedder)

    :attr:`~BaseEmbedder.NAME` is the only thing that needs to be set —
    no separate string argument required.
    """

    _registry:       ClassVar[dict[str, type[BaseEmbedder]]] = {}
    _instance_cache: ClassVar[dict[tuple, BaseEmbedder]]     = {}

    @classmethod
    def register(cls, klass: type[BaseEmbedder]) -> type[BaseEmbedder]:
        """Add *klass* to the registry using ``klass.NAME``.

        Raises:
            TypeError:  If *klass* does not subclass :class:`BaseEmbedder`.
            ValueError: If :attr:`~BaseEmbedder.NAME` is empty.
        """
        if not issubclass(klass, BaseEmbedder):
            raise TypeError(f"{klass} must subclass BaseEmbedder")
        if not klass.NAME:
            raise ValueError(f"{klass.__name__}.NAME must be a non-empty string")
        cls._registry[klass.NAME] = klass
        return klass

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseEmbedder:
        """Return a cached :class:`BaseEmbedder` instance, creating it on first use.

        Instances are keyed by ``(name, *sorted_kwargs)`` so that different
        configurations each own a separate ONNX/CUDA session while the same
        configuration reuses the existing one — keeping the CUDA execution
        provider context alive between calls.

        All *kwargs* are forwarded to the embedder's ``__init__`` on first
        construction only.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            raise KeyError(
                f"No embedder registered as {name!r}. "
                f"Available: {list(cls._registry)}"
            )
        key = (name, tuple(sorted(kwargs.items())))
        if key not in cls._instance_cache:
            cls._instance_cache[key] = cls._registry[name](**kwargs)
        return cls._instance_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Release all cached embedder instances and free their resources.

        Call this to explicitly drop CUDA/ONNX sessions and reclaim GPU
        memory (e.g. before loading a different model or on shutdown).
        """
        cls._instance_cache.clear()

    @classmethod
    def list(cls) -> list[str]:
        """Return the names of all registered embedders."""
        return sorted(cls._registry)


# ---------------------------------------------------------------------------
# mHuBERT-147 (ONNX Runtime backend)
# ---------------------------------------------------------------------------

@EmbedderFactory.register
class mHuBERTEmbedder(BaseEmbedder):
    """mHuBERT-147 via ONNX Runtime, downloaded on demand from HuggingFace."""

    NAME:        ClassVar[str] = "mhubert"
    LICENSE:     ClassVar[str] = "CC-BY-NC-SA-4.0"
    SAMPLE_RATE: ClassVar[int] = 16_000
    HOP_SIZE:    ClassVar[int] = 320      # ~50 FPS

    REPO_ID: ClassVar[str] = "NewComer00/mHuBERT-147-ONNX"

    MODEL_FILES: ClassVar[dict[str, str]] = {
        "fp32":  "model.onnx",
        "fp16":  "model_fp16.onnx",
        "q4":    "model_q4.onnx",
        "q4f16": "model_q4f16.onnx",
        "bnb4":  "model_bnb4.onnx",
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        variant: str = "q4f16",
        device: str = "cpu",
    ) -> None:
        if model_path is None:
            model_path = self.download(variant=variant)
        super().__init__(model_path, device=device)

    def _load_model(self, model_path: Path, device: str) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session     = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        print(_("Loaded {name} model from '{path}'.").format(name=self.NAME, path=model_path))
        if "CUDAExecutionProvider" in self._session.get_providers():
            print(_("ONNX Runtime is using CUDA for inference acceleration."))

    def __call__(
        self, wav_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray]:
        audio   = self._load_audio(wav_path)
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: audio[np.newaxis, :]},
        )
        embeddings = outputs[0].squeeze(0)              # (T, D)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected embeddings of shape (T, D), got {embeddings.shape}"
            )
        return embeddings, self._frame_times(len(embeddings))

    @classmethod
    def download(                                       # type: ignore[override]
        cls,
        variant: str = "q4f16",
        cache_dir: str | Path | None = None,
    ) -> Path:
        """Download the requested ONNX variant from HuggingFace.

        Args:
            variant:   One of :attr:`MODEL_FILES` keys.
            cache_dir: Override HF Hub cache directory.

        Returns:
            Local path to the downloaded ``.onnx`` file.
        """
        if variant not in cls.MODEL_FILES:
            raise ValueError(
                f"Unknown variant {variant!r}. "
                f"Choose from: {list(cls.MODEL_FILES)}"
            )

        print(_("Fetching {name} ({variant}) from Internet or local cache...").format(name=cls.NAME, variant=variant))
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=cls.REPO_ID,
            filename=f"onnx/{cls.MODEL_FILES[variant]}",
            cache_dir=cache_dir,
        )


# ---------------------------------------------------------------------------
# Embedding feature extractors
# ---------------------------------------------------------------------------

def emb_boundary(emb: np.ndarray, smooth_sigma: float = 1.0) -> np.ndarray:
    """Phoneme boundary novelty curve (L2 distance between consecutive frames).

    Peaks sharply at phoneme/syllable transitions regardless of pitch or
    timbre, making it a strong DTW anchor for note boundary alignment.

    Args:
        emb:          Frame embeddings, shape ``(T, D)``.
        smooth_sigma: Gaussian smoothing sigma. Larger values widen peaks.

    Returns:
        Boundary novelty curve, shape ``(T,)``. Non-negative; peaks ≈ boundaries.
    """
    diff = np.linalg.norm(np.diff(emb, axis=0), axis=1)
    diff = np.append(diff, diff[-1])

    if smooth_sigma > 0:
        diff = gaussian_filter1d(diff, sigma=smooth_sigma)

    return diff.astype(np.float32)


def emb_frame_entropy(
    emb: np.ndarray,
    n_clusters: int = 128,
    temperature: float = 1.0,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Per-frame entropy over soft k-means cluster assignments.

    Low entropy  → confident phoneme centre (sustained vowel core).
    High entropy → ambiguous region, typically at phoneme transitions.

    Complements :func:`emb_boundary`: boundary peaks *at* the change;
    entropy stays elevated *through* the transition region.

    Args:
        emb:          Frame embeddings, shape ``(T, D)``.
        n_clusters:   Number of pseudo-phoneme clusters for k-means.
        temperature:  Softmax temperature applied to cosine distances.
        smooth_sigma: Gaussian smoothing sigma.

    Returns:
        Per-frame entropy values, shape ``(T,)``.
    """
    from sklearn.cluster import MiniBatchKMeans

    norms    = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.maximum(norms, 1e-8)

    km = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=0)
    km.fit(emb_norm)

    centroids = km.cluster_centers_
    c_norms   = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(c_norms, 1e-8)

    sim    = emb_norm @ centroids.T
    logits = sim / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs   = np.exp(logits)
    probs  /= probs.sum(axis=1, keepdims=True)

    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    if smooth_sigma > 0:
        entropy = gaussian_filter1d(entropy, sigma=smooth_sigma)

    return entropy.astype(np.float32)


def emb_self_similarity(
    emb: np.ndarray,
    window: int = 20,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Local self-similarity novelty curve.

    Returns the row-wise standard deviation of a local cosine-similarity
    matrix — high where the frame is changing rapidly, low in stable
    phonetic regions.

    Args:
        emb:          Frame embeddings, shape ``(T, D)``.
        window:       Half-width of the comparison window in frames.
                      At 50 FPS, ``window=20`` covers ±0.4 s.
        smooth_sigma: Gaussian smoothing sigma.

    Returns:
        Novelty curve, shape ``(T,)``.
    """
    norms    = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.maximum(norms, 1e-8)

    T = len(emb_norm)
    W = window

    # Pad both ends so every frame has a full 2W+1 neighbourhood
    padded  = np.pad(emb_norm, ((W, W), (0, 0)), mode="edge")   # (T+2W, D)
    # Build (T, 2W+1, D) sliding window view without copying data
    shape   = (T, 2 * W + 1, emb_norm.shape[1])
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    # Cosine similarity of each frame against its neighbourhood: (T, 2W+1)
    sim     = np.einsum("twd,td->tw", windows, emb_norm)
    novelty = sim.std(axis=1)

    if smooth_sigma > 0:
        novelty = gaussian_filter1d(novelty, sigma=smooth_sigma)

    return novelty.astype(np.float32)


def emb_velocity_acceleration(
    emb: np.ndarray,
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame velocity and acceleration in embedding space.

    Analogous to delta/delta-delta MFCCs but in the mHuBERT space.
    Velocity peaks at the *onset* of a transition; acceleration peaks
    slightly earlier, giving DTW an early-warning signal.

    Args:
        emb:          Frame embeddings, shape ``(T, D)``.
        smooth_sigma: Pre-smoothing sigma before differentiation.

    Returns:
        ``(velocity, acceleration)`` — each shape ``(T,)``.
    """
    emb_smooth = (
        gaussian_filter1d(emb, sigma=smooth_sigma, axis=0)
        if smooth_sigma > 0
        else emb
    )

    grad1 = np.gradient(emb_smooth, axis=0)
    grad2 = np.gradient(grad1,      axis=0)

    velocity     = np.linalg.norm(grad1, axis=1).astype(np.float32)
    acceleration = np.linalg.norm(grad2, axis=1).astype(np.float32)
    return velocity, acceleration


def emb_features(
    emb: np.ndarray,
    boundary_sigma: float = 1.0,
    entropy_clusters: int = 128,
    entropy_temperature: float = 1.0,
    entropy_sigma: float = 1.0,
    similarity_window: int = 20,
    similarity_sigma: float = 1.0,
    velocity_sigma: float = 1.0,
) -> np.ndarray:
    """Stack all DTW-oriented features into a single matrix.

    Row order:

    =========  =================================================
    Row 0      boundary novelty (sharp spikes at transitions)
    Row 1      frame entropy (elevated through transition regions)
    Row 2      self-similarity (stable in sustained phonemes)
    Row 3      velocity (magnitude of embedding change)
    Row 4      acceleration (rate of change of velocity)
    =========  =================================================

    Args:
        emb:                  Frame embeddings, shape ``(T, D)``.
        boundary_sigma:       Smoothing for boundary curve.
        entropy_clusters:     K-means clusters for entropy.
        entropy_temperature:  Softmax temperature for entropy.
        entropy_sigma:        Smoothing for entropy curve.
        similarity_window:    Half-window for self-similarity.
        similarity_sigma:     Smoothing for self-similarity.
        velocity_sigma:       Pre-smoothing before differentiation.

    Returns:
        Stacked feature matrix, shape ``(5, T)``.
    """
    boundary   = emb_boundary(emb, smooth_sigma=boundary_sigma)
    entropy    = emb_frame_entropy(
        emb,
        n_clusters=entropy_clusters,
        temperature=entropy_temperature,
        smooth_sigma=entropy_sigma,
    )
    similarity             = emb_self_similarity(emb, window=similarity_window, smooth_sigma=similarity_sigma)
    velocity, acceleration = emb_velocity_acceleration(emb, smooth_sigma=velocity_sigma)

    return np.vstack([boundary, entropy, similarity, velocity, acceleration])  # (5, T)
