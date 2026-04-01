import logging
from typing import Any
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
from filelock import FileLock

from utils.plot import Plot
from utils.i18n import _, _l
from utils.ustx import UstxEditor
from utils.fs import APP_RUNTIME_PATH
from utils.seqtool import set_tick_converters
from utils.wavtool import ClampedWav, sec2timestamp
from utils.relay import Collector, history_cleanup, set_relay_dir, ExpressionLoaderCollectorNaming


set_relay_dir(APP_RUNTIME_PATH / "relay")


@dataclass
class Args:
    name: str
    type: type
    default: Any | None
    help: str
    choices: list | None = None


class ExpressionLoader():
    """Base class for expression loaders.

    An expression loader extracts a single OpenUtau expression curve (e.g.
    ``dyn``, ``pitd``, ``tenc``) by comparing a reference audio recording
    against the rendered UTAU audio, then writes the result back into a
    ``.ustx`` project file.

    Subclasses must set :attr:`expression_name` and override
    :meth:`get_expression`.  Registering a subclass with
    :func:`register_expression` makes it discoverable via
    :func:`getExpressionLoader`.

    The loader opens an exclusive :class:`~utils.ustx.UstxEditor` on
    *ustx_path* during ``__init__`` and holds it until the instance is
    garbage-collected, so only one loader per file should be alive at a time
    within a single process.  Across processes the file lock prevents
    concurrent writes.

    Class attributes:
        expression_name (str):  Short abbreviation used as the USTX curve key
                                (e.g. ``"dyn"``).  Must be set on the subclass.
        expression_info (str):  Human-readable description of the expression.
        args (SimpleNamespace): Declared CLI / GUI arguments for this loader.
                                Each value is an :class:`Args` instance.
        plots (SimpleNamespace): Declared plot names for this loader.
                                Each value is a :class:`Plot` instance.

    USTX attributes:
        ustx_path (str):              Path to the ``.ustx`` project file.
        ustx_editor (UstxEditor):     Live editor holding the file lock.
        ustx_time_axis (TimeAxis):    Tempo-map-aware tick converter built from
                                      the project's tempo and time-signature maps.

    Audio attributes:
        ref_path (str):       Path to the (possibly trimmed) reference audio.
        ref_offset (float):   Start offset of the reference clip in seconds.
        ref_duration (float): Duration of the reference clip in seconds.
        utau_path (str):      Path to the (possibly trimmed) UTAU audio.
        utau_offset (float):  Start offset of the UTAU clip in seconds.
        utau_duration (float): Duration of the UTAU clip in seconds.

    Result attributes:
        expression_tick (ndarray): Tick positions produced by the last
                                   :meth:`get_expression` call.
        expression_val (ndarray):  Curve values produced by the last
                                   :meth:`get_expression` call.
    """
    _id_counter: int = 0
    expression_name: str = ""
    expression_info: str = ""
    _init_lock: FileLock = FileLock(
        APP_RUNTIME_PATH / "expressionloader_init.lock",
        thread_local=False, is_singleton=True, timeout=-1,
    )
    args = SimpleNamespace(
        ref_path     = Args(name="ref_path"    , type=str, default=""  , help=_l("Path to the **reference** audio file")),  # noqa: E501
        utau_path    = Args(name="utau_path"   , type=str, default=""  , help=_l("Path to the **UTAU** audio file")),  # noqa: E501
        ustx_path    = Args(name="ustx_path"   , type=str, default=""  , help=_l("Path to the `.ustx` project file to be processed")),  # noqa: E501
        track_number = Args(name="track_number", type=int, default=1   , help=_l("**Track number** to apply expressions to (1-based index)")),  # noqa: E501
        ref_start    = Args(name="ref_start"   , type=str, default=None, help=_l("**Start time** of the **reference** audio (format `M:S`, e.g. `0:10.01`). Omit to specify the beginning")),  # noqa: E501
        ref_end      = Args(name="ref_end"     , type=str, default=None, help=_l("**End time** of the **reference** audio (format `M:S`, e.g. `0:10.01`). Omit to specify the ending")),  # noqa: E501
        utau_start   = Args(name="utau_start"  , type=str, default=None, help=_l("**Start time** of the **UTAU** audio (format `M:S`, e.g. `0:10.01`). Omit to specify the beginning")),  # noqa: E501
        utau_end     = Args(name="utau_end"    , type=str, default=None, help=_l("**End time** of the **UTAU** audio (format `M:S`, e.g. `0:10.01`). Omit to specify the ending")),  # noqa: E501
    )
    plots = SimpleNamespace(
        expression = Plot(tag="expression", title=expression_info, x_label="Tick", y_label=expression_name, legends=[expression_name]),  # noqa: E501
    )

    @classmethod
    def get_args_dict(cls) -> dict[str, Args]:
        return cls.args.__dict__

    def __init__(self, ref_path: str, utau_path: str, ustx_path: str,
                 ref_start: str | None = None, ref_end: str | None = None,
                 utau_start: str | None = None, utau_end: str | None = None):

        # Identify this loader instance and clean up plots history
        with ExpressionLoader._init_lock:
            ExpressionLoader._id_counter += 1
            self.id = ExpressionLoader._id_counter
            if self.id == 1:
                history_cleanup()

        # Set up logging
        self.logger = logging.getLogger(f"{ExpressionLoader.__name__}.{self.expression_name}.{self.id}")
        self.logger = logging.LoggerAdapter(self.logger, {"expression": self.expression_name})
        self.logger.setLevel(logging.DEBUG)

        # Init relay collector
        self.collector = Collector(ExpressionLoaderCollectorNaming.make(self.expression_name, self.id))

        # Init USTX editor (with exclusive file lock)
        self.ustx_path = ustx_path
        self.ustx_editor = UstxEditor(self.ustx_path)
        self.ustx_time_axis = self.ustx_editor.build_time_axis()
        # Register tempo-map-aware tick converters
        set_tick_converters(
            self.ustx_time_axis.seconds_to_ticks,
            self.ustx_time_axis.ticks_to_seconds,
        )

        # Clamp audio files
        self._clamped_ref = ClampedWav(ref_path,  ref_start,  ref_end,  logger=self.logger)
        self.ref_path,  self.ref_offset,  self.ref_duration  = (
            self._clamped_ref.path, self._clamped_ref.offset_sec, self._clamped_ref.duration_sec)
        self.logger.info(_("ref  [{} → {}] {:.3f}s").format(
            sec2timestamp(self.ref_offset),
            sec2timestamp(self.ref_offset  + self.ref_duration),
            self.ref_duration))

        self._clamped_utau = ClampedWav(utau_path, utau_start, utau_end, logger=self.logger)
        self.utau_path, self.utau_offset, self.utau_duration = (
            self._clamped_utau.path, self._clamped_utau.offset_sec, self._clamped_utau.duration_sec)
        self.logger.info(_("utau [{} → {}] {:.3f}s").format(
            sec2timestamp(self.utau_offset),
            sec2timestamp(self.utau_offset + self.utau_duration),
            self.utau_duration))

        # Init other attributes
        self.expression_tick: list | np.ndarray = []
        self.expression_val:  list | np.ndarray = []
        self.logger.info(_("Initialization complete."))

    def __del__(self):
        self.collector.flush()
        self.ustx_editor.close()

    def get_expression(self, *args, **kwargs):
        self.collect_plot(self.plots.expression, (self.expression_tick, self.expression_val))
        return self.expression_tick, self.expression_val

    def load_to_ustx(self, track_number: int):
        if len(self.expression_tick) > 0 and len(self.expression_val) > 0:
            track_no = track_number - 1
            # Apply offset first
            shifted_ticks = self.ustx_time_axis.shift_ticks_by_seconds(
                np.asarray(self.expression_tick), self.utau_offset
            )
            self.ustx_editor.add_expression_to_track(
                track_no,
                self.__class__.expression_name,
                shifted_ticks,
                self.expression_val,
            )
            self.ustx_editor.save()
            self.logger.info(_("Expression written to USTX file: '{}'").format(self.ustx_path))
        else:
            self.logger.warning(_("Expression result is empty. Skipping USTX update."))

    def collect_plot(self, plot: "Plot", *series: tuple) -> None:
        self.collector.register(**plot.fig(*series))


# Dictionary to hold registered expression loader classes
# This dictionary maps expression names to their corresponding loader classes
EXPRESSION_LOADER_TABLE: dict[str, type[ExpressionLoader]] = {}


def register_expression(cls: type[ExpressionLoader]):
    """Register an expression loader class.

    This function adds the class to the EXPRESSION_LOADER_TABLE dictionary
    using the class's expression_name attribute as the key.

    Args:
        cls (type[ExpressionLoader]): The expression loader class to register.
    """
    EXPRESSION_LOADER_TABLE[cls.expression_name] = cls
    return cls


def getExpressionLoader(expression_name: str | None) -> type[ExpressionLoader]:
    """Get the expression loader class for the specified expression name.

    This function returns the class from the EXPRESSION_LOADER_TABLE dictionary
    that corresponds to the given expression name. If expression_name is None,
    it returns the base ExpressionLoader class.
    If the expression name is not found in the table, a ValueError is raised.

    Args:
        expression_name (str | None): The name of the expression to get the loader for.

    Returns:
        type[ExpressionLoader]: The class of the expression loader.

    Raises:
        ValueError: If the expression name is not found in the EXPRESSION_LOADER_TABLE.
    """
    if expression_name is None:
        return ExpressionLoader
    if expression_name not in EXPRESSION_LOADER_TABLE:
        raise ValueError(f"Expression '{expression_name}' is not registered or not supported.")
    return EXPRESSION_LOADER_TABLE[expression_name]


def get_registered_expressions() -> list[str]:
    """Get a list of registered expression names.

    This function returns a list of all expression names that have been
    registered in the EXPRESSION_LOADER_TABLE dictionary.

    Returns:
        list[str]: A list of registered expression names.
    """
    return list(EXPRESSION_LOADER_TABLE)
