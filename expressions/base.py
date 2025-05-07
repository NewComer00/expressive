import logging
import threading
from typing import Any
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np

from utils.i18n import _
from utils.ustx import load_ustx, save_ustx, edit_ustx_expression_curve


@dataclass
class Args:
    name: str
    type: type
    default: Any | None
    help: str


class ExpressionLoader():
    _id_counter: int = 0
    expression_name: str = ""
    expression_info: str = ""
    ustx_lock = threading.Lock()
    args = SimpleNamespace(
        ref_path     = Args(name="ref_path"    , type=str, default="", help=_("Path to the reference audio file")),
        utau_path    = Args(name="utau_path"   , type=str, default="", help=_("Path to the UTAU audio file")),
        ustx_path    = Args(name="ustx_path"   , type=str, default="", help=_("Path to the USTX project file to be processed")),
        track_number = Args(name="track_number", type=int, default=1 , help=_("Track number to apply expressions")),
    )

    @classmethod
    def get_args_dict(cls) -> dict[str, Args]:
        return cls.args.__dict__

    def __init__(self, ref_path: str, utau_path: str, ustx_path: str):
        ExpressionLoader._id_counter += 1
        self.id = ExpressionLoader._id_counter
        self.logger = logging.getLogger(f"{ExpressionLoader.__name__}.{self.expression_name}.{self.id}")
        self.logger = logging.LoggerAdapter(self.logger, {"expression": self.expression_name})
        self.logger.setLevel(logging.DEBUG)

        self.expression_tick: list | np.ndarray = []
        self.expression_val: list | np.ndarray = []
        self.ref_path = ref_path
        self.utau_path = utau_path
        self.ustx_path = ustx_path
        self.tempo = load_ustx(self.ustx_path)["tempos"][0]["bpm"]
        self.logger.info(_("Initialization complete."))

    def get_expression(self, *args, **kwargs):
        return self.expression_tick, self.expression_val

    def load_to_ustx(self, track_number: int):
        if len(self.expression_tick) > 0 and len(self.expression_val) > 0:
            with self.__class__.ustx_lock:
                ustx_dict = load_ustx(self.ustx_path)
                edit_ustx_expression_curve(
                    ustx_dict,
                    track_number,
                    self.__class__.expression_name,
                    self.expression_tick,
                    self.expression_val,
                )
                save_ustx(ustx_dict, self.ustx_path)
                self.logger.info(_("Expression written to USTX file: '{}'").format(self.ustx_path))
        else:
            self.logger.warning(_("Expression result is empty. Skipping USTX update."))


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
