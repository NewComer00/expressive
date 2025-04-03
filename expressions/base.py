import threading

import numpy as np

from utils.ustx import load_ustx, save_ustx, edit_ustx_expression_curve


class ExpressionLoader:
    expression_name: str = ""
    ustx_lock = threading.Lock()

    def __init__(self, ref_path: str, utau_path: str, ustx_path: str):
        self.expression_tick: list | np.ndarray = []
        self.expression_val: list | np.ndarray = []
        self.ref_path = ref_path
        self.utau_path = utau_path
        self.ustx_path = ustx_path
        self.tempo = load_ustx(self.ustx_path)["tempos"][0]["bpm"]

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


EXPRESSION_LOADER_TABLE = {}


def register_expression(cls: type[ExpressionLoader]):
    EXPRESSION_LOADER_TABLE[cls.expression_name] = cls


def createExpressionLoader(expression_name: str) -> type[ExpressionLoader]:
    return EXPRESSION_LOADER_TABLE[expression_name]
