from shutil import copy

from utils.gpu import add_cuda_to_path
from expressions.base import getExpressionLoader, get_registered_expressions


def process_expressions(
    utau_wav: str,
    ref_wav: str,
    ustx_input: str,
    ustx_output: str,
    track_number: int,
    expressions: list[dict],
):
    copy(ustx_input, ustx_output)

    for exp in expressions:
        exp_type = exp["expression"]

        if exp_type not in get_registered_expressions():
            raise ValueError(f"Expression '{exp_type}' is not supported.")

        loader = getExpressionLoader(exp_type)(ref_wav, utau_wav, ustx_output)
        loader_args = {
            arg_name: exp.get(arg_name, arg.default)
            for arg_name, arg in loader.get_args_dict().items()
        }
        loader.get_expression(**loader_args)
        loader.load_to_ustx(track_number)


if __name__ == "__main__":
    add_cuda_to_path()

    utau_wav = "examples/Прекрасное Далеко/utau.wav"
    ref_wav = "examples/Прекрасное Далеко/reference.wav"
    ustx_input = "examples/Прекрасное Далеко/project.ustx"
    ustx_output = "examples/Прекрасное Далеко/output.ustx"
    track_number = 1
    expressions = [
        {
            "expression": "dyn",
            "align_radius": 1,
            "smoothness": 2,
            "scaler": 2.0,
        },
        {
            "expression": "pitd",
            "confidence_utau": 0.8,
            "confidence_ref": 0.6,
            "align_radius": 1,
            "semitone_shift": None,
            "smoothness": 2,
            "scaler": 2.0,
        },
        {
            "expression": "tenc",
            "align_radius": 1,
            "smoothness": 2,
            "scaler": 2.0,
            "bias": 20,
        },
    ]

    process_expressions(
        utau_wav, ref_wav, ustx_input, ustx_output, track_number, expressions
    )
