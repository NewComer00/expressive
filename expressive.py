from shutil import copy

from utils.gpu import add_cuda11_to_path
from expressions.base import createExpressionLoader


def process_expressions(
    utau_wav, ref_wav, ustx_input, ustx_output, track_number, expressions
):
    copy(ustx_input, ustx_output)

    for exp in expressions:
        exp_type = exp["expression"]
        loader = createExpressionLoader(exp_type)(ref_wav, utau_wav, ustx_output)

        if exp_type == "dyn":
            loader.get_expression(
                align_radius=exp.get("align_radius", 1),
                smoothness=exp.get("smoothness", 2),
                scaler=exp.get("scaler", 2.0),
            )

        elif exp_type == "pitd":
            loader.get_expression(
                confidence_utau=exp.get("confidence_utau", 0.8),
                confidence_ref=exp.get("confidence_ref", 0.6),
                align_radius=exp.get("align_radius", 1),
                semitone_shift=exp.get("semitone_shift", None),
                smoothness=exp.get("smoothness", 2),
                scaler=exp.get("scaler", 2.0),
            )

        loader.load_to_ustx(track_number)


if __name__ == "__main__":
    add_cuda11_to_path()

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
    ]

    process_expressions(
        utau_wav, ref_wav, ustx_input, ustx_output, track_number, expressions
    )
