import logging
import argparse
import tempfile
from shutil import copy
from datetime import datetime
from contextlib import contextmanager
from os.path import splitext, basename

from __version__ import VERSION
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
    """
    Process the specified expressions and apply them to the USTX project.

    Args:
        utau_wav (str): Path to the UTAU audio file.
        ref_wav (str): Path to the reference audio file.
        ustx_input (str): Path to the input USTX project file.
        ustx_output (str): Path to save the processed USTX project file.
        track_number (int): Track number to apply expressions.
        expressions (list[dict]): List of expressions to process, each containing:
            - "expression": Expression type (e.g., "dyn", "pitd", "tenc").
            - Additional parameters specific to the expression type.
            - Example:
```
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
```
    """
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



@contextmanager
def setup_loggers():
    log_file = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=f"expressive_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}_",
        suffix=".log",
    )
    log_path = log_file.name
    log_file.close()  # Close immediately; we'll reopen via FileHandler

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set up file handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8-sig")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Set up stdout handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # Main application logger
    logger_app = logging.getLogger(splitext(basename(__file__))[0])
    logger_app.setLevel(logging.DEBUG)
    logger_app.addHandler(file_handler)
    logger_app.addHandler(stream_handler)

    # Expression loader logger
    logger_exp = logging.getLogger(getExpressionLoader(None).__name__)
    logger_exp.setLevel(logging.DEBUG)
    logger_exp.addHandler(file_handler)

    try:
        yield logger_app, logger_exp, log_path
    finally:
        logger_app.info(f"Logs saved to {log_path}")
        for logger in [logger_app, logger_exp]:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate expressions from real singers to DiffSingers (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General arguments
    general_args = getExpressionLoader(None).args
    parser.add_argument("-u", "--utau_wav",     type=general_args.utau_path.type,    required=True, help=general_args.utau_path.help)  # noqa: E501
    parser.add_argument("-r", "--ref_wav",      type=general_args.ref_path.type,     required=True, help=general_args.ref_path.help)  # noqa: E501
    parser.add_argument("-i", "--ustx_input",   type=general_args.ustx_path.type,    required=True, help=general_args.ustx_path.help)  # noqa: E501
    parser.add_argument("-o", "--ustx_output",  type=str,                            required=True, help="Path to save the processed USTX file")  # noqa: E501
    parser.add_argument("-t", "--track_number", type=general_args.track_number.type, required=True, help=general_args.track_number.help)  # noqa: E501

    parser.add_argument("-e", "--expression", type=str, action="append", required=True, choices=get_registered_expressions(), 
                        help="Specify expressions to apply (e.g., --expression dyn --expression pitd)")
    parser.add_argument("--version", action="version", version=f"%(prog)s v{VERSION}")

    # Expression-specific arguments
    expression_names = get_registered_expressions()
    get_expression_args = lambda exp_name: getExpressionLoader(exp_name).get_args_dict()

    for exp_name in expression_names:
        group = parser.add_argument_group(f"{exp_name.upper()} Expression")
        for arg_name, arg in get_expression_args(exp_name).items():
            group.add_argument(f"--{exp_name}.{arg_name}",
                                type=arg.type, default=arg.default, help=arg.help)

    # Parse arguments
    args = parser.parse_args()
    expressions = [
        {
            "expression": exp_name,
            **{
                arg.name: getattr(args, f"{exp_name}.{arg.name}")
                for arg in get_expression_args(exp_name).values()
            }
        } for exp_name in expression_names if exp_name in args.expression
    ]

    # Process expressions
    with setup_loggers() as (logger_app, _, _):
        logger_app.info("Starting Expressive CLI...")
        try:
            process_expressions(
                args.utau_wav, args.ref_wav, args.ustx_input,
                args.ustx_output, args.track_number, expressions
            )
        except Exception as e:
            logger_app.error(f"Error occurred during processing: {e}")
            raise
        else:
            logger_app.info("Processing completed successfully!")


if __name__ == "__main__":
    add_cuda_to_path()
    main()
