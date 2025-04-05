import argparse

from utils.gpu import add_cuda11_to_path
from expressive import process_expressions
from expressions.base import getExpressionLoader, get_registered_expressions


def main():
    parser = argparse.ArgumentParser(
        description="Process USTX expressions from reference and UTAU WAV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General arguments
    general_args = getExpressionLoader(None).args
    parser.add_argument("-u", "--utau_wav",     type=general_args.utau_path.type,    required=True, help=general_args.utau_path.help)
    parser.add_argument("-r", "--ref_wav",      type=general_args.ref_path.type,     required=True, help=general_args.ref_path.help)
    parser.add_argument("-i", "--ustx_input",   type=general_args.ustx_path.type,    required=True, help=general_args.ustx_path.help)
    parser.add_argument("-o", "--ustx_output",  type=str,                            required=True, help="Path to save the processed USTX file")
    parser.add_argument("-t", "--track_number", type=general_args.track_number.type, required=True, help=general_args.track_number.help)

    # Expression selection
    parser.add_argument("-e", "--expression", type=str, action="append", required=True, choices=["dyn", "pitd"], 
                        help="Specify expressions to apply (e.g., --expression dyn --expression pitd)")

    def group_expr_args(parser: argparse.ArgumentParser, name, args_ns):
        group = parser.add_argument_group(f"{name.upper()} Expression")
        for arg_name, arg in args_ns.__dict__.items():
            group.add_argument(f"--{name}.{arg_name}", type=arg.type, default=arg.default, help=arg.help)
        return group

    def collect_expr_values(name, args_ns, parsed_args):
        return {
            "expression": name,
            **{
                arg.name: getattr(parsed_args, f"{name}.{arg.name}")
                for arg in args_ns.__dict__.values()
            }
        }

    expression_names = get_registered_expressions()

    # Add groups
    arg_sources = {name: getExpressionLoader(name).args for name in expression_names}
    for name, args_ns in arg_sources.items():
        group_expr_args(parser, name, args_ns)

    # Parse
    args = parser.parse_args()

    # Collect selected expressions
    expressions = [
        collect_expr_values(name, arg_sources[name], args)
        for name in expression_names
        if name in args.expression
    ]

    # Process
    process_expressions(
        args.utau_wav, args.ref_wav, args.ustx_input,
        args.ustx_output, args.track_number, expressions
    )

if __name__ == "__main__":
    add_cuda11_to_path()
    main()
