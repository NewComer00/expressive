import argparse
from typing import ClassVar
from argparse import ArgumentDefaultsHelpFormatter

from rich.text import Text
from rich.containers import Lines
from rich_argparse import RichHelpFormatter

from expressions.base import getExpressionLoader


class WrappedTextRichHelpFormatter(RichHelpFormatter):
    """RichHelpFormatter that wraps long lines in help text while preserving rich formatting.
    Cited from https://github.com/hamdanal/rich-argparse/issues/78#issuecomment-1627395697
    """
    highlights: ClassVar[list[str]] = RichHelpFormatter.highlights + [r"\*\*(?P<syntax>[^*\n]+)\*\*"]

    def _rich_split_lines(self, text: Text, width: int) -> Lines:
        lines = Lines()
        for line in text.split():
            lines.extend(line.wrap(self.console, width))
        return lines

    def _rich_fill_text(self, text: Text, width: int, indent: Text) -> Text:
        lines = self._rich_split_lines(text, width)
        return Text("\n").join(indent + line for line in lines) + "\n"


class ArgumentDefaultsWrappedTextRichHelpFormatter(ArgumentDefaultsHelpFormatter, WrappedTextRichHelpFormatter):
    """Combines ArgumentDefaultsHelpFormatter with WrappedTextRichHelpFormatter."""
    pass


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false)")


def add_expression_args_group(parser, exp_name: str):
    """Add an argument group for the specified expression to the parser.

    For boolean arguments, uses ``store_true``/``store_false`` actions
    instead of a typed value, based on the argument's default.

    Args:
        parser: The argument parser to add the group to.
        exp_name (str): The registered expression name.
    """
    exp_info = getExpressionLoader(exp_name).expression_info
    group = parser.add_argument_group(f"[{exp_name.upper()}] {exp_info} Expression")
    for arg_name, arg in getExpressionLoader(exp_name).get_args_dict().items():
        if arg.type is bool:
            # TODO: Manually convert str to bool, might introduce bugs in future 
            group.add_argument(f"--{exp_name}.{arg_name}",
                                type=str2bool, default=arg.default, help=arg.help,
                                choices=[True, False])
        else:
            group.add_argument(f"--{exp_name}.{arg_name}",
                                type=arg.type, default=arg.default, help=arg.help,
                                choices=arg.choices)
