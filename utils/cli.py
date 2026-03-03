from typing import ClassVar
from argparse import ArgumentDefaultsHelpFormatter

from rich.text import Text
from rich.containers import Lines
from rich_argparse import RichHelpFormatter


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
