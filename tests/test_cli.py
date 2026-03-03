"""Tests for CLI utilities."""

import argparse

from rich.text import Text

from utils.cli import (
    ArgumentDefaultsWrappedTextRichHelpFormatter,
    WrappedTextRichHelpFormatter,
)


class TestWrappedTextRichHelpFormatter:
    """Tests for WrappedTextRichHelpFormatter."""

    def test_highlights_includes_bold_markdown(self):
        """Test that highlights includes pattern for **bold** markdown."""
        assert r"\*\*(?P<syntax>[^*\n]+)\*\*" in WrappedTextRichHelpFormatter.highlights

    def test_highlights_inherits_from_rich_help_formatter(self):
        """Test that highlights extends RichHelpFormatter's defaults."""
        from rich_argparse import RichHelpFormatter
        # Our highlights should be longer than the base class
        assert len(WrappedTextRichHelpFormatter.highlights) > len(RichHelpFormatter.highlights)

    def test_rich_split_lines_wraps_long_text(self):
        """Test that _rich_split_lines properly wraps long lines."""
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("This is a very long line that should be wrapped into multiple lines when rendered")
        lines = formatter._rich_split_lines(text, width=40)
        # Should produce multiple lines due to wrapping
        assert len(lines) > 1

    def test_rich_split_lines_preserves_short_lines(self):
        """Test that short lines are not unnecessarily split."""
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("Short line")
        lines = formatter._rich_split_lines(text, width=80)
        assert len(lines) == 1

    def test_rich_fill_text_adds_newline(self):
        """Test that _rich_fill_text appends a newline."""
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("Test paragraph")
        result = formatter._rich_fill_text(text, width=80, indent=Text())
        assert result.plain.endswith("\n")

    def test_rich_fill_text_with_indent(self):
        """Test that _rich_fill_text properly indents wrapped lines."""
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("This is a very long line that should be wrapped with proper indentation")
        indent = Text("    ")  # 4 spaces
        result = formatter._rich_fill_text(text, width=40, indent=indent)
        # Check that wrapped lines are indented
        lines = result.plain.strip().split("\n")
        for line in lines[1:]:  # Skip first line
            assert line.startswith("    ")


class TestArgumentDefaultsWrappedTextRichHelpFormatter:
    """Tests for ArgumentDefaultsWrappedTextRichHelpFormatter."""

    def test_inherits_from_argument_defaults_help_formatter(self):
        """Test that the class inherits from ArgumentDefaultsHelpFormatter."""
        assert issubclass(
            ArgumentDefaultsWrappedTextRichHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
        )

    def test_inherits_from_wrapped_text_rich_help_formatter(self):
        """Test that the class inherits from WrappedTextRichHelpFormatter."""
        assert issubclass(
            ArgumentDefaultsWrappedTextRichHelpFormatter, WrappedTextRichHelpFormatter
        )

    def test_adds_default_values_to_help(self):
        """Test that default values are added to help text."""
        parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter
        )
        parser.add_argument("--test", default="default_value", help="Test argument")

        # Capture help output
        help_text = parser.format_help()

        # Should contain the default value
        assert "default_value" in help_text

    def test_handles_long_help_text_with_wrapping(self):
        """Test that long help text is properly wrapped."""
        parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter
        )
        long_help = "This is a very long help text that should be wrapped " * 5
        parser.add_argument("--test", default="default", help=long_help)

        # Should not raise an exception
        help_text = parser.format_help()
        assert "default" in help_text

    def test_handles_multiple_arguments(self):
        """Test formatting with multiple arguments."""
        parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter
        )
        parser.add_argument("--arg1", default="val1", help="First argument")
        parser.add_argument("--arg2", default="val2", help="Second argument")
        parser.add_argument("--arg3", type=int, default=42, help="Third argument")

        help_text = parser.format_help()

        assert "val1" in help_text
        assert "val2" in help_text
        assert "42" in help_text

    def test_preserves_bold_markdown_in_help(self):
        """Test that **bold** markdown syntax is preserved in help text."""
        parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter
        )
        parser.add_argument("--test", help="This is **bold** text")

        help_text = parser.format_help()
        assert "**bold**" in help_text


class TestIntegrationWithArgumentParser:
    """Integration tests with argparse."""

    def test_full_help_output(self):
        """Test complete help output formatting."""
        parser = argparse.ArgumentParser(
            prog="test_prog",
            description="Test **description** with bold text",
            formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        parser.add_argument(
            "--output",
            default="output.txt",
            help="Output file path"
        )

        help_text = parser.format_help()

        # Check structure
        assert "test_prog" in help_text
        assert "Test **description**" in help_text
        assert "output.txt" in help_text

    def test_subparser_compatibility(self):
        """Test that formatter works with subparsers."""
        parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter
        )
        subparsers = parser.add_subparsers()
        sub = subparsers.add_parser("subcommand")
        sub.add_argument("--sub-arg", default="sub-default", help="Subcommand argument")

        # Should not raise
        help_text = parser.format_help()
        assert "subcommand" in help_text

    def test_positional_arguments(self):
        """Test formatting with positional arguments."""
        parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter
        )
        parser.add_argument("input", help="Input file")
        parser.add_argument("--output", default="out.txt", help="Output file")

        help_text = parser.format_help()
        assert "Input file" in help_text
        assert "out.txt" in help_text
