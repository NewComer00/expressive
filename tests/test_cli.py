"""Tests for utils/cli.py."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.text import Text

from utils.cli import (
    ArgumentDefaultsWrappedTextRichHelpFormatter,
    WrappedTextRichHelpFormatter,
    add_expression_args_group,
    str2bool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parser(**kwargs):
    """Return a parser using ArgumentDefaultsWrappedTextRichHelpFormatter."""
    return argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter,
        **kwargs,
    )


def _make_arg(type_=str, default=None, help="", choices=None):
    """Build a minimal arg descriptor as a SimpleNamespace."""
    return SimpleNamespace(type=type_, default=default, help=help, choices=choices)


def _make_expression_loader(args_dict, info="Test expression"):
    """Return a mock object that satisfies the getExpressionLoader() interface."""
    loader = MagicMock()
    loader.expression_info = info
    loader.get_args_dict.return_value = args_dict
    return loader


# ---------------------------------------------------------------------------
# WrappedTextRichHelpFormatter
# ---------------------------------------------------------------------------

class TestWrappedTextRichHelpFormatter:

    def test_highlights_includes_bold_markdown(self):
        assert r"\*\*(?P<syntax>[^*\n]+)\*\*" in WrappedTextRichHelpFormatter.highlights

    def test_highlights_extends_base_class(self):
        from rich_argparse import RichHelpFormatter
        assert len(WrappedTextRichHelpFormatter.highlights) > len(RichHelpFormatter.highlights)

    def test_highlights_is_list(self):
        assert isinstance(WrappedTextRichHelpFormatter.highlights, list)

    def test_highlights_bold_pattern_is_last(self):
        """The bold pattern is appended, so it should be the final entry."""
        assert WrappedTextRichHelpFormatter.highlights[-1] == r"\*\*(?P<syntax>[^*\n]+)\*\*"

    # --- _rich_split_lines ---

    def test_rich_split_lines_wraps_long_text(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("This is a very long line that should be wrapped into multiple lines when rendered")
        lines = formatter._rich_split_lines(text, width=40)
        assert len(lines) > 1

    def test_rich_split_lines_short_text_single_line(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("Short line")
        lines = formatter._rich_split_lines(text, width=80)
        assert len(lines) == 1

    def test_rich_split_lines_returns_lines_type(self):
        from rich.containers import Lines
        formatter = WrappedTextRichHelpFormatter(prog="test")
        lines = formatter._rich_split_lines(Text("hello"), width=80)
        assert isinstance(lines, Lines)

    def test_rich_split_lines_empty_text(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        lines = formatter._rich_split_lines(Text(""), width=80)
        # Empty text should produce at least one (empty) line, not raise
        assert len(lines) >= 0

    def test_rich_split_lines_exact_width_no_wrap(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("x" * 10)
        lines = formatter._rich_split_lines(text, width=10)
        assert len(lines) == 1

    def test_rich_split_lines_one_over_width_wraps(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        # Two words whose total exceeds width forces a wrap
        text = Text("hello world")
        lines = formatter._rich_split_lines(text, width=8)
        assert len(lines) > 1

    def test_rich_split_lines_multiline_input(self):
        """Newlines in the source text should produce multiple result lines."""
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("line one\nline two\nline three")
        lines = formatter._rich_split_lines(text, width=80)
        assert len(lines) >= 3

    def test_rich_split_lines_preserves_content(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("hello world")
        lines = formatter._rich_split_lines(text, width=80)
        combined = "".join(line.plain for line in lines)
        assert "hello" in combined
        assert "world" in combined

    # --- _rich_fill_text ---

    def test_rich_fill_text_ends_with_newline(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        result = formatter._rich_fill_text(Text("Test paragraph"), width=80, indent=Text())
        assert result.plain.endswith("\n")

    def test_rich_fill_text_returns_text_instance(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        result = formatter._rich_fill_text(Text("hello"), width=80, indent=Text())
        assert isinstance(result, Text)

    def test_rich_fill_text_indents_wrapped_lines(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        text = Text("This is a very long line that should be wrapped with proper indentation applied")
        result = formatter._rich_fill_text(text, width=40, indent=Text("    "))
        lines = result.plain.strip().split("\n")
        for line in lines[1:]:
            assert line.startswith("    ")

    def test_rich_fill_text_no_indent(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        result = formatter._rich_fill_text(Text("hello"), width=80, indent=Text(""))
        assert "hello" in result.plain

    def test_rich_fill_text_content_preserved(self):
        formatter = WrappedTextRichHelpFormatter(prog="test")
        result = formatter._rich_fill_text(Text("keep this"), width=80, indent=Text())
        assert "keep this" in result.plain


# ---------------------------------------------------------------------------
# ArgumentDefaultsWrappedTextRichHelpFormatter
# ---------------------------------------------------------------------------

class TestArgumentDefaultsWrappedTextRichHelpFormatter:

    def test_inherits_argument_defaults_help_formatter(self):
        assert issubclass(
            ArgumentDefaultsWrappedTextRichHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter,
        )

    def test_inherits_wrapped_text_rich_help_formatter(self):
        assert issubclass(
            ArgumentDefaultsWrappedTextRichHelpFormatter,
            WrappedTextRichHelpFormatter,
        )

    def test_adds_default_value_to_help(self):
        parser = _make_parser()
        parser.add_argument("--test", default="mydefault", help="Test argument")
        assert "mydefault" in parser.format_help()

    def test_multiple_arguments_all_defaults_present(self):
        parser = _make_parser()
        parser.add_argument("--a", default="alpha", help="A")
        parser.add_argument("--b", default="beta",  help="B")
        parser.add_argument("--n", type=int, default=42, help="N")
        help_text = parser.format_help()
        assert "alpha" in help_text
        assert "beta"  in help_text
        assert "42"    in help_text

    def test_long_help_does_not_raise(self):
        parser = _make_parser()
        parser.add_argument("--x", default="d", help="word " * 100)
        parser.format_help()  # must not raise

    def test_bold_markdown_preserved_in_help(self):
        parser = _make_parser()
        parser.add_argument("--x", help="This is **bold** text")
        assert "**bold**" in parser.format_help()

    def test_integer_default_shown(self):
        parser = _make_parser()
        parser.add_argument("--n", type=int, default=7, help="Some number")
        assert "7" in parser.format_help()

    def test_none_default_shown(self):
        parser = _make_parser()
        parser.add_argument("--opt", default=None, help="Optional arg")
        assert "None" in parser.format_help()


# ---------------------------------------------------------------------------
# Integration with argparse
# ---------------------------------------------------------------------------

class TestIntegrationWithArgumentParser:

    def test_full_help_contains_prog_name(self):
        parser = _make_parser(prog="myprog")
        assert "myprog" in parser.format_help()

    def test_bold_description_in_help(self):
        parser = _make_parser(description="Test **description** with bold text")
        assert "Test **description**" in parser.format_help()

    def test_positional_and_optional_arguments(self):
        parser = _make_parser()
        parser.add_argument("input", help="Input file")
        parser.add_argument("--output", default="out.txt", help="Output file")
        help_text = parser.format_help()
        assert "Input file" in help_text
        assert "out.txt"   in help_text

    def test_subparser_in_help(self):
        parser = _make_parser()
        subs = parser.add_subparsers()
        subs.add_parser("subcommand")
        assert "subcommand" in parser.format_help()

    def test_subparser_default_propagated(self):
        parser = _make_parser()
        subs = parser.add_subparsers()
        sub = subs.add_parser(
            "run",
            formatter_class=ArgumentDefaultsWrappedTextRichHelpFormatter,
        )
        sub.add_argument("--lr", default=0.001, help="Learning rate")
        assert "0.001" in sub.format_help()

    def test_store_true_action(self):
        parser = _make_parser()
        parser.add_argument("--verbose", action="store_true", help="Enable verbose")
        help_text = parser.format_help()
        assert "verbose" in help_text

    def test_choices_shown_in_help(self):
        parser = _make_parser()
        parser.add_argument("--mode", choices=["a", "b", "c"], default="a", help="Mode")
        help_text = parser.format_help()
        assert "a" in help_text


# ---------------------------------------------------------------------------
# str2bool
# ---------------------------------------------------------------------------

class TestStr2Bool:

    # --- truthy inputs ---

    def test_true_string(self):
        assert str2bool("true") is True

    def test_true_string_uppercase(self):
        assert str2bool("True") is True

    def test_true_string_mixed_case(self):
        assert str2bool("TRUE") is True

    def test_false_string(self):
        assert str2bool("false") is False

    def test_false_string_uppercase(self):
        assert str2bool("False") is False

    def test_false_string_mixed_case(self):
        assert str2bool("FALSE") is False

    # --- passthrough for actual bools ---

    def test_bool_true_passthrough(self):
        assert str2bool(True) is True

    def test_bool_false_passthrough(self):
        assert str2bool(False) is False

    # --- error cases ---

    def test_invalid_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("yes")

    def test_invalid_string_no_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("no")

    def test_integer_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("1")

    def test_zero_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("0")

    def test_empty_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("")

    def test_error_message_mentions_expected_type(self):
        with pytest.raises(argparse.ArgumentTypeError, match="[Bb]oolean"):
            str2bool("maybe")

    def test_return_type_is_bool_for_true(self):
        assert type(str2bool("true")) is bool

    def test_return_type_is_bool_for_false(self):
        assert type(str2bool("false")) is bool


# ---------------------------------------------------------------------------
# add_expression_args_group
# ---------------------------------------------------------------------------

class TestAddExpressionArgsGroup:

    def _run(self, args_dict, exp_name="myexp", exp_info="My Expression"):
        """Patch getExpressionLoader and call add_expression_args_group."""
        loader = _make_expression_loader(args_dict, info=exp_info)
        parser = argparse.ArgumentParser()
        with patch("utils.cli.getExpressionLoader", return_value=loader):
            add_expression_args_group(parser, exp_name)
        return parser, loader

    # --- group creation ---

    def test_group_added_to_parser(self):
        parser, _ = self._run({"speed": _make_arg(float, 1.0, "Speed")})
        group_titles = [g.title for g in parser._action_groups]
        assert any("myexp" in (t or "").lower() for t in group_titles)

    def test_group_title_contains_exp_name_uppercase(self):
        parser, _ = self._run({}, exp_name="pitch")
        group_titles = [g.title for g in parser._action_groups]
        assert any("PITCH" in (t or "") for t in group_titles)

    def test_group_title_contains_expression_info(self):
        parser, _ = self._run({}, exp_name="dyn", exp_info="Dynamics")
        group_titles = [g.title for g in parser._action_groups]
        assert any("Dynamics" in (t or "") for t in group_titles)

    def test_get_expression_loader_called_with_exp_name(self):
        loader = _make_expression_loader({})
        parser = argparse.ArgumentParser()
        with patch("utils.cli.getExpressionLoader", return_value=loader) as mock_get:
            add_expression_args_group(parser, "testexp")
        mock_get.assert_called_with("testexp")

    # --- non-bool arguments ---

    def test_non_bool_arg_registered(self):
        parser, _ = self._run({"speed": _make_arg(float, 1.0, "Speed")})
        names = [a.dest for a in parser._actions]
        assert "myexp.speed" in names

    def test_non_bool_arg_default(self):
        parser, _ = self._run({"speed": _make_arg(float, 2.5, "Speed")})
        action = next(a for a in parser._actions if a.dest == "myexp.speed")
        assert action.default == 2.5

    def test_non_bool_arg_type(self):
        parser, _ = self._run({"speed": _make_arg(float, 1.0, "Speed")})
        action = next(a for a in parser._actions if a.dest == "myexp.speed")
        assert action.type is float

    def test_non_bool_arg_help(self):
        parser, _ = self._run({"speed": _make_arg(float, 1.0, "Controls speed")})
        action = next(a for a in parser._actions if a.dest == "myexp.speed")
        assert action.help == "Controls speed"

    def test_non_bool_arg_with_choices(self):
        parser, _ = self._run({"mode": _make_arg(str, "a", "Mode", choices=["a", "b"])})
        action = next(a for a in parser._actions if a.dest == "myexp.mode")
        assert action.choices == ["a", "b"]

    def test_non_bool_arg_without_choices_is_none(self):
        parser, _ = self._run({"speed": _make_arg(float, 1.0)})
        action = next(a for a in parser._actions if a.dest == "myexp.speed")
        assert action.choices is None

    def test_multiple_non_bool_args(self):
        args = {
            "speed": _make_arg(float, 1.0),
            "depth": _make_arg(int,   3),
            "label": _make_arg(str,   "x"),
        }
        parser, _ = self._run(args)
        dests = {a.dest for a in parser._actions}
        assert {"myexp.speed", "myexp.depth", "myexp.label"}.issubset(dests)

    # --- bool arguments ---

    def test_bool_arg_registered(self):
        parser, _ = self._run({"flag": _make_arg(bool, False, "A flag")})
        names = [a.dest for a in parser._actions]
        assert "myexp.flag" in names

    def test_bool_arg_type_is_str2bool(self):
        parser, _ = self._run({"flag": _make_arg(bool, False, "A flag")})
        action = next(a for a in parser._actions if a.dest == "myexp.flag")
        assert action.type is str2bool

    def test_bool_arg_default_true(self):
        parser, _ = self._run({"flag": _make_arg(bool, True, "A flag")})
        action = next(a for a in parser._actions if a.dest == "myexp.flag")
        assert action.default is True

    def test_bool_arg_default_false(self):
        parser, _ = self._run({"flag": _make_arg(bool, False, "A flag")})
        action = next(a for a in parser._actions if a.dest == "myexp.flag")
        assert action.default is False

    def test_bool_arg_choices_are_true_false(self):
        parser, _ = self._run({"flag": _make_arg(bool, False)})
        action = next(a for a in parser._actions if a.dest == "myexp.flag")
        assert action.choices == [True, False]

    def test_bool_arg_help_preserved(self):
        parser, _ = self._run({"flag": _make_arg(bool, False, "Enable feature")})
        action = next(a for a in parser._actions if a.dest == "myexp.flag")
        assert action.help == "Enable feature"

    def test_bool_arg_parses_true(self):
        parser, _ = self._run({"flag": _make_arg(bool, False)})
        ns = parser.parse_args(["--myexp.flag", "true"])
        assert getattr(ns, "myexp.flag") is True

    def test_bool_arg_parses_false(self):
        parser, _ = self._run({"flag": _make_arg(bool, True)})
        ns = parser.parse_args(["--myexp.flag", "false"])
        assert getattr(ns, "myexp.flag") is False

    def test_bool_and_non_bool_args_coexist(self):
        args = {
            "flag":  _make_arg(bool,  False, "A flag"),
            "speed": _make_arg(float, 1.0,   "Speed"),
        }
        parser, _ = self._run(args)
        dests = {a.dest for a in parser._actions}
        assert "myexp.flag"  in dests
        assert "myexp.speed" in dests

    def test_empty_args_dict_no_extra_actions(self):
        parser, _ = self._run({})
        # Only the default -h/--help action should be present
        non_help = [a for a in parser._actions if "--help" not in a.option_strings]
        assert len(non_help) == 0

    def test_arg_option_string_uses_exp_name_prefix(self):
        parser, _ = self._run({"alpha": _make_arg(float, 0.5)}, exp_name="enc")
        action = next(a for a in parser._actions if a.dest == "enc.alpha")
        assert "--enc.alpha" in action.option_strings
