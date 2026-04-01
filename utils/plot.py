"""
utils/plot.py — declarative plot descriptors and figure helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import plotly.graph_objects as go


@dataclass
class Plot:
    """Declarative plot descriptor — mirrors Args in style.

    ``legends`` declares the expected series names in order. ``fig()`` accepts
    one positional ``(x, y)`` tuple per legend, keeping call sites clean and
    enforcing the declared structure.

    Usage in a loader::

        plots = SimpleNamespace(
            raw_rms     = Plot(tag="raw_rms",     title="Raw RMS",     x_label="Time (s)", y_label="RMS", legends=["UTAU", "Reference"]),
            aligned_rms = Plot(tag="aligned_rms", title="Aligned RMS", x_label="Tick",     y_label="RMS", legends=["UTAU", "Reference"]),
            expression  = Plot(tag="expression",  title="Expression",  x_label="Tick",     y_label="dyn", legends=["dyn"]),
        )

        self.collect_plot(p.raw_rms, (utau_time, utau_rms), (ref_time, ref_rms))
    """  # noqa: E501
    tag:     str
    title:   str
    x_label: str
    y_label: str
    legends: list[str] = field(default_factory=list)

    def fig(self, *series: tuple) -> dict:
        """Return a dict suitable for ``collector.register(**...)``.

        Positional args are ``(x, y)`` tuples, matched to ``legends`` in order.

        Raises:
            ValueError: if the number of series doesn't match ``legends``.
        """
        if len(series) != len(self.legends):
            raise ValueError(
                f"Plot {self.tag!r} expects {len(self.legends)} series "
                f"({self.legends}), got {len(series)}."
            )
        named = dict(zip([str(lg) for lg in self.legends], series, strict=False))
        return dict(tag=str(self.tag), data=_lines(str(self.title), str(self.x_label), str(self.y_label), named))


def _lines(title: str, x_label: str, y_label: str, series: dict[str, tuple]) -> go.Figure:
    """Create a plotly line figure from a dict of named (x, y) series."""
    return go.Figure(
        data=[go.Scatter(x=xd, y=yd, name=name) for name, (xd, yd) in series.items()],
        layout=dict(title=title, xaxis_title=x_label, yaxis_title=y_label),
    )
