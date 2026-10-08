"""The widget kit every page is built from."""

from .basics import ElidedLabel, KeyValues, PathLabel, Pill, label, repolish
from .charts import MeterBar, TimeSeriesChart
from .confusion import ConfusionMatrixView
from .curves import EpochChart
from .layout import Card, Page, PageHead, ResponsiveRow
from .stats import StatStrip, StatTile
from .table import DataTable

__all__ = ["Card", "ConfusionMatrixView", "DataTable", "ElidedLabel", "EpochChart", "KeyValues", "MeterBar",
           "Page", "PageHead",
           "PathLabel",
           "Pill", "ResponsiveRow", "StatStrip", "StatTile", "TimeSeriesChart", "label", "repolish"]
