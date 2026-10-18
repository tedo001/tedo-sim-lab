"""The widget kit every page is built from."""

from .basics import ElidedLabel, KeyValues, PathLabel, Pill, label, repolish
from .charts import MeterBar, TimeSeriesChart
from .confusion import ConfusionMatrixView
from .curves import EpochChart
from .layout import Card, FlowGrid, Page, PageHead, ResponsiveRow
from .segmented import Segmented
from .stats import StatStrip, StatTile
from .table import DataTable

__all__ = ["Card", "ConfusionMatrixView", "DataTable", "ElidedLabel", "EpochChart", "FlowGrid", "KeyValues",
           "MeterBar", "Page", "PageHead", "PathLabel", "Pill", "ResponsiveRow", "Segmented", "StatStrip",
           "StatTile", "TimeSeriesChart", "label", "repolish"]
