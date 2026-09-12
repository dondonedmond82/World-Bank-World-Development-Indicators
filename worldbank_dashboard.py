"""
World Bank Development Indicators Dashboard, PDF Generator & ML Lab
=====================================================================
An interactive Panel dashboard for exploring World Bank development
indicators across countries and years, combined with:
  - An automated PDF generator (executive summary charts + commentary)
  - A Machine Learning tab: Linear Regression, Perceptron, Logistic
    Regression classification, K-Means clustering, and Hierarchical
    (agglomerative) clustering — each with its own visualization.

Run with:
    panel serve worldbank_dashboard.py --show
"""

import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas  # noqa: F401

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
)

warnings.filterwarnings("ignore")
pn.extension("tabulator", "echarts", notifications=True)

# ==================================================================
# CONFIG & PALETTE
# ==================================================================
FILE_PATH = Path("./data/world_bank_development_indicators.csv")
LOGO_PATH = Path("./logo.png")

FEATURED_INDICATORS = [
    "GDP_current_US",
    "population",
    "life_expectancy_at_birth",
    "CO2_emisions",
    "inflation_annual%",
    "individuals_using_internet%",
    "access_to_electricity%",
    "agricultural_land%",
    "forest_land%",
    "gini_index",
]

NON_COUNTRY_ENTITIES = [
    "World", "Africa Eastern and Southern", "Africa Western and Central",
    "Arab World", "Central Europe and the Baltics", "Caribbean small states",
    "East Asia & Pacific", "East Asia & Pacific (excluding high income)",
    "Euro area", "Europe & Central Asia", "Europe & Central Asia (excluding high income)",
    "European Union", "Fragile and conflict affected situations",
    "Heavily indebted poor countries (HIPC)", "High income", "IBRD only",
    "IDA & IBRD total", "IDA blend", "IDA only", "IDA total",
    "Latin America & Caribbean", "Latin America & Caribbean (excluding high income)",
    "Least developed countries: UN classification", "Low & middle income",
    "Low income", "Lower middle income", "Middle East & North Africa",
    "Middle East & North Africa (excluding high income)", "Middle income",
    "North America", "OECD members", "Other small states",
    "Pacific island small states", "Post-demographic dividend",
    "Pre-demographic dividend", "Small states", "South Asia",
    "Sub-Saharan Africa", "Sub-Saharan Africa (excluding high income)",
    "Upper middle income", "Late-demographic dividend", "Early-demographic dividend",
]

COLOR_PRIMARY = "#1B5E20"
COLOR_ACCENT = "#2E7D32"
COLOR_NEUTRAL = "#37474F"
COLOR_ALERT = "#C62828"
COLOR_BG = "#FFFFFF"

PIE_PALETTE = [
    "#1B5E20", "#2E7D32", "#43A047",
    "#66BB6A", "#81C784", "#A5D6A7",
    COLOR_NEUTRAL
]

CLUSTER_PALETTE = [
    "#1B5E20", "#F9A825", "#1565C0", "#C62828",
    "#6A1B9A", "#00838F", "#EF6C00", "#4E342E"
]

MONEY_KEYWORDS = ["GDP", "debt", "expense"]


# ==================================================================
# DATA LOADING
# ==================================================================
@pn.cache
def load_data(path: Path) -> pd.DataFrame:
    """Load dataset or fallback to mock structure if file missing."""
    if path.exists():
        data = pd.read_csv(path)
    else:
        np.random.seed(42)

        countries = [
            "United States", "Germany", "India", "Brazil", "China",
            "Nigeria", "Japan", "France", "Kenya", "Mexico"
        ]

        years = list(range(1990, 2024))
        rows = []

        for c in countries:
            base_gdp = np.random.uniform(1e11, 2e13)

            for y in years:
                rows.append({
                    "country": c,
                    "date": f"{y}-01-01",
                    "GDP_current_US":
                        base_gdp *
                        (1.03 ** (y - 1990)) *
                        np.random.uniform(0.95, 1.05),
                    "population": np.random.uniform(5e6, 1.4e9),
                    "life_expectancy_at_birth": np.random.uniform(55, 84),
                    "CO2_emisions": np.random.uniform(0.5, 20),
                    "inflation_annual%": np.random.uniform(-2, 15),
                    "individuals_using_internet%": np.random.uniform(0, 95),
                    "access_to_electricity%": np.random.uniform(20, 100),
                    "agricultural_land%": np.random.uniform(5, 60),
                    "forest_land%": np.random.uniform(2, 60),
                    "gini_index": np.random.uniform(25, 60),
                })

        data = pd.DataFrame(rows)

    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year

    return data


df = load_data(FILE_PATH)

NUMERIC_COLUMNS = sorted(
    c for c in df.select_dtypes("number").columns
    if c != "year"
)

INDICATOR_OPTIONS = (
    [c for c in FEATURED_INDICATORS if c in NUMERIC_COLUMNS]
    + [c for c in NUMERIC_COLUMNS if c not in FEATURED_INDICATORS]
)

COUNTRY_OPTIONS = sorted(
    c for c in df["country"].dropna().unique()
    if c not in NON_COUNTRY_ENTITIES
)

YEAR_MIN = int(df["year"].min())
YEAR_MAX = int(df["year"].max())

DEFAULT_COUNTRY = (
    "United States"
    if "United States" in COUNTRY_OPTIONS
    else COUNTRY_OPTIONS[0]
)


# ==================================================================
# SMALL DISPLAY HELPERS
# ==================================================================
def human_format(value: float, indicator: str = "") -> str:
    """
    Format numerical values for dashboard display.

    Examples:
        25440000000000 -> $25.44T
        543300000000 -> $543.30B
        82.345 -> 82.35%
    """

    if pd.isna(value):
        return "N/A"

    value = float(value)

    sign = "-" if value < 0 else ""
    value = abs(value)

    is_money = any(
        k.lower() in indicator.lower()
        for k in MONEY_KEYWORDS
    )

    is_pct = indicator.endswith("%")

    if is_pct:
        return f"{sign}{value:,.2f}%"

    prefix = "$" if is_money else ""

    for unit, div in (
        ("T", 1e12),
        ("B", 1e9),
        ("M", 1e6),
        ("K", 1e3),
    ):
        if value >= div:
            return f"{sign}{prefix}{value / div:,.2f}{unit}"

    return f"{sign}{prefix}{value:,.2f}"


def pretty_label(col: str) -> str:
    return (
        col
        .replace("_", " ")
        .replace("%", " %")
        .strip()
        .title()
    )


# ==================================================================
# HORIZONTAL KPI BAR
# ==================================================================
def kpi_html_card(
    label: str,
    value: str,
    color: str = COLOR_NEUTRAL,
    border_right: bool = True
) -> str:
    """
    Generate one KPI section inside the single horizontal KPI bar.
    """

    border = (
        "border-right:1px solid #E0E4E1;"
        if border_right
        else ""
    )

    return f"""
        <div style="
            flex:1 1 25%;
            min-width:220px;
            box-sizing:border-box;
            padding:10px 18px;
            text-align:center;
            {border}
        ">
            <div style="
                font-size:11px;
                font-weight:700;
                color:#78909C;
                letter-spacing:.05em;
                text-transform:uppercase;
                white-space:nowrap;
            ">
                {label}
            </div>

            <div style="
                font-size:22px;
                line-height:1.2;
                font-weight:700;
                color:{color};
                margin-top:5px;
                white-space:nowrap;
            ">
                {value}
            </div>
        </div>
    """


def mpl_pane(fig, height=480):
    pane = pn.pane.Matplotlib(
        fig,
        tight=True,
        format="svg",
        height=height,
        sizing_mode="stretch_width"
    )

    plt.close(fig)

    return pane


def desc_pane(
    markdown_text: str,
    width: int = 300
) -> pn.pane.Markdown:
    """Small descriptive text block placed beside a chart."""

    return pn.pane.Markdown(
        markdown_text,
        width=width,
        sizing_mode="fixed",
        margin=(10, 10, 10, 25),
        styles={
            "background": "#F5F7F5",
            "border": "1px solid #E0E4E1",
            "border-left": f"4px solid {COLOR_PRIMARY}",
            "border-radius": "6px",
            "padding": "12px 14px",
        },
    )


def chart_page(
    chart,
    description: str,
    chart_width_policy: str = "stretch_width"
) -> pn.Row:

    return pn.Row(
        pn.Column(
            chart,
            sizing_mode=chart_width_policy
        ),
        desc_pane(description),
        sizing_mode="stretch_width",
    )


def empty_fig(message, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=10,
        color=COLOR_NEUTRAL
    )

    ax.axis("off")

    return fig


# ==================================================================
# GLOBAL FILTER WIDGETS
# ==================================================================
select_indicator = pn.widgets.Select(
    name="Indicator",
    options=INDICATOR_OPTIONS,
    value=INDICATOR_OPTIONS[0],
    sizing_mode="stretch_width"
)

select_country = pn.widgets.Select(
    name="Country",
    options=COUNTRY_OPTIONS,
    value=DEFAULT_COUNTRY,
    sizing_mode="stretch_width"
)

year_range = pn.widgets.IntRangeSlider(
    name="Year Range",
    start=YEAR_MIN,
    end=YEAR_MAX,
    value=(YEAR_MIN, YEAR_MAX),
    step=1,
    sizing_mode="stretch_width"
)

reset_button = pn.widgets.Button(
    name="↺ Reset Filters",
    button_type="default",
    sizing_mode="stretch_width"
)


def reset_filters(event=None):
    select_indicator.value = INDICATOR_OPTIONS[0]
    select_country.value = DEFAULT_COUNTRY
    year_range.value = (YEAR_MIN, YEAR_MAX)


reset_button.on_click(reset_filters)


def get_filtered_data() -> pd.DataFrame:
    data = df.copy()

    data = data[
        (data["year"] >= year_range.value[0])
        & (data["year"] <= year_range.value[1])
    ]

    return data


def get_country_series(indicator: str) -> pd.DataFrame:
    data = get_filtered_data()

    data = data[
        data["country"] == select_country.value
    ]

    return (
        data[["year", indicator]]
        .dropna()
        .sort_values("year")
    )


def latest_year_with_data(
    data: pd.DataFrame,
    indicator: str
):
    """Some indicators stop reporting before dataset's max year."""

    valid = data.dropna(subset=[indicator])

    if valid.empty:
        return None

    return int(valid["year"].max())


def cross_country_snapshot(
    columns,
    top_n=None,
    sort_by=None,
    min_rows=5
):
    """
    Most recent year within filtered range with enough complete rows
    across requested columns, for real countries only.
    """

    data = get_filtered_data()

    data = data[
        ~data["country"].isin(NON_COUNTRY_ENTITIES)
    ]

    years_desc = sorted(
        data["year"].dropna().unique(),
        reverse=True
    )

    snap = pd.DataFrame(
        columns=["country"] + columns
    )

    used_year = None

    for yr in years_desc:

        candidate = (
            data[data["year"] == yr]
            [["country"] + columns]
            .dropna()
        )

        if len(candidate) >= min_rows:
            snap = candidate
            used_year = int(yr)
            break

    if (
        top_n
        and sort_by in snap.columns
        and not snap.empty
    ):
        snap = (
            snap
            .sort_values(sort_by, ascending=False)
            .head(top_n)
        )

    return snap.reset_index(drop=True), used_year


# ==================================================================
# SUMMARY TAB PANELS
# ==================================================================
SUMMARY_DESCRIPTIONS = {
    "scatter": (
        "**What this shows**\n\n"
        "Every country plotted by the selected indicator against GDP "
        "(or, when GDP is selected, against population), for the latest "
        "year with data.\n\n"
        "Hover over a point to see which country it is. Use it to spot "
        "correlations or outliers between the two measures."
    ),

    "bar": (
        "**What this shows**\n\n"
        "The 10 countries with the highest value of the selected "
        "indicator in the latest year with data.\n\n"
        "A quick way to see who leads the world on this measure."
    ),
}


# ==================================================================
# DYNAMIC SINGLE-LINE KPI BAR
# ==================================================================
@pn.depends(
    select_indicator,
    select_country,
    year_range
)
def kpi_cards(indicator, country, yrange):

    series = get_country_series(indicator)

    if series.empty:

        latest = avg = mn = mx = float("nan")
        latest_year = "—"

    else:

        latest = series[indicator].iloc[-1]
        avg = series[indicator].mean()
        mn = series[indicator].min()
        mx = series[indicator].max()

        latest_year = int(
            series["year"].iloc[-1]
        )

    latest_text = human_format(
        latest,
        indicator
    )

    avg_text = human_format(
        avg,
        indicator
    )

    min_text = human_format(
        mn,
        indicator
    )

    max_text = human_format(
        mx,
        indicator
    )

    html = f"""
    <div style="
        width:100%;
        box-sizing:border-box;
        background:#FFFFFF;
        border:1px solid #E0E4E1;
        border-radius:10px;
        box-shadow:0 1px 3px rgba(0,0,0,0.06);
        overflow-x:auto;
        overflow-y:hidden;
        margin:4px 0 12px 0;
    ">

        <div style="
            display:flex;
            flex-direction:row;
            align-items:stretch;
            width:100%;
            min-width:920px;
            min-height:82px;
        ">

            {kpi_html_card(
                f"Latest ({latest_year})",
                latest_text,
                COLOR_ACCENT,
                True
            )}

            {kpi_html_card(
                "Average (period)",
                avg_text,
                COLOR_ACCENT,
                True
            )}

            {kpi_html_card(
                "Minimum (period)",
                min_text,
                COLOR_NEUTRAL,
                True
            )}

            {kpi_html_card(
                "Maximum (period)",
                max_text,
                COLOR_NEUTRAL,
                False
            )}

        </div>
    </div>
    """

    return pn.pane.HTML(
        html,
        sizing_mode="stretch_width",
        height=105,
        margin=(0, 0, 5, 0)
    )


# ==================================================================
# SUMMARY — SCATTER PLOT
# ==================================================================
@pn.depends(
    select_indicator,
    year_range
)
def scatter_plot(indicator, yrange):

    """Cross-country relationship between indicator and GDP."""

    data = get_filtered_data()

    y_col = (
        "GDP_current_US"
        if indicator != "GDP_current_US"
        else "population"
    )

    latest_year = latest_year_with_data(
        data,
        indicator
    )

    if latest_year is None:

        return (
            pd.DataFrame({
                "country": [],
                indicator: [],
                y_col: []
            })
            .hvplot.scatter(
                x=indicator,
                y=y_col,
                height=460,
                responsive=True
            )
            .opts(bgcolor=COLOR_BG)
        )

    snap = (
        data[data["year"] == latest_year]
        [["country", indicator, y_col]]
        .dropna()
    )

    return snap.hvplot.scatter(
        x=indicator,
        y=y_col,
        hover_cols=["country"],
        title=(
            f"{pretty_label(indicator)} vs "
            f"{pretty_label(y_col)} ({latest_year})"
        ),
        color=COLOR_ACCENT,
        height=460,
        responsive=True,
    ).opts(bgcolor=COLOR_BG)


# ==================================================================
# SUMMARY — TOP 10 BAR CHART
# ==================================================================
@pn.depends(
    select_indicator,
    year_range
)
def bar_plot(indicator, yrange):

    data = get_filtered_data()

    latest_year = latest_year_with_data(
        data,
        indicator
    )

    if latest_year is None:

        return (
            pd.DataFrame({
                "country": [],
                indicator: []
            })
            .hvplot.barh(
                x="country",
                y=indicator,
                height=460,
                responsive=True
            )
            .opts(bgcolor=COLOR_BG)
        )

    top = (
        data[data["year"] == latest_year]
        [["country", indicator]]
        .dropna()
        .loc[
            lambda d:
            ~d["country"].isin(NON_COUNTRY_ENTITIES)
        ]
        .sort_values(
            indicator,
            ascending=False
        )
        .head(10)
    )

    return top.hvplot.barh(
        x="country",
        y=indicator,
        title=(
            f"Top 10 Countries by "
            f"{pretty_label(indicator)} ({latest_year})"
        ),
        color=COLOR_NEUTRAL,
        height=460,
        responsive=True,
    ).opts(bgcolor=COLOR_BG)


# ==================================================================
# DATA TABLE
# ==================================================================
@pn.depends(
    select_indicator,
    select_country,
    year_range
)
def table_view(indicator, country, yrange):

    data = get_filtered_data()

    data = data[
        data["country"] == country
    ]

    columns = [
        "country",
        "year",
        indicator
    ]

    columns = [
        c for c in columns
        if c in data.columns
    ]

    data = (
        data[columns]
        .sort_values(
            "year",
            ascending=False
        )
        .head(50)
    )

    money_cols = (
        [indicator]
        if any(
            k.lower() in indicator.lower()
            for k in MONEY_KEYWORDS
        )
        else []
    )

    return pn.widgets.Tabulator(
        data,
        pagination="local",
        page_size=10,
        sizing_mode="stretch_width",
        height=420,
        layout="fit_columns",
        show_index=False,
        disabled=True,
        theme="materialize",
        formatters={
            c: {
                "type": "money",
                "symbol": "$",
                "precision": 0
            }
            for c in money_cols
        },
    )


# ==================================================================
# INSIGHTS TAB
# ==================================================================
INSIGHTS_DESCRIPTIONS = {

    "line": (
        "**What this shows**\n\n"
        "The year-by-year trend of the selected indicator for the selected "
        "country across the chosen year range.\n\n"
        "Use it to spot long-term growth, decline, or turning points."
    ),

    "pie": (
        "**What this shows**\n\n"
        "The share of the selected indicator held by the top 6 countries "
        "in the most recent year with available data."
    ),

    "decade": (
        "**What this shows**\n\n"
        "The average value of the selected indicator for the selected "
        "country, grouped by decade."
    ),

    "histogram": (
        "**What this shows**\n\n"
        "The distribution of the selected indicator across all countries "
        "in the most recent year with data."
    ),
}


# ==================================================================
# HISTORICAL LINE
# ==================================================================
@pn.depends(
    select_indicator,
    select_country,
    year_range
)
def line_chart_panel(
    indicator,
    country,
    yrange
):

    series = get_country_series(indicator)

    return series.hvplot.line(
        x="year",
        y=indicator,
        color=COLOR_PRIMARY,
        title=(
            f"{pretty_label(indicator)} "
            f"Trend — {country}"
        ),
        height=440,
        responsive=True
    ).opts(bgcolor=COLOR_BG)


# ==================================================================
# GLOBAL SHARE PIE
# ==================================================================
@pn.depends(
    select_indicator,
    year_range
)
def pie_chart_panel(
    indicator,
    yrange
):

    data = get_filtered_data()

    latest_year = latest_year_with_data(
        data,
        indicator
    )

    if latest_year is None:

        return mpl_pane(
            empty_fig(
                "No data available\nfor this indicator",
                (5, 3.8)
            ),
            height=440
        )

    totals = (
        data[data["year"] == latest_year]
        [["country", indicator]]
        .dropna()
        .loc[
            lambda d:
            ~d["country"].isin(NON_COUNTRY_ENTITIES)
        ]
        .set_index("country")[indicator]
        .sort_values(ascending=False)
        .head(6)
    )

    fig, ax = plt.subplots(
        figsize=(5, 3.8)
    )

    if totals.empty or (totals < 0).any():

        ax.text(
            0.5,
            0.5,
            "No positive data\nfor this indicator/year",
            ha="center",
            va="center"
        )

        ax.axis("off")

    else:

        ax.pie(
            totals.values,
            labels=totals.index,
            autopct="%1.1f%%",
            colors=PIE_PALETTE[:len(totals)]
        )

    ax.set_title(
        f"{pretty_label(indicator)} Share ({latest_year})",
        fontsize=10,
        fontweight="bold"
    )

    return mpl_pane(
        fig,
        height=440
    )


# ==================================================================
# DECADE AVERAGES
# ==================================================================
@pn.depends(
    select_indicator,
    select_country,
    year_range
)
def decade_bar_panel(
    indicator,
    country,
    yrange
):

    series = get_country_series(indicator)

    if series.empty:

        return mpl_pane(
            empty_fig(
                "No data available\nfor this indicator",
                (5.5, 3.8)
            ),
            height=440
        )

    series = series.copy()

    series["decade"] = (
        series["year"] // 10 * 10
    ).astype(str) + "s"

    decade_avg = (
        series
        .groupby("decade")[indicator]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 3.8)
    )

    ax.bar(
        decade_avg.index,
        decade_avg.values,
        color=COLOR_ACCENT,
        width=0.6
    )

    ax.set_title(
        f"{pretty_label(indicator)} — "
        f"Decade Averages ({country})",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Decade",
        fontsize=8
    )

    ax.set_ylabel(
        pretty_label(indicator),
        fontsize=8
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
        axis="y"
    )

    return mpl_pane(
        fig,
        height=440
    )


# ==================================================================
# HISTOGRAM
# ==================================================================
@pn.depends(
    select_indicator,
    year_range
)
def histogram_panel(
    indicator,
    yrange
):

    data = get_filtered_data()

    latest_year = latest_year_with_data(
        data,
        indicator
    )

    if latest_year is None:

        return mpl_pane(
            empty_fig(
                "No data available\nfor this indicator",
                (5.5, 3.8)
            ),
            height=440
        )

    values = (
        data[data["year"] == latest_year]
        [["country", indicator]]
        .dropna()
        .loc[
            lambda d:
            ~d["country"].isin(NON_COUNTRY_ENTITIES)
        ][indicator]
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 3.8)
    )

    ax.hist(
        values,
        bins=20,
        color=COLOR_NEUTRAL,
        edgecolor="white"
    )

    ax.set_title(
        f"{pretty_label(indicator)} Distribution "
        f"Across Countries ({latest_year})",
        fontsize=9.5,
        fontweight="bold"
    )

    ax.set_xlabel(
        pretty_label(indicator),
        fontsize=8
    )

    ax.set_ylabel(
        "Number of Countries",
        fontsize=8
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
        axis="y"
    )

    return mpl_pane(
        fig,
        height=440
    )


# ==================================================================
# MACHINE LEARNING TAB
# ==================================================================
ML_DESCRIPTIONS = {

    "regression": (
        "**Linear Regression**\n\n"
        "Fits a straight line through the selected indicator's history "
        "for the selected country, then projects it 5 years into the future.\n\n"
        "The **R²** value shows how well a straight line explains the trend."
    ),

    "heatmap": (
        "**Correlation Heatmap**\n\n"
        "Shows how strongly each pair of chosen indicators moves together "
        "across all countries and years in range."
    ),

    "kmeans": (
        "**K-Means Clustering**\n\n"
        "Groups countries into *k* clusters based on the two chosen "
        "features after standardization."
    ),

    "hierarchical": (
        "**Hierarchical Clustering (Dendrogram)**\n\n"
        "Builds a tree of countries by repeatedly merging the most similar "
        "countries based on the selected features."
    ),

    "classification": (
        "**Classification (Logistic Regression / Perceptron)**\n\n"
        "Splits countries into above/below median groups for the selected "
        "classification target."
    ),
}


ml_x_feature = pn.widgets.Select(
    name="Feature X",
    options=NUMERIC_COLUMNS,
    value=(
        "GDP_current_US"
        if "GDP_current_US" in NUMERIC_COLUMNS
        else NUMERIC_COLUMNS[0]
    ),
    sizing_mode="stretch_width"
)

ml_y_feature = pn.widgets.Select(
    name="Feature Y",
    options=NUMERIC_COLUMNS,
    value=(
        "life_expectancy_at_birth"
        if "life_expectancy_at_birth" in NUMERIC_COLUMNS
        else NUMERIC_COLUMNS[1]
    ),
    sizing_mode="stretch_width"
)

ml_target_indicator = pn.widgets.Select(
    name="Classification Target",
    options=NUMERIC_COLUMNS,
    value=(
        "CO2_emisions"
        if "CO2_emisions" in NUMERIC_COLUMNS
        else NUMERIC_COLUMNS[2]
    ),
    sizing_mode="stretch_width"
)

ml_classifier_algo = pn.widgets.Select(
    name="Classifier",
    options=[
        "Logistic Regression",
        "Perceptron"
    ],
    value="Logistic Regression",
    sizing_mode="stretch_width"
)

ml_n_clusters = pn.widgets.IntSlider(
    name="K (clusters)",
    start=2,
    end=8,
    value=4,
    step=1,
    sizing_mode="stretch_width"
)

ml_top_n = pn.widgets.IntSlider(
    name="Countries to include",
    start=10,
    end=100,
    value=40,
    step=5,
    sizing_mode="stretch_width"
)

ml_heatmap_features = pn.widgets.MultiChoice(
    name="Heatmap Indicators",
    options=NUMERIC_COLUMNS,
    value=[
        c
        for c in FEATURED_INDICATORS
        if c in NUMERIC_COLUMNS
    ][:8],
    max_items=12,
    sizing_mode="stretch_width"
)


# ==================================================================
# LINEAR REGRESSION
# ==================================================================
@pn.depends(
    select_indicator,
    select_country,
    year_range
)
def linear_regression_panel(
    indicator,
    country,
    yrange
):

    series = get_country_series(indicator)

    if len(series) < 3:

        return mpl_pane(
            empty_fig(
                "Not enough data points\nfor a regression fit",
                (7, 4.2)
            ),
            height=460
        )

    X = series[["year"]].values
    y = series[indicator].values

    model = LinearRegression().fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(
        y,
        y_pred
    )

    future_years = np.arange(
        series["year"].max() + 1,
        series["year"].max() + 6
    ).reshape(-1, 1)

    future_pred = model.predict(
        future_years
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.2)
    )

    ax.scatter(
        series["year"],
        y,
        color=COLOR_NEUTRAL,
        s=22,
        label="Actual",
        zorder=3
    )

    ax.plot(
        series["year"],
        y_pred,
        color=COLOR_ACCENT,
        lw=2.2,
        label="Fitted line"
    )

    ax.plot(
        future_years.flatten(),
        future_pred,
        "--",
        color=COLOR_ALERT,
        lw=2,
        label="5-year projection"
    )

    ax.set_title(
        f"Linear Regression — "
        f"{pretty_label(indicator)} ({country}) | R² = {r2:.3f}",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year",
        fontsize=8.5
    )

    ax.set_ylabel(
        pretty_label(indicator),
        fontsize=8.5
    )

    ax.legend(
        fontsize=8,
        loc="best"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    return mpl_pane(
        fig,
        height=460
    )


# ==================================================================
# CORRELATION HEATMAP
# ==================================================================
@pn.depends(
    ml_heatmap_features,
    year_range
)
def heatmap_panel(
    features,
    yrange
):

    features = [
        f for f in features
        if f in NUMERIC_COLUMNS
    ]

    if len(features) < 2:

        return mpl_pane(
            empty_fig(
                "Select at least 2 indicators\nfor the heatmap",
                (6, 5)
            ),
            height=480
        )

    data = get_filtered_data()[features]

    corr = data.corr()

    fig, ax = plt.subplots(
        figsize=(
            max(5.5, 0.55 * len(features) + 2),
            max(4.5, 0.55 * len(features) + 1.5)
        )
    )

    im = ax.imshow(
        corr.values,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1
    )

    labels = [
        pretty_label(f)
        for f in features
    ]

    ax.set_xticks(
        range(len(features))
    )

    ax.set_yticks(
        range(len(features))
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=7.5
    )

    ax.set_yticklabels(
        labels,
        fontsize=7.5
    )

    for i in range(len(features)):
        for j in range(len(features)):

            val = corr.values[i, j]

            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white"
                    if abs(val) > 0.55
                    else "black"
                )
            )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=10,
        fontweight="bold"
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    return mpl_pane(
        fig,
        height=480
    )


# ==================================================================
# K-MEANS
# ==================================================================
@pn.depends(
    ml_x_feature,
    ml_y_feature,
    ml_n_clusters,
    ml_top_n,
    year_range
)
def kmeans_panel(
    x_feature,
    y_feature,
    k,
    top_n,
    yrange
):

    if x_feature == y_feature:

        return mpl_pane(
            empty_fig(
                "Choose two different features\nfor clustering",
                (7, 4.2)
            ),
            height=460
        )

    snap, used_year = cross_country_snapshot(
        [x_feature, y_feature],
        top_n=top_n,
        sort_by=x_feature,
        min_rows=max(k * 2, 8)
    )

    if snap.empty or len(snap) < k:

        return mpl_pane(
            empty_fig(
                "Not enough overlapping data\nfor these two features",
                (7, 4.2)
            ),
            height=460
        )

    X_raw = snap[
        [x_feature, y_feature]
    ].values

    X_scaled = StandardScaler().fit_transform(
        X_raw
    )

    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    ).fit(X_scaled)

    snap = snap.copy()

    snap["cluster"] = km.labels_

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11, 4.3)
    )

    for c in sorted(
        snap["cluster"].unique()
    ):

        sub = snap[
            snap["cluster"] == c
        ]

        ax1.scatter(
            sub[x_feature],
            sub[y_feature],
            s=32,
            color=CLUSTER_PALETTE[
                c % len(CLUSTER_PALETTE)
            ],
            label=f"Cluster {c}"
        )

    ax1.set_title(
        f"K-Means (k={k}) — {used_year}",
        fontsize=9.5,
        fontweight="bold"
    )

    ax1.set_xlabel(
        pretty_label(x_feature),
        fontsize=8
    )

    ax1.set_ylabel(
        pretty_label(y_feature),
        fontsize=8
    )

    ax1.legend(
        fontsize=7,
        loc="best"
    )

    ax1.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    inertias = []

    k_range = (
        range(
            2,
            min(10, len(snap)) + 1
        )
        if len(snap) > 2
        else range(2, 3)
    )

    for kk in k_range:

        inertias.append(
            KMeans(
                n_clusters=kk,
                n_init=10,
                random_state=42
            )
            .fit(X_scaled)
            .inertia_
        )

    ax2.plot(
        list(k_range),
        inertias,
        marker="o",
        color=COLOR_PRIMARY
    )

    ax2.axvline(
        k,
        color=COLOR_ALERT,
        ls="--",
        lw=1.5,
        label=f"Selected k={k}"
    )

    ax2.set_title(
        "Elbow Method (Inertia vs k)",
        fontsize=9.5,
        fontweight="bold"
    )

    ax2.set_xlabel(
        "k",
        fontsize=8
    )

    ax2.set_ylabel(
        "Inertia",
        fontsize=8
    )

    ax2.legend(
        fontsize=7.5
    )

    ax2.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    return mpl_pane(
        fig,
        height=440
    )


# ==================================================================
# HIERARCHICAL CLUSTERING
# ==================================================================
@pn.depends(
    ml_x_feature,
    ml_y_feature,
    ml_top_n,
    year_range
)
def hierarchical_panel(
    x_feature,
    y_feature,
    top_n,
    yrange
):

    if x_feature == y_feature:

        return mpl_pane(
            empty_fig(
                "Choose two different features\nfor clustering",
                (8, 5)
            ),
            height=480
        )

    capped_n = min(
        top_n,
        35
    )

    snap, used_year = cross_country_snapshot(
        [x_feature, y_feature],
        top_n=capped_n,
        sort_by=x_feature,
        min_rows=5
    )

    if snap.empty or len(snap) < 3:

        return mpl_pane(
            empty_fig(
                "Not enough overlapping data\nfor these two features",
                (8, 5)
            ),
            height=480
        )

    X_scaled = StandardScaler().fit_transform(
        snap[
            [x_feature, y_feature]
        ].values
    )

    Z = linkage(
        X_scaled,
        method="ward"
    )

    fig, ax = plt.subplots(
        figsize=(
            max(7, 0.28 * len(snap)),
            5
        )
    )

    dendrogram(
        Z,
        labels=snap["country"].tolist(),
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
        color_threshold=0.7 * max(Z[:, 2])
    )

    ax.set_title(
        f"Hierarchical Clustering "
        f"(Ward linkage) — "
        f"{pretty_label(x_feature)} & "
        f"{pretty_label(y_feature)}, {used_year}",
        fontsize=9.5,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Distance",
        fontsize=8
    )

    ax.tick_params(
        axis="y",
        labelsize=8
    )

    return mpl_pane(
        fig,
        height=500
    )


# ==================================================================
# CLASSIFICATION
# ==================================================================
@pn.depends(
    ml_x_feature,
    ml_y_feature,
    ml_target_indicator,
    ml_classifier_algo,
    year_range
)
def classification_panel(
    x_feature,
    y_feature,
    target_indicator,
    algo,
    yrange
):

    if x_feature == y_feature:

        return mpl_pane(
            empty_fig(
                "Choose two different features\nfor classification",
                (11, 4.5)
            ),
            height=460
        )

    snap, used_year = cross_country_snapshot(
        [
            x_feature,
            y_feature,
            target_indicator
        ],
        top_n=None,
        min_rows=15
    )

    if snap.empty or len(snap) < 10:

        return mpl_pane(
            empty_fig(
                "Not enough overlapping data\nfor these features",
                (11, 4.5)
            ),
            height=460
        )

    median_val = snap[
        target_indicator
    ].median()

    snap = snap.copy()

    snap["label"] = np.where(
        snap[target_indicator] > median_val,
        1,
        0
    )

    X = snap[
        [x_feature, y_feature]
    ].values

    y = snap["label"].values

    scaler = StandardScaler().fit(X)

    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=42,
        stratify=(
            y
            if len(set(y)) > 1
            else None
        )
    )

    clf = (
        LogisticRegression()
        if algo == "Logistic Regression"
        else Perceptron(
            max_iter=1000,
            random_state=42
        )
    )

    clf.fit(
        X_train,
        y_train
    )

    y_pred = clf.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11, 4.3)
    )

    # Decision boundary
    x_min = (
        X_scaled[:, 0].min()
        - 0.5
    )

    x_max = (
        X_scaled[:, 0].max()
        + 0.5
    )

    y_min = (
        X_scaled[:, 1].min()
        - 0.5
    )

    y_max = (
        X_scaled[:, 1].max()
        + 0.5
    )

    xx, yy = np.meshgrid(
        np.linspace(
            x_min,
            x_max,
            200
        ),
        np.linspace(
            y_min,
            y_max,
            200
        )
    )

    zz = clf.predict(
        np.c_[
            xx.ravel(),
            yy.ravel()
        ]
    ).reshape(
        xx.shape
    )

    ax1.contourf(
        xx,
        yy,
        zz,
        alpha=0.25,
        cmap=ListedColormap([
            "#90CAF9",
            "#A5D6A7"
        ])
    )

    ax1.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=y,
        cmap=ListedColormap([
            COLOR_NEUTRAL,
            COLOR_PRIMARY
        ]),
        edgecolors="k",
        lw=0.5,
        s=30
    )

    ax1.set_title(
        f"{algo} Boundary ({used_year})",
        fontsize=9.5,
        fontweight="bold"
    )

    ax1.set_xlabel(
        f"{pretty_label(x_feature)} (std)",
        fontsize=8
    )

    ax1.set_ylabel(
        f"{pretty_label(y_feature)} (std)",
        fontsize=8
    )

    ax1.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    # Confusion Matrix
    im = ax2.imshow(
        cm,
        cmap="Greens"
    )

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])

    ax2.set_xticklabels(
        ["Below Med", "Above Med"],
        fontsize=8
    )

    ax2.set_yticklabels(
        ["Below Med", "Above Med"],
        fontsize=8
    )

    ax2.set_xlabel(
        "Predicted",
        fontsize=8
    )

    ax2.set_ylabel(
        "Actual",
        fontsize=8
    )

    ax2.set_title(
        f"Confusion Matrix "
        f"(Accuracy: {acc:.1%})",
        fontsize=9.5,
        fontweight="bold"
    )

    for i in range(2):

        for j in range(2):

            ax2.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                )
            )

    return mpl_pane(
        fig,
        height=460
    )


# ==================================================================
# PDF GENERATOR
# ==================================================================
def generate_pdf_report(
    indicator,
    country,
    yrange
):

    """Generate executive PDF report into memory."""

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor(
            COLOR_PRIMARY
        ),
        alignment=0
    )

    heading_style = ParagraphStyle(
        "DocHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(
            COLOR_PRIMARY
        ),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor(
            COLOR_NEUTRAL
        )
    )

    story = []

    # Title
    story.append(
        Paragraph(
            f"Executive Report: "
            f"{pretty_label(indicator)}",
            title_style
        )
    )

    story.append(
        Spacer(1, 4)
    )

    story.append(
        Paragraph(
            f"<b>Target Country:</b> {country} | "
            f"<b>Period:</b> {yrange[0]} – {yrange[1]} | "
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%Y-%m-%d')}",
            body_style
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor(
                COLOR_PRIMARY
            ),
            spaceBefore=8,
            spaceAfter=12
        )
    )

    # Key statistics
    series = get_country_series(
        indicator
    )

    if not series.empty:

        latest_val = human_format(
            series[indicator].iloc[-1],
            indicator
        )

        avg_val = human_format(
            series[indicator].mean(),
            indicator
        )

        min_val = human_format(
            series[indicator].min(),
            indicator
        )

        max_val = human_format(
            series[indicator].max(),
            indicator
        )

    else:

        latest_val = avg_val = min_val = max_val = "N/A"

    table_data = [
        [
            "Metric",
            "Value",
            "Metric",
            "Value"
        ],
        [
            f"Latest ({series['year'].iloc[-1] if not series.empty else '—'})",
            latest_val,
            "Period Average",
            avg_val
        ],
        [
            "Period Minimum",
            min_val,
            "Period Maximum",
            max_val
        ]
    ]

    t = Table(
        table_data,
        colWidths=[
            130,
            130,
            130,
            130
        ]
    )

    t.setStyle(
        TableStyle([
            (
                'BACKGROUND',
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    COLOR_PRIMARY
                )
            ),
            (
                'TEXTCOLOR',
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                'FONTNAME',
                (0, 0),
                (-1, 0),
                'Helvetica-Bold'
            ),
            (
                'FONTSIZE',
                (0, 0),
                (-1, -1),
                8.5
            ),
            (
                'ALIGN',
                (0, 0),
                (-1, -1),
                'CENTER'
            ),
            (
                'BOTTOMPADDING',
                (0, 0),
                (-1, -1),
                5
            ),
            (
                'TOPPADDING',
                (0, 0),
                (-1, -1),
                5
            ),
            (
                'GRID',
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#CCCCCC")
            ),
        ])
    )

    story.append(t)

    story.append(
        Spacer(1, 14)
    )

    # Executive commentary
    story.append(
        Paragraph(
            "Executive Summary & Strategic Insights",
            heading_style
        )
    )

    commentary = (
        f"This automated analytical report reviews the overall trajectory "
        f"of <b>{pretty_label(indicator)}</b> for <b>{country}</b> "
        f"across the period from {yrange[0]} to {yrange[1]}. "
        f"Over this window, the indicator reached a maximum value of "
        f"{max_val} and a minimum of {min_val}, settling at an average "
        f"value of {avg_val}. Monitoring these macro-level dynamics is "
        f"critical for structuring resilient public policy, regional "
        f"economic integration, and global development initiatives."
    )

    story.append(
        Paragraph(
            commentary,
            body_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


def download_pdf_action():

    return generate_pdf_report(
        select_indicator.value,
        select_country.value,
        year_range.value
    )


pdf_download_button = pn.widgets.FileDownload(
    filename="WorldBank_Report.pdf",
    callback=download_pdf_action,
    button_type="success",
    label="📄 Download PDF Summary",
    sizing_mode="stretch_width"
)


# ==================================================================
# DASHBOARD LAYOUT
# ==================================================================

# ------------------------------------------------------------------
# 1. SUMMARY VIEW
# ------------------------------------------------------------------
summary_tab = pn.Column(

    # Dynamic single-line KPI bar
    kpi_cards,

    pn.Tabs(

        (
            "Scatter Plot",
            chart_page(
                scatter_plot,
                SUMMARY_DESCRIPTIONS["scatter"]
            )
        ),

        (
            "Top 10 Leaderboard",
            chart_page(
                bar_plot,
                SUMMARY_DESCRIPTIONS["bar"]
            )
        ),

        (
            "Data Table View",
            table_view
        ),

        sizing_mode="stretch_width",
        tabs_location="above"
    ),

    sizing_mode="stretch_width"
)


# ------------------------------------------------------------------
# 2. INSIGHTS VIEW
# ------------------------------------------------------------------
insights_tab = pn.Tabs(

    (
        "Historical Trend",
        chart_page(
            line_chart_panel,
            INSIGHTS_DESCRIPTIONS["line"]
        )
    ),

    (
        "Global Share",
        chart_page(
            pie_chart_panel,
            INSIGHTS_DESCRIPTIONS["pie"]
        )
    ),

    (
        "Decade Averages",
        chart_page(
            decade_bar_panel,
            INSIGHTS_DESCRIPTIONS["decade"]
        )
    ),

    (
        "Distribution",
        chart_page(
            histogram_panel,
            INSIGHTS_DESCRIPTIONS["histogram"]
        )
    ),

    sizing_mode="stretch_width",
    tabs_location="above"
)


# ------------------------------------------------------------------
# 3. MACHINE LEARNING VIEW
# ------------------------------------------------------------------
ml_controls_box = pn.Card(

    pn.Column(

        pn.pane.Markdown(
            "### Machine Learning Configuration"
        ),

        pn.Row(
            ml_x_feature,
            ml_y_feature,
            ml_target_indicator
        ),

        pn.Row(
            ml_classifier_algo,
            ml_n_clusters,
            ml_top_n
        ),

        ml_heatmap_features,

    ),

    title="⚙️ ML Model Controls & Hyperparameters",

    collapsed=False,

    sizing_mode="stretch_width",

    margin=(0, 0, 15, 0)
)


ml_tab = pn.Column(

    ml_controls_box,

    pn.Tabs(

        (
            "Linear Regression",
            chart_page(
                linear_regression_panel,
                ML_DESCRIPTIONS["regression"]
            )
        ),

        (
            "Correlation Heatmap",
            chart_page(
                heatmap_panel,
                ML_DESCRIPTIONS["heatmap"]
            )
        ),

        (
            "K-Means Clustering",
            chart_page(
                kmeans_panel,
                ML_DESCRIPTIONS["kmeans"]
            )
        ),

        (
            "Hierarchical Clustering",
            chart_page(
                hierarchical_panel,
                ML_DESCRIPTIONS["hierarchical"]
            )
        ),

        (
            "Classification",
            chart_page(
                classification_panel,
                ML_DESCRIPTIONS["classification"]
            )
        ),

        sizing_mode="stretch_width",
        tabs_location="above"
    ),

    sizing_mode="stretch_width"
)


# ==================================================================
# SIDEBAR
# ==================================================================
sidebar = [

    pn.pane.Markdown(
        "## Global Filters"
    ),

    select_indicator,

    select_country,

    year_range,

    reset_button,

    pn.layout.Divider(),

    pn.pane.Markdown(
        "## Reporting"
    ),

    pdf_download_button
]


# ==================================================================
# MAIN APPLICATION TEMPLATE
# ==================================================================
template = pn.template.FastListTemplate(

    title=(
        "World Bank Indicators — "
        "Analytics & ML Lab"
    ),
    sidebar=sidebar,
    sidebar_width=300,
    main=[
        pn.Tabs(("Executive Summary", summary_tab), ("In-Depth Insights",insights_tab), 
            ("Machine Learning Lab", ml_tab), sizing_mode="stretch_width", tabs_location="above")
    ],

    accent_base_color=COLOR_PRIMARY,
    header_background=COLOR_PRIMARY,
)


# ==================================================================
# SERVE APPLICATION
# ==================================================================
template.servable()