# World Bank Development Indicators Dashboard, PDF Generator & ML Lab

An interactive **World Bank Development Indicators analytics dashboard** built with Python and [Panel](https://panel.holoviz.org/).

The application combines interactive data visualization, statistical exploration, automated PDF reporting, and a Machine Learning laboratory in a single web-based analytical platform.

---

## 📊 Overview

The **World Bank Development Indicators Dashboard** allows users to explore development indicators across countries and years through an interactive interface.

The application provides three main analytical areas:

1. **Executive Summary**

   * KPI cards
   * Country comparison
   * Top-10 country leaderboard
   * Interactive data table
   * Cross-country scatter analysis

2. **In-Depth Insights**

   * Historical trend analysis
   * Global share analysis
   * Decade averages
   * Indicator distributions

3. **Machine Learning Lab**

   * Linear Regression
   * Correlation Heatmap
   * K-Means Clustering
   * Hierarchical/Agglomerative Clustering
   * Logistic Regression
   * Perceptron classification

The dashboard also includes an **automated PDF executive report generator** that produces a downloadable summary for a selected country, indicator, and time period.

---

## 🚀 Key Features

### Interactive Dashboard

The application is powered by **Panel**, providing interactive widgets and responsive dashboards without requiring a separate frontend framework.

Users can dynamically select:

* Development indicator
* Country
* Year range
* Machine-learning features
* Classification target
* Classification algorithm
* Number of clusters
* Number of countries
* Heatmap indicators

All relevant visualizations automatically update when the filters change.

---

# 📈 Executive Summary

The **Executive Summary** tab provides a high-level overview of the selected indicator and country.

### KPI Cards

The dashboard calculates:

* Latest available value
* Period average
* Period minimum
* Period maximum

Values are automatically formatted for readability.

For example:

```text
Latest:        $2.45T
Average:       $1.87T
Minimum:       $950.2B
Maximum:       $2.45T
```

---

### Scatter Plot

The scatter plot provides a cross-country comparison using the selected indicator.

By default:

```text
Selected Indicator → GDP
```

When GDP itself is selected:

```text
GDP → Population
```

The latest available year containing data for the selected indicator is automatically used.

Countries can be identified through hover information.

This visualization is useful for identifying:

* Relationships
* Correlations
* Outliers
* Country-level differences
* Economic and demographic patterns

---

### Top 10 Leaderboard

The dashboard displays the **10 countries with the highest values** for the selected indicator.

This provides a quick ranking of countries for the latest available year.

---

### Data Table

The data table provides a detailed view of the selected country and indicator.

The table includes:

* Country
* Year
* Selected indicator

Up to 50 records are displayed with local pagination.

---

# 📊 In-Depth Insights

The **In-Depth Insights** section provides additional statistical visualizations.

## Historical Trend

A line chart displays the evolution of the selected indicator for the selected country over the chosen period.

This can be used to identify:

* Long-term growth
* Declining trends
* Structural changes
* Turning points
* Periods of volatility

---

## Global Share

A pie chart displays the contribution/share of the **top six countries** for the selected indicator in the latest available year.

This is useful for understanding how concentrated an indicator is among countries.

---

## Decade Averages

The application calculates the average value of the selected indicator for each decade.

For example:

```text
1990s
2000s
2010s
2020s
```

This smooths short-term fluctuations and makes long-term comparisons easier.

---

## Distribution

A histogram shows the distribution of the selected indicator across countries in the latest available year.

It can help identify:

* Central tendency
* Dispersion
* Skewness
* Outliers
* Country concentration

---

# 🤖 Machine Learning Lab

The Machine Learning Lab provides several supervised and unsupervised learning techniques.

The machine-learning controls allow users to configure:

* Feature X
* Feature Y
* Classification target
* Classification algorithm
* Number of clusters
* Number of countries
* Heatmap indicators

All models are generated dynamically from the selected World Bank data.

---

# 1. Linear Regression

The Linear Regression module models the relationship between:

```text
Year → Selected Indicator
```

for the selected country.

The model uses:

```python
sklearn.linear_model.LinearRegression
```

The application calculates:

```text
R²
```

to measure how well the linear model explains the historical data.

### Forecast

The model projects the indicator **five years into the future**.

The visualization contains:

* Historical observations
* Fitted regression line
* Five-year projection

### Important Note

The projection is a simple linear extrapolation. It should therefore be interpreted as an analytical scenario rather than an official economic forecast.

---

# 2. Correlation Heatmap

The correlation heatmap calculates Pearson correlations between selected indicators.

The correlation coefficient ranges from:

```text
-1 → Strong negative relationship
 0 → Little/no linear relationship
+1 → Strong positive relationship
```

Users can select up to **12 indicators**.

The heatmap displays both the color intensity and numerical correlation value.

---

# 3. K-Means Clustering

K-Means groups countries into clusters according to two selected features.

Before clustering, the features are standardized using:

```python
StandardScaler
```

The algorithm uses:

```python
sklearn.cluster.KMeans
```

with:

```text
random_state = 42
n_init = 10
```

### Cluster Visualization

The application displays countries according to their assigned cluster.

### Elbow Method

The dashboard also calculates cluster inertia for multiple values of `k`.

The elbow method can help determine an appropriate number of clusters.

The available cluster range is:

```text
2–8
```

---

# 4. Hierarchical Clustering

Hierarchical clustering creates a tree-like structure showing similarities between countries.

The application uses:

```python
scipy.cluster.hierarchy.linkage
```

with:

```text
Ward linkage
```

The resulting dendrogram shows how countries progressively merge into larger groups.

Because large dendrograms become difficult to read, the application limits the displayed dataset to a maximum of approximately **35 countries**.

---

# 5. Classification

The classification module converts a continuous indicator into two groups using its median value.

The classes are:

```text
0 = Below Median
1 = Above Median
```

The classifier uses two selected features to predict the class.

Two algorithms are available:

### Logistic Regression

Implemented using:

```python
sklearn.linear_model.LogisticRegression
```

### Perceptron

Implemented using:

```python
sklearn.linear_model.Perceptron
```

---

## Classification Workflow

The workflow is:

```text
World Bank Data
      ↓
Select Features
      ↓
Select Classification Target
      ↓
Calculate Median
      ↓
Create Binary Classes
      ↓
Standardize Features
      ↓
Train/Test Split
      ↓
Train Classifier
      ↓
Generate Predictions
      ↓
Evaluate Accuracy
      ↓
Display Decision Boundary
      ↓
Display Confusion Matrix
```

The test set represents approximately **30% of the available observations**.

The dashboard displays:

* Decision boundary
* Classification groups
* Confusion matrix
* Overall accuracy

---

# 📄 Automated PDF Report

The application includes an automated PDF reporting system using **ReportLab**.

Users can select:

* Indicator
* Country
* Year range

and click:

```text
📄 Download PDF Summary
```

The generated report contains:

### Report Header

* Selected indicator
* Target country
* Reporting period
* Generation date

### Key Statistics

* Latest value
* Period average
* Period minimum
* Period maximum

### Executive Summary

The application automatically generates a short analytical commentary describing the selected indicator and period.

---

# 🗂️ Project Structure

A recommended project structure is:

```text
worldbank-dashboard/
│
├── worldbank_dashboard.py
├── README.md
├── requirements.txt
│
├── data/
│   └── world_bank_development_indicators.csv
│
├── logo.png
│
└── outputs/
    └── reports/
```

The minimum required structure is:

```text
.
├── worldbank_dashboard.py
├── data/
│   └── world_bank_development_indicators.csv
└── logo.png
```

---

# 📁 Dataset

The application expects the World Bank dataset at:

```text
./data/world_bank_development_indicators.csv
```

The CSV should contain at least:

```text
country
date
GDP_current_US
population
life_expectancy_at_birth
CO2_emisions
inflation_annual%
individuals_using_internet%
access_to_electricity%
agricultural_land%
forest_land%
gini_index
```

Additional numeric indicators can also be included.

The application automatically discovers numeric columns and adds them to the available indicator and machine-learning feature selections.

---

# 🌍 Country Filtering

The application automatically removes regional and aggregate entities such as:

```text
World
Africa Eastern and Southern
Africa Western and Central
European Union
High income
Low income
Middle income
OECD members
Sub-Saharan Africa
South Asia
Latin America & Caribbean
```

This ensures that country-level comparisons focus primarily on individual countries.

The list is maintained in:

```python
NON_COUNTRY_ENTITIES
```

Additional aggregate entities can be added to this list if required.

---

# 🛠️ Technologies Used

## Programming Language

* Python 3

## Data Processing

* Pandas
* NumPy

## Dashboard

* Panel
* HoloViews
* hvPlot

## Visualization

* Matplotlib
* hvPlot

## Machine Learning

* Scikit-learn
* SciPy

## PDF Generation

* ReportLab

---

# 📦 Installation

## 1. Clone or copy the project

```bash
git clone <your-repository-url>
cd worldbank-dashboard
```

Or simply copy the project files to your preferred directory.

---

## 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies

Create a `requirements.txt` file containing:

```text
numpy
pandas
panel
hvplot
matplotlib
scikit-learn
scipy
reportlab
```

Then run:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Dashboard

Start the application with:

```bash
panel serve worldbank_dashboard.py --show
```

Panel will start a local web server and automatically open the dashboard in your browser.

The application can also be started without automatically opening the browser:

```bash
panel serve worldbank_dashboard.py
```

Then open the address displayed by Panel, typically:

```text
http://localhost:5006/worldbank_dashboard
```

---

# 🔧 Configuration

The main configuration variables are located near the beginning of the Python file.

## Dataset Path

```python
FILE_PATH = Path("./data/world_bank_development_indicators.csv")
```

Change this if your dataset is stored somewhere else.

For example:

```python
FILE_PATH = Path("./datasets/world_bank.csv")
```

---

## Logo

The application defines:

```python
LOGO_PATH = Path("./logo.png")
```

The logo path is currently configured but is not actively inserted into the generated PDF or dashboard layout in the supplied implementation.

---

## Featured Indicators

The main indicators are configured through:

```python
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
```

You can add additional indicators as long as they exist in the dataset.

---

# ⚡ Data Loading and Fallback Dataset

The application uses a cached data-loading function:

```python
@pn.cache
def load_data(path: Path) -> pd.DataFrame:
```

If the World Bank CSV exists, it is loaded normally.

If the CSV is missing, the application automatically generates a mock dataset containing:

* 10 countries
* 1990–2023
* GDP
* Population
* Life expectancy
* CO₂ emissions
* Inflation
* Internet usage
* Electricity access
* Agricultural land
* Forest land
* Gini index

This fallback mechanism allows the dashboard to be tested even when the real dataset is unavailable.

**For production analysis, always use the real World Bank dataset.**

---

# 📊 Data Processing Logic

The application converts the `date` column into a Pandas datetime:

```python
data["date"] = pd.to_datetime(data["date"])
```

A separate year column is then generated:

```python
data["year"] = data["date"].dt.year
```

This year field is used throughout the dashboard for:

* Year filtering
* Trend analysis
* Regression
* Latest-year detection
* Decade aggregation
* Cross-country comparisons

---

# 📐 Statistical Methodology

## Latest Available Year

The dashboard does not necessarily assume that every indicator has observations for the maximum dataset year.

Instead, it searches for the latest year containing valid observations for the selected indicator.

This is especially useful because World Bank indicators may have different reporting frequencies and end dates.

---

## Cross-Country Snapshot

For machine-learning models, the application searches backward from the most recent year until it finds a year with sufficient complete observations for the selected features.

This helps reduce problems caused by missing values.

---

# 🎨 Dashboard Design

The dashboard uses a green development/environment-oriented visual palette.

Primary colors include:

```text
Forest Green       #1B5E20
Medium Green       #2E7D32
Charcoal Slate     #37474F
Crimson Alert      #C62828
White              #FFFFFF
```

The application uses:

```python
pn.template.FastListTemplate
```

with a responsive layout.

---

# 🔄 Dashboard Workflow

The overall analytical workflow is:

```text
                     ┌─────────────────────┐
                     │ World Bank CSV Data │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Data Loading &      │
                     │ Validation          │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Global Filters      │
                     │ Country / Indicator │
                     │ Year Range          │
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
       Executive Summary   In-Depth Insights   ML Lab
              │                 │                 │
              ▼                 ▼                 ▼
          KPIs              Line Chart       Regression
          Scatter           Pie Chart        Heatmap
          Top 10            Decades          K-Means
          Table             Histogram        Hierarchical
                                               Classification
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Automated PDF       │
                     │ Executive Report    │
                     └─────────────────────┘
```

---

# 🧪 Reproducibility

Machine-learning components use fixed random seeds where appropriate.

For example:

```python
random_state=42
```

This improves reproducibility of:

* K-Means clustering
* Train/test splitting
* Perceptron training

However, results can still vary depending on:

* Dataset version
* Missing values
* Available countries
* Selected indicators
* Selected year range
* Installed package versions

---

# ⚠️ Important Analytical Considerations

This dashboard is intended for **exploratory data analysis, portfolio demonstration, analytical learning, and decision-support prototyping**.

The machine-learning outputs should not automatically be interpreted as authoritative forecasts or policy recommendations.

### Linear Regression

The five-year projection is based on a simple linear relationship between year and the selected indicator.

It does not account for:

* Economic shocks
* Policy changes
* Inflation
* Structural breaks
* Geopolitical events
* Non-linear trends

### Correlation

Correlation does not imply causation.

A high correlation between two indicators does not establish that one indicator causes the other.

### K-Means

K-Means results depend on:

* Feature selection
* Scaling
* Number of clusters
* Available observations
* Distance relationships

### Classification

The classification target is artificially created using the median of the selected indicator.

Therefore:

```text
Above Median
```

and

```text
Below Median
```

are analytical classes rather than official World Bank classifications.

---

# 🐛 Troubleshooting

## Dashboard does not start

Make sure Panel is installed:

```bash
pip install panel
```

Then run:

```bash
panel serve worldbank_dashboard.py --show
```

---

## `ModuleNotFoundError`

Install all dependencies:

```bash
pip install numpy pandas panel hvplot matplotlib scikit-learn scipy reportlab
```

---

## Dataset not found

Verify that the dataset exists at:

```text
./data/world_bank_development_indicators.csv
```

You can also change:

```python
FILE_PATH
```

to the correct location.

If the file is missing, the application will use its built-in mock dataset.

---

## Empty visualizations

Some World Bank indicators contain missing observations.

The application automatically searches for the latest year containing valid data, but some combinations of indicators may still have insufficient observations.

Try:

* Selecting another indicator
* Expanding the year range
* Selecting another country
* Selecting different ML features

---

## PDF generation problems

Make sure ReportLab is installed:

```bash
pip install reportlab
```

If the PDF contains no charts, this is expected in the current implementation: the supplied `generate_pdf_report()` function generates an **executive statistics and commentary report**, but does not currently embed the dashboard charts.

Charts can be added later by saving Matplotlib figures to `BytesIO` objects and inserting them into the ReportLab document.

---

# 🔐 Production Deployment

For local use:

```bash
panel serve worldbank_dashboard.py --show
```

For a server deployment:

```bash
panel serve worldbank_dashboard.py \
    --address 0.0.0.0 \
    --port 5006
```

For production environments, it is recommended to place the Panel application behind a reverse proxy such as:

* Nginx
* Apache
* Traefik

HTTPS should also be configured when the application is exposed outside a trusted local network.

---

# 📌 Example Use Cases

This dashboard can be used for:

* Development economics analysis
* Country benchmarking
* Socio-economic research
* Data analytics portfolios
* World Bank indicator exploration
* Government decision support
* International development research
* Environmental analysis
* Digital development analysis
* Data science demonstrations
* Machine-learning education
* Exploratory predictive analytics

---

# 🎓 Portfolio Value

This project demonstrates practical skills across several areas of modern data analytics.

### Data Engineering

* CSV ingestion
* Data validation
* Data transformation
* Missing-data handling
* Dynamic feature discovery

### Data Analytics

* Descriptive statistics
* Cross-country benchmarking
* Trend analysis
* Distribution analysis
* Correlation analysis

### Data Visualization

* Scatter plots
* Line charts
* Bar charts
* Horizontal bar charts
* Pie charts
* Histograms
* Heatmaps
* Dendrograms
* Confusion matrices

### Machine Learning

* Linear Regression
* Logistic Regression
* Perceptron
* K-Means
* Hierarchical Clustering
* Feature standardization
* Train/test splitting
* Model evaluation

### Reporting

* Automated PDF generation
* KPI reporting
* Executive commentary
* Downloadable analytical reports

### Application Development

* Python
* Panel
* Reactive dashboards
* Interactive widgets
* Dynamic visualization
* Cached data loading

---

# 📜 License

Add the appropriate license for your project.

For example:

```text
MIT License
```

If the project uses World Bank data, also review and comply with the applicable World Bank data terms and attribution requirements.

---

# 👨‍💻 Author

**Edmond Dondon**

Data Analytics & Software Development Specialist

Bangui, Central African Republic

Areas of expertise:

* Data Analytics
* Data Engineering
* Software Development
* Business Intelligence
* Machine Learning
* GIS & Spatial Analytics
* Digital Transformation
* Development Information Systems

GitHub:

```text
https://github.com/dondonedmond82
```

LinkedIn:

```text
https://linkedin.com/in/edmond-dondon-bb364870
```

Email:

```text
dondonedmond@gmail.com
```

---

# 🚀 Future Enhancements

Potential improvements include:

* [ ] Embed dashboard charts directly into PDF reports
* [ ] Add interactive maps
* [ ] Add World Bank API integration
* [ ] Add automated data refresh
* [ ] Add forecasting models such as ARIMA and Prophet
* [ ] Add Random Forest and Gradient Boosting
* [ ] Add XGBoost
* [ ] Add model-performance comparison
* [ ] Add ROC/AUC analysis
* [ ] Add feature importance analysis
* [ ] Add downloadable CSV datasets
* [ ] Add Excel export
* [ ] Add Power BI-compatible exports
* [ ] Add user authentication
* [ ] Add database connectivity
* [ ] Add Docker deployment
* [ ] Add automated testing
* [ ] Add CI/CD with GitHub Actions
* [ ] Add cloud deployment
* [ ] Add multilingual English/French interface

---

# 📄 Quick Start

The fastest way to run the project is:

```bash
# Create environment
python -m venv .venv

# Activate environment - Linux/macOS
source .venv/bin/activate

# Activate environment - Windows
.venv\Scripts\activate

# Install dependencies
pip install numpy pandas panel hvplot matplotlib scikit-learn scipy reportlab

# Run dashboard
panel serve worldbank_dashboard.py --show
```

The dashboard will then be available through the local Panel web interface.

---

## ⭐ Project Summary

**World Bank Development Indicators Dashboard & ML Lab** is a complete Python-based analytical application combining:

```text
World Bank Data
      +
Interactive Dashboard
      +
Descriptive Analytics
      +
Data Visualization
      +
Machine Learning
      +
Predictive Analysis
      +
Automated PDF Reporting
```

It provides an integrated environment for transforming development indicators into interactive visual insights, statistical analysis, machine-learning experiments, and executive-level reporting.
