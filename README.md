# GLDAS 2.2 Groundwater Evaluation Framework

## Overview

This framework evaluates the reliability of GLDAS 2.2 (Global Land Data Assimilation System) for groundwater storage estimation by systematically comparing it with in-situ well measurements from the USGS across the Mississippi River Basin. The tool automates data acquisition, processing, and statistical analysis to determine whether GLDAS can serve as a viable proxy for groundwater estimation.

## Features

- **Automated Data Acquisition**: Retrieves USGS well data and corresponding GLDAS measurements
- **Spatial & Temporal Analysis**: Evaluates alignment between satellite and ground-based measurements
- **Comprehensive Metrics**: Calculates correlation, RMSE, Nash-Sutcliffe Efficiency, and other statistical measures 
- **Visualization Tools**: Generates time series comparisons, correlation maps, and statistical distributions
- **Modular Architecture**: Well-structured codebase for extension to other regions or datasets

## Project Structure

```
gldas_evaluation/
├── config/
│   └── config.yaml       # Configuration parameters
├── data/
│   ├── raw/              # Store downloaded data
│   │   ├── wells/        # Raw well measurements
│   │   └── gldas/        # Raw GLDAS data
│   └── processed/        # Processed data
│       ├── wells/        # Processed well data
│       └── gldas/        # Processed GLDAS data
├── scripts/
│   ├── download_wells.py # Download well data utility
│   └── extract_gldas.py  # Extract GLDAS data utility
├── src/
│   ├── data/             # Data acquisition modules
│   ├── processing/       # Data processing modules
│   ├── analysis/         # Analysis and comparison tools
│   └── utils/            # Helper functions
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_data_processing.ipynb
│   └── 03_comparison_analysis.ipynb
├── results/
│   ├── run_TIMESTAMP/    # Results from each analysis run
│   └── figures/          # Generated visualizations
├── environment.yml       # Conda environment definition
├── main.py               # Main execution script
└── README.md             # This documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda package manager
- Google Earth Engine account
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/gldas_evaluation.git
   cd gldas_evaluation
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate gldas_evaluation
   ```

3. **Google Earth Engine authentication**
   ```bash
   earthengine authenticate
   ```

4. **Update the configuration**
   
   Edit `config/config.yaml` to specify your GEE project ID and other parameters

## Configuration

The `config.yaml` file controls all aspects of the analysis:

```yaml
# Study region
region:
  name: "Mississippi River Basin"
  bounds:
    lat_min: 29.0
    lat_max: 49.0
    lon_min: -98.0
    lon_max: -80.0

# Time period
time:
  start_date: "2003-01-01"
  end_date: "2022-12-31"

# USGS NWIS parameters
usgs:
  service: "groundwater"
  site_status: "active"
  min_data_points: 200
  parameter_code: "72019"  # Depth to water level code

# GLDAS parameters
gldas:
  collection: "NASA/GLDAS/V022/CLSM/G025/DA1D"
  variable: "GWS_tavg"
  scale_factor: 277.1  # 77.0153 * 3599.01 / 1000 (converted to meters)
  project_id: "your-gee-project-id"  # Your GEE project ID

# Analysis parameters
analysis:
  temporal_resolution: "monthly"
  comparison_metrics: ["correlation", "rmse", "nse"]
```

## Usage

### Running the Complete Workflow

```bash
python main.py --config config/config.yaml --mode all
```

### Running Individual Stages

1. **Data acquisition only**
   ```bash
   python main.py --mode download
   ```

2. **Data processing only**
   ```bash
   python main.py --mode process
   ```

3. **Analysis only** (requires previously downloaded and processed data)
   ```bash
   python main.py --mode analyze
   ```

## Example Workflow

### 1. Configure the Study Area

Edit the `config.yaml` file to define your region of interest and time period.

### 2. Download Data

Run the data acquisition stage to collect well measurements and corresponding GLDAS data:

```bash
python main.py --mode download
```

This will:
- Query the USGS NWIS database for all available groundwater wells in the region
- Filter wells based on data availability criteria
- Extract GLDAS data at each well location using Google Earth Engine

### 3. Process Data

Process the raw data to prepare it for comparison:

```bash
python main.py --mode process
```

This stage:
- Resamples data to a common temporal resolution (monthly)
- Calculates groundwater anomalies for both datasets
- Applies any necessary unit conversions and corrections

### 4. Analyze and Visualize

Run the analysis to compare the datasets:

```bash
python main.py --mode analyze
```

The analysis:
- Calculates correlation, RMSE, NSE and other metrics for each well
- Generates time series comparison plots
- Creates summary visualizations and statistics
- Identifies best and worst performing locations

## Interpreting Results

After running the analysis, results are stored in `results/run_TIMESTAMP/`. Key outputs include:

- **all_metrics.csv**: Contains statistical measures for each well site
- **Site comparison plots**: Time series visualization for each well
- **Summary plots**: Distribution of correlation values, spatial patterns, etc.
- **summary.json**: Overall statistics across all sites

Good performance indicators:
- Correlation values > 0.6
- NSE values > 0.5
- RMSE values < 0.2 m

## Extending the Framework

### Adding New Regions

1. Update the `region` section in `config.yaml` with new coordinates
2. Run the workflow with `--mode all`

### Using Different Well Data

The modular design allows for integration with other groundwater datasets:
1. Create a new data collector class in `src/data/`
2. Implement the required methods following the existing pattern
3. Update the main script to use your new collector

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- USGS for providing groundwater monitoring data
- NASA for the GLDAS dataset
- Google Earth Engine for facilitating access to satellite data