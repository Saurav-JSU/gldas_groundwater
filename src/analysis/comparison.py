import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

class DataComparison:
    """Class to compare well data with GLDAS data."""
    
    def __init__(self, config):
        """
        Initialize the data comparison.
        
        Parameters:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.metrics = {}
    
    def merge_datasets(self, well_df, gldas_df):
        """
        Merge well and GLDAS datasets for comparison.
        
        Parameters:
            well_df (pandas.DataFrame): Well data
            gldas_df (pandas.DataFrame): GLDAS data
            
        Returns:
            pandas.DataFrame: Merged dataset
        """
        # Ensure date columns are datetime
        well_df['date'] = pd.to_datetime(well_df['date'])
        gldas_df['date'] = pd.to_datetime(gldas_df['date'])
        
        # Merge on date
        merged = pd.merge(well_df, gldas_df, on='date', how='inner', suffixes=('_well', '_gldas'))
        
        return merged
    
    def calculate_metrics(self, merged_df, well_col='gw_anomaly_m', gldas_col='gldas_gws_anomaly'):
        """
        Calculate comparison metrics.
        
        Parameters:
            merged_df (pandas.DataFrame): Merged dataset
            well_col (str): Column name for well data
            gldas_col (str): Column name for GLDAS data
            
        Returns:
            dict: Calculated metrics
        """
        metrics = {}
        
        # Correlation
        metrics['correlation'] = merged_df[well_col].corr(merged_df[gldas_col])
        
        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((merged_df[well_col] - merged_df[gldas_col])**2))
        
        # Nash-Sutcliffe Efficiency
        mean_obs = merged_df[well_col].mean()
        numerator = np.sum((merged_df[well_col] - merged_df[gldas_col])**2)
        denominator = np.sum((merged_df[well_col] - mean_obs)**2)
        metrics['nse'] = 1 - (numerator / denominator) if denominator != 0 else np.nan
        
        # Mean Bias Error
        metrics['mbe'] = np.mean(merged_df[gldas_col] - merged_df[well_col])
        
        # R-squared
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            merged_df[well_col], merged_df[gldas_col]
        )
        metrics['r_squared'] = r_value**2
        metrics['slope'] = slope
        metrics['intercept'] = intercept
        metrics['p_value'] = p_value
        
        return metrics
    
    def analyze_site(self, site_no, well_file, gldas_file, output_dir):
        """
        Analyze comparison for a specific site.
        
        Parameters:
            site_no (str): Site number
            well_file (str): Path to processed well data
            gldas_file (str): Path to processed GLDAS data
            output_dir (str): Directory to save results
            
        Returns:
            dict: Metrics for this site
        """
        # Read data
        well_df = pd.read_csv(well_file)
        gldas_df = pd.read_csv(gldas_file)
        
        # Merge datasets
        merged = self.merge_datasets(well_df, gldas_df)
        
        if len(merged) < 5:  # Need enough data points for meaningful comparison
            print(f"Not enough overlapping data for site {site_no}")
            return None
        
        # Calculate metrics
        metrics = self.calculate_metrics(merged)
        metrics['site_no'] = site_no
        metrics['data_points'] = len(merged)
        
        # Save merged data
        os.makedirs(output_dir, exist_ok=True)
        merged_file = os.path.join(output_dir, f"{site_no}_comparison.csv")
        merged.to_csv(merged_file, index=False)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot time series
        ax.plot(merged['date'], merged['gw_anomaly_m'], 'b-', label='Well Data')
        ax.plot(merged['date'], merged['gldas_gws_anomaly'], 'r-', label='GLDAS Data')
        
        # Format axes
        ax.set_xlabel('Date')
        ax.set_ylabel('Groundwater Level Anomaly (m)')
        ax.set_title(f'Site {site_no}: Well vs GLDAS Groundwater Anomalies')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Add metrics text
        text = (f"Correlation: {metrics['correlation']:.3f}\n"
                f"RMSE: {metrics['rmse']:.3f} m\n"
                f"NSE: {metrics['nse']:.3f}\n"
                f"Data Points: {metrics['data_points']}")
        ax.text(0.02, 0.95, text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"{site_no}_comparison.png")
        plt.savefig(plot_file)
        plt.close()
        
        # Add file paths to metrics
        metrics['merged_file'] = merged_file
        metrics['plot_file'] = plot_file
        
        return metrics
    
    def analyze_all_sites(self, well_files, gldas_files, output_dir):
        """
        Analyze all sites and compile results.
        
        Parameters:
            well_files (dict): Dictionary of site numbers to well data files
            gldas_files (dict): Dictionary of site numbers to GLDAS data files
            output_dir (str): Directory to save results
            
        Returns:
            pandas.DataFrame: Compiled metrics for all sites
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze each site
        all_metrics = []
        
        # Get common sites
        common_sites = set(well_files.keys()).intersection(set(gldas_files.keys()))
        
        for site_no in common_sites:
            print(f"Analyzing site {site_no}")
            metrics = self.analyze_site(
                site_no, well_files[site_no], gldas_files[site_no], output_dir
            )
            
            if metrics is not None:
                all_metrics.append(metrics)
        
        # Compile metrics
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, "all_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        # Create summary plots
        if not metrics_df.empty:
            self.create_summary_plots(metrics_df, output_dir)
        
        return metrics_df
    
    def create_summary_plots(self, metrics_df, output_dir):
        """
        Create summary plots for all sites.
        
        Parameters:
            metrics_df (pandas.DataFrame): Metrics for all sites
            output_dir (str): Directory to save plots
        """
        # Histogram of correlation values
        plt.figure(figsize=(10, 6))
        sns.histplot(metrics_df['correlation'], kde=True)
        plt.title('Distribution of Correlation Values')
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "correlation_histogram.png"))
        plt.close()
        
        # Histogram of NSE values
        plt.figure(figsize=(10, 6))
        sns.histplot(metrics_df['nse'], kde=True)
        plt.title('Distribution of Nash-Sutcliffe Efficiency Values')
        plt.xlabel('NSE')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "nse_histogram.png"))
        plt.close()
        
        # Scatter plot of correlation vs RMSE
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=metrics_df, x='correlation', y='rmse')
        plt.title('Correlation vs RMSE')
        plt.xlabel('Correlation')
        plt.ylabel('RMSE (m)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "corr_vs_rmse.png"))
        plt.close()
        
        # Box plot of metrics
        metrics_long = pd.melt(
            metrics_df[['correlation', 'nse', 'r_squared']], 
            var_name='Metric', value_name='Value'
        )
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics_long, x='Metric', y='Value')
        plt.title('Distribution of Key Metrics')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "metrics_boxplot.png"))
        plt.close()