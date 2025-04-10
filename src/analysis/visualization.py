import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

class VisualizationGenerator:
    """Class for creating publication-quality visualizations."""
    
    def __init__(self, output_dir=None):
        """Initialize the visualization generator."""
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set visualization styles
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Define colors
        self.well_color = '#1f77b4'  # blue
        self.gldas_color = '#ff7f0e'  # orange
    
    def create_time_series_plot(self, merged_df, site_no, site_metadata=None, metrics=None, 
                               well_col='gw_anomaly_m', gldas_col='gldas_gws_anomaly'):
        """Create a publication-quality time series comparison plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Ensure date is datetime
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        
        # Plot well data
        ax1.plot(merged_df['date'], merged_df[well_col], '-', color=self.well_color, 
               marker='o', markersize=4, label='Well Data')
        
        # Plot GLDAS data
        ax1.plot(merged_df['date'], merged_df[gldas_col], '-', color=self.gldas_color, 
               marker='s', markersize=4, label='GLDAS Data')
        
        # Add trend lines if there are enough data points
        if len(merged_df) >= 12:
            from scipy import stats
            
            # Add well trend line
            if np.sum(~np.isnan(merged_df[well_col])) >= 3:
                x_numeric = mdates.date2num(merged_df['date'])
                mask = ~np.isnan(merged_df[well_col])
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_numeric[mask], merged_df.loc[mask, well_col]
                )
                trend_line = intercept + slope * x_numeric
                ax1.plot(merged_df['date'], trend_line, '--', color=self.well_color, alpha=0.7, 
                       linewidth=1.5, label='Well Trend')
            
            # Add GLDAS trend line
            if np.sum(~np.isnan(merged_df[gldas_col])) >= 3:
                x_numeric = mdates.date2num(merged_df['date'])
                mask = ~np.isnan(merged_df[gldas_col])
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_numeric[mask], merged_df.loc[mask, gldas_col]
                )
                trend_line = intercept + slope * x_numeric
                ax1.plot(merged_df['date'], trend_line, '--', color=self.gldas_color, alpha=0.7, 
                       linewidth=1.5, label='GLDAS Trend')
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        # Add metrics text box
        if metrics:
            metrics_text = (
                f"Correlation: {metrics.get('correlation', 'N/A'):.3f}\n"
                f"RMSE: {metrics.get('rmse', 'N/A'):.3f}\n"
                f"NSE: {metrics.get('nse', 'N/A'):.3f}"
            )
            
            # Add lag information if available
            if 'lag_months' in metrics:
                metrics_text += f"\nOptimal Lag: {metrics.get('lag_months', 'N/A')} months"
            
            ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        # Title and labels
        site_name = site_metadata.get('site_name', f'Site {site_no}') if site_metadata else f'Site {site_no}'
        ax1.set_title(f'Groundwater Anomaly Time Series: {site_name}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Groundwater Anomaly (m)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot difference in lower panel
        merged_df['difference'] = merged_df[gldas_col] - merged_df[well_col]
        ax2.bar(merged_df['date'], merged_df['difference'], width=20, color='gray', alpha=0.6)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Format y-axis for difference panel
        ax2.set_ylabel('Difference\n(GLDAS - Well)', fontsize=10)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        # Add mean and std of difference
        mean_diff = merged_df['difference'].mean()
        std_diff = merged_df['difference'].std()
        ax2.axhline(y=mean_diff, color='r', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_diff:.3f}')
        
        # Add shaded area for ±1 std
        ax2.axhspan(mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color='r',
                  label=f'±1σ: {std_diff:.3f}')
        
        ax2.legend(loc='lower right', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots
        
        # Save figure if output directory is set
        if self.output_dir:
            file_path = os.path.join(self.output_dir, f"{site_no}_time_series.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_scatter_plot(self, merged_df, site_no, site_metadata=None, metrics=None,
                           well_col='gw_anomaly_m', gldas_col='gldas_gws_anomaly'):
        """Create a publication-quality scatter plot with regression line."""
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create scatter plot with dates as colors
        scatter = ax.scatter(merged_df[well_col], merged_df[gldas_col], alpha=0.7, 
                          edgecolor='k', s=60, c=merged_df['date'], cmap='viridis')
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            merged_df[well_col].dropna(), merged_df[gldas_col].dropna()
        )
        
        # Get axis limits with padding
        min_val = min(merged_df[well_col].min(), merged_df[gldas_col].min())
        max_val = max(merged_df[well_col].max(), merged_df[gldas_col].max())
        range_val = max_val - min_val
        min_val = min_val - 0.1 * range_val
        max_val = max_val + 0.1 * range_val
        
        x_vals = np.array([min_val, max_val])
        y_vals = intercept + slope * x_vals
        
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, 
              label=f'y = {slope:.3f}x + {intercept:.3f}')
        
        # Add 1:1 line
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, 
              label='1:1 Line')
        
        # Add statistics text
        if metrics:
            stats_text = (
                f"$R = {metrics.get('correlation', r_value):.3f}$\n"
                f"$R^2 = {metrics.get('r_squared', r_value**2):.3f}$\n"
                f"$RMSE = {metrics.get('rmse', 'N/A'):.3f}$\n"
                f"$NSE = {metrics.get('nse', 'N/A'):.3f}$\n"
                f"$n = {len(merged_df)}$"
            )
        else:
            stats_text = (
                f"$R = {r_value:.3f}$\n"
                f"$R^2 = {r_value**2:.3f}$\n"
                f"$n = {len(merged_df)}$"
            )
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
              verticalalignment='top')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Set axis limits
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        # Add colorbar to show temporal evolution
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Date', fontsize=10)
        
        # Format colorbar ticks to show dates
        if len(merged_df) > 0:
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, pos: pd.to_datetime(x).strftime('%Y-%m')
            ))
        
        # Title and labels
        site_name = site_metadata.get('site_name', f'Site {site_no}') if site_metadata else f'Site {site_no}'
        ax.set_title(f'Well vs GLDAS: {site_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Well Groundwater Anomaly (m)', fontsize=12)
        ax.set_ylabel('GLDAS Groundwater Anomaly', fontsize=12)
        
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if output directory is set
        if self.output_dir:
            file_path = os.path.join(self.output_dir, f"{site_no}_scatter.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        return fig
    def create_summary_metrics_plot(self, metrics_df, output_dir=None):
        """Create summary plots of metrics across all sites."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        
        # Use output_dir from arguments if provided, otherwise use class attribute
        if output_dir is None:
            output_dir = self.output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Dictionary to store figures
        figures = {}
        
        # Set up the style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Histogram of correlation values
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(metrics_df['correlation'], kde=True, ax=ax1)
        ax1.set_title('Distribution of Correlation Values', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Correlation', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = (
            f"Mean: {metrics_df['correlation'].mean():.3f}\n"
            f"Median: {metrics_df['correlation'].median():.3f}\n"
            f"Std Dev: {metrics_df['correlation'].std():.3f}\n"
            f"N: {len(metrics_df)}"
        )
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                verticalalignment='top')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'correlation_histogram.png'), dpi=300, bbox_inches='tight')
        figures['correlation_histogram'] = fig1
        
        # 2. Histogram of RMSE values
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(metrics_df['rmse'], kde=True, ax=ax2)
        ax2.set_title('Distribution of RMSE Values', fontsize=14, fontweight='bold')
        ax2.set_xlabel('RMSE', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        stats_text = (
            f"Mean: {metrics_df['rmse'].mean():.3f}\n"
            f"Median: {metrics_df['rmse'].median():.3f}\n"
            f"Std Dev: {metrics_df['rmse'].std():.3f}\n"
            f"N: {len(metrics_df)}"
        )
        ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                verticalalignment='top')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'rmse_histogram.png'), dpi=300, bbox_inches='tight')
        figures['rmse_histogram'] = fig2
        
        # 3. Histogram of NSE values
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        # Filter out extreme NSE values for better visualization
        filtered_nse = metrics_df['nse'].copy()
        filtered_nse = filtered_nse[filtered_nse > -5]  # Remove extreme negative values
        
        sns.histplot(filtered_nse, kde=True, ax=ax3)
        ax3.set_title('Distribution of Nash-Sutcliffe Efficiency Values', fontsize=14, fontweight='bold')
        ax3.set_xlabel('NSE', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        stats_text = (
            f"Mean: {metrics_df['nse'].mean():.3f}\n"
            f"Median: {metrics_df['nse'].median():.3f}\n"
            f"Std Dev: {metrics_df['nse'].std():.3f}\n"
            f"N: {len(metrics_df)}"
        )
        ax3.text(0.02, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                verticalalignment='top')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'nse_histogram.png'), dpi=300, bbox_inches='tight')
        figures['nse_histogram'] = fig3
        
        # 4. Scatter plot of correlation vs RMSE
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=metrics_df, x='correlation', y='rmse', ax=ax4, alpha=0.6)
        ax4.set_title('Correlation vs RMSE', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Correlation', fontsize=12)
        ax4.set_ylabel('RMSE', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add correlation value between the two metrics
        corr_val = metrics_df['correlation'].corr(metrics_df['rmse'])
        ax4.text(0.02, 0.95, f"Correlation: {corr_val:.3f}", transform=ax4.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                verticalalignment='top')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'corr_vs_rmse.png'), dpi=300, bbox_inches='tight')
        figures['corr_vs_rmse'] = fig4
        
        # 5. Box plot of metrics
        metrics_long = metrics_df[['correlation', 'nse', 'r_squared']].copy()
        
        # Handle potential NaN values
        metrics_long = metrics_long.fillna(0)
        
        # Convert to long format for boxplot
        metrics_long = metrics_long.melt(var_name='Metric', value_name='Value')
        
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=metrics_long, x='Metric', y='Value', ax=ax5)
        ax5.set_title('Distribution of Key Metrics', fontsize=14, fontweight='bold')
        ax5.set_xlabel('', fontsize=12)
        ax5.set_ylabel('Value', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'metrics_boxplot.png'), dpi=300, bbox_inches='tight')
        figures['metrics_boxplot'] = fig5
        
        # Close figures to free memory (if we've saved them)
        if output_dir:
            for fig in figures.values():
                plt.close(fig)
        
        return figures
    def create_seasonal_plot(self, merged_df, site_no, site_metadata=None,
                            well_col='gw_anomaly_m', gldas_col='gldas_gws_anomaly'):
        """Create a seasonal pattern comparison plot."""
        # Add month column
        df = merged_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        
        # Calculate monthly statistics
        monthly_stats = df.groupby('month').agg({
            well_col: ['mean', 'std', 'count'],
            gldas_col: ['mean', 'std', 'count']
        })
        
        # Flatten column names
        monthly_stats.columns = [f"{col[0]}_{col[1]}" for col in monthly_stats.columns]
        monthly_stats = monthly_stats.reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot monthly means with standard deviation bands
        ax.plot(monthly_stats['month'], monthly_stats[f"{well_col}_mean"], 'o-', 
              color=self.well_color, linewidth=2, markersize=6,
              label='Well Data')
        
        ax.fill_between(
            monthly_stats['month'],
            monthly_stats[f"{well_col}_mean"] - monthly_stats[f"{well_col}_std"],
            monthly_stats[f"{well_col}_mean"] + monthly_stats[f"{well_col}_std"],
            alpha=0.2, color=self.well_color
        )
        
        ax.plot(monthly_stats['month'], monthly_stats[f"{gldas_col}_mean"], 's-', 
              color=self.gldas_color, linewidth=2, markersize=6,
              label='GLDAS Data')
        
        ax.fill_between(
            monthly_stats['month'],
            monthly_stats[f"{gldas_col}_mean"] - monthly_stats[f"{gldas_col}_std"],
            monthly_stats[f"{gldas_col}_mean"] + monthly_stats[f"{gldas_col}_std"],
            alpha=0.2, color=self.gldas_color
        )
        
        # Format x-axis as months
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Calculate seasonal correlation
        seasonal_corr = np.corrcoef(monthly_stats[f"{well_col}_mean"], monthly_stats[f"{gldas_col}_mean"])[0, 1]
        
        # Add correlation text
        ax.text(0.02, 0.95, f"Seasonal Correlation: {seasonal_corr:.3f}", 
              transform=ax.transAxes, fontsize=10,
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        # Title and labels
        site_name = site_metadata.get('site_name', f'Site {site_no}') if site_metadata else f'Site {site_no}'
        ax.set_title(f'Seasonal Patterns: {site_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Groundwater Anomaly', fontsize=12)
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if output directory is set
        if self.output_dir:
            file_path = os.path.join(self.output_dir, f"{site_no}_seasonal.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        return fig