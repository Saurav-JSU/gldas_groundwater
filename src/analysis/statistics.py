import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

class StatisticalAnalyzer:
    """Class for statistical analysis of well and GLDAS data."""
    
    def __init__(self, n_workers=None):
        """Initialize the statistical analyzer."""
        self.n_workers = n_workers or min(180, mp.cpu_count())
    
    def calculate_metrics(self, merged_df, well_column='gw_anomaly_m', gldas_column='gldas_gws_anomaly'):
        """Calculate comprehensive comparison metrics for a single site."""
        metrics = {}
        
        if len(merged_df) < 12:  # Require at least 12 months of data
            return {'insufficient_data': True}
        
        # Basic statistics
        metrics['data_points'] = len(merged_df)
        metrics['well_mean'] = merged_df[well_column].mean()
        metrics['gldas_mean'] = merged_df[gldas_column].mean()
        metrics['well_std'] = merged_df[well_column].std()
        metrics['gldas_std'] = merged_df[gldas_column].std()
        
        # Correlation metrics
        metrics['correlation'] = merged_df[well_column].corr(merged_df[gldas_column])
        metrics['spearman_corr'] = merged_df[well_column].corr(merged_df[gldas_column], method='spearman')
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            merged_df[well_column], merged_df[gldas_column]
        )
        metrics['slope'] = slope
        metrics['intercept'] = intercept
        metrics['r_squared'] = r_value**2
        metrics['p_value'] = p_value
        
        # Error metrics
        metrics['rmse'] = np.sqrt(np.mean((merged_df[well_column] - merged_df[gldas_column])**2))
        metrics['mae'] = np.mean(np.abs(merged_df[well_column] - merged_df[gldas_column]))
        
        # Nash-Sutcliffe Efficiency
        mean_obs = merged_df[well_column].mean()
        numerator = np.sum((merged_df[well_column] - merged_df[gldas_column])**2)
        denominator = np.sum((merged_df[well_column] - mean_obs)**2)
        metrics['nse'] = 1 - (numerator / denominator) if denominator != 0 else np.nan
        
        # Percent Bias
        sum_obs = np.sum(merged_df[well_column])
        if sum_obs != 0:
            metrics['pbias'] = 100 * np.sum(merged_df[gldas_column] - merged_df[well_column]) / sum_obs
        else:
            metrics['pbias'] = np.nan
        
        # Kling-Gupta Efficiency
        if metrics['well_std'] > 0 and metrics['gldas_std'] > 0 and metrics['well_mean'] != 0:
            r = metrics['correlation']
            alpha = metrics['gldas_std'] / metrics['well_std']
            beta = metrics['gldas_mean'] / metrics['well_mean']
            metrics['kge'] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            metrics['kge'] = np.nan
        
        # Cross-correlation and lag analysis
        if len(merged_df) >= 24:  # Need enough data for meaningful lag analysis
            from scipy import signal
            
            # Create array of values
            well_vals = merged_df[well_column].values
            gldas_vals = merged_df[gldas_column].values
            
            # Remove means
            well_vals = well_vals - np.mean(well_vals)
            gldas_vals = gldas_vals - np.mean(gldas_vals)
            
            # Calculate cross-correlation (limit lag to ±12 months)
            max_lag = min(12, len(merged_df) // 3)
            
            corr = signal.correlate(well_vals, gldas_vals, mode='full')
            lags = signal.correlation_lags(len(well_vals), len(gldas_vals), mode='full')
            
            # Normalize by the product of the standard deviations and length
            norm_factor = len(well_vals) * np.std(well_vals) * np.std(gldas_vals)
            if norm_factor > 0:
                corr = corr / norm_factor
            
            # Find lag with maximum correlation
            valid_indices = np.where((lags >= -max_lag) & (lags <= max_lag))[0]
            valid_corr = corr[valid_indices]
            valid_lags = lags[valid_indices]
            
            if len(valid_corr) > 0:
                max_idx = np.argmax(np.abs(valid_corr))
                metrics['max_cross_corr'] = valid_corr[max_idx]
                metrics['lag_months'] = valid_lags[max_idx]
        
        return metrics
    
    def _process_site(self, site_info):
        """
        Process a single site.
        
        Parameters:
            site_info (tuple): (site_no, file_path, metadata_df)
            
        Returns:
            dict: Metrics for this site
        """
        site_no, file_path, metadata_df = site_info
        
        try:
            # Read merged data
            merged_df = pd.read_csv(file_path)
            
            # Convert date column to datetime
            if 'date' in merged_df.columns:
                merged_df['date'] = pd.to_datetime(merged_df['date'])
            
            # Get site metadata if available
            site_metadata = None
            if metadata_df is not None and not metadata_df.empty:
                site_rows = metadata_df[metadata_df['site_no'] == site_no]
                if len(site_rows) > 0:
                    site_metadata = site_rows.iloc[0].to_dict()
            
            # Calculate metrics
            metrics = self.calculate_metrics(merged_df)
            metrics['site_no'] = site_no
            
            # Add metadata if available
            if site_metadata:
                for key, value in site_metadata.items():
                    if key not in metrics and key != 'site_no':
                        metrics[key] = value
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing site {site_no}: {e}")
            return {'site_no': site_no, 'error': str(e)}
    
    def analyze_all_sites(self, merged_files, metadata_df=None, output_dir=None):
        """
        Analyze all sites in parallel.
        
        Parameters:
            merged_files (dict): Dictionary mapping site numbers to merged data files
            metadata_df (pd.DataFrame, optional): DataFrame with metadata for all sites
            output_dir (str, optional): Directory to save results
            
        Returns:
            pd.DataFrame: DataFrame with metrics for all sites
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Prepare arguments for mapping
        site_info = [(site_no, file_path, metadata_df) for site_no, file_path in merged_files.items()]
        
        # Process in parallel
        print(f"Analyzing {len(merged_files)} sites in parallel...")
        all_metrics = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(tqdm(
                executor.map(self._process_site, site_info),
                total=len(site_info),
                desc="Analyzing sites"
            ))
            
            all_metrics.extend([m for m in results if 'site_no' in m])
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save results if output directory is provided
        if output_dir and not metrics_df.empty:
            metrics_df.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)
        
        return metrics_df