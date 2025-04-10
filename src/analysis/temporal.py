import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt

class TemporalAnalyzer:
    """Class for temporal analysis of well and GLDAS data."""
    
    def __init__(self):
        """Initialize the temporal analyzer."""
        pass
    
    def analyze_lag_correlation(self, df, col1, col2, max_lag=12):
        """
        Analyze lagged correlations between two time series.
        
        Parameters:
            df (pd.DataFrame): DataFrame with time series data
            col1 (str): First column name
            col2 (str): Second column name
            max_lag (int): Maximum lag to consider in months
            
        Returns:
            dict: Dictionary with lag correlation results
        """
        # Ensure data is sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # Get values (drop NaNs)
        mask = ~np.isnan(df[col1]) & ~np.isnan(df[col2])
        vals1 = df.loc[mask, col1].values
        vals2 = df.loc[mask, col2].values
        
        if len(vals1) < 24:  # Need enough data for meaningful lag analysis
            return {
                'optimal_lag': np.nan,
                'max_correlation': np.nan,
                'lag_correlations': {}
            }
        
        # Remove means
        vals1 = vals1 - np.mean(vals1)
        vals2 = vals2 - np.mean(vals2)
        
        # Calculate cross-correlation
        corr = signal.correlate(vals1, vals2, mode='full')
        lags = signal.correlation_lags(len(vals1), len(vals2), mode='full')
        
        # Normalize correlations
        norm_factor = len(vals1) * np.std(vals1) * np.std(vals2)
        if norm_factor > 0:
            corr = corr / norm_factor
        
        # Limit to specified max lag
        valid_indices = np.where((lags >= -max_lag) & (lags <= max_lag))[0]
        valid_corr = corr[valid_indices]
        valid_lags = lags[valid_indices]
        
        # Find optimal lag
        max_idx = np.argmax(np.abs(valid_corr))
        optimal_lag = valid_lags[max_idx]
        max_corr = valid_corr[max_idx]
        
        # Create lag correlation dictionary for all lags
        lag_corrs = {lag: corr for lag, corr in zip(valid_lags, valid_corr)}
        
        return {
            'optimal_lag': optimal_lag,
            'max_correlation': max_corr,
            'lag_correlations': lag_corrs
        }
    
    def calculate_trends(self, df, columns, date_column='date'):
        """
        Calculate linear trends for time series.
        
        Parameters:
            df (pd.DataFrame): DataFrame with time series data
            columns (list): List of column names to analyze
            date_column (str): Column name with dates
            
        Returns:
            dict: Dictionary with trend results
        """
        # Convert date to numeric values (years)
        df['date_numeric'] = pd.to_datetime(df[date_column]).astype(np.int64) / 10**9 / 86400 / 365.25
        
        trends = {}
        for column in columns:
            # Skip missing columns
            if column not in df.columns:
                continue
                
            # Create mask for non-NaN values
            mask = ~np.isnan(df[column])
            
            if mask.sum() < 2:  # Need at least 2 points for regression
                trends[column] = {
                    'slope': np.nan,
                    'intercept': np.nan,
                    'r_squared': np.nan,
                    'p_value': np.nan
                }
                continue
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df.loc[mask, 'date_numeric'], df.loc[mask, column]
            )
            
            trends[column] = {
                'slope': slope,  # Change per year
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err
            }
        
        # Remove temporary column
        df.drop('date_numeric', axis=1, inplace=True)
        
        return trends
    
    def analyze_seasonal_patterns(self, df, column, date_column='date'):
        """
        Analyze seasonal patterns in time series.
        
        Parameters:
            df (pd.DataFrame): DataFrame with time series data
            column (str): Column name to analyze
            date_column (str): Column name with dates
            
        Returns:
            dict: Dictionary with seasonal analysis results
        """
        # Ensure date column is datetime
        df['date_dt'] = pd.to_datetime(df[date_column])
        
        # Extract month
        df['month'] = df['date_dt'].dt.month
        
        # Calculate monthly statistics
        monthly_stats = df.groupby('month').agg({
            column: ['count', 'mean', 'std', 'min', 'max']
        })
        
        # Flatten column names
        monthly_stats.columns = [f"{column}_{stat}" for col, stat in monthly_stats.columns]
        monthly_stats = monthly_stats.reset_index()
        
        # Calculate annual cycle strength
        if monthly_stats[f"{column}_count"].min() >= 3:  # Need at least 3 years of data for each month
            annual_amplitude = (monthly_stats[f"{column}_mean"].max() - monthly_stats[f"{column}_mean"].min()) / 2
            annual_mean = monthly_stats[f"{column}_mean"].mean()
            annual_cycle_strength = annual_amplitude / annual_mean if annual_mean != 0 else np.nan
        else:
            annual_amplitude = np.nan
            annual_cycle_strength = np.nan
        
        # Identify peak and trough months
        if not monthly_stats[f"{column}_mean"].isna().all():
            peak_month = monthly_stats.loc[monthly_stats[f"{column}_mean"].idxmax(), 'month']
            trough_month = monthly_stats.loc[monthly_stats[f"{column}_mean"].idxmin(), 'month']
        else:
            peak_month = np.nan
            trough_month = np.nan
        
        # Clean up
        df.drop(['date_dt', 'month'], axis=1, inplace=True)
        
        return {
            'monthly_stats': monthly_stats,
            'annual_amplitude': annual_amplitude,
            'annual_cycle_strength': annual_cycle_strength,
            'peak_month': peak_month,
            'trough_month': trough_month
        }
        
    def analyze_anomalies(self, df, column, window=12, threshold=2):
        """
        Identify anomalies in time series.
        
        Parameters:
            df (pd.DataFrame): DataFrame with time series data
            column (str): Column name to analyze
            window (int): Rolling window size for mean/std calculation
            threshold (float): Number of standard deviations for anomaly detection
            
        Returns:
            dict: Dictionary with anomaly analysis results
        """
        # Create copy to avoid modifying original
        df_analysis = df.copy()
        
        # Ensure date is index for rolling operations
        if 'date' in df_analysis.columns:
            df_analysis = df_analysis.set_index('date')
        
        # Calculate rolling mean and std
        rolling_mean = df_analysis[column].rolling(window=window, center=True).mean()
        rolling_std = df_analysis[column].rolling(window=window, center=True).std()
        
        # Identify anomalies
        anomalies = np.abs(df_analysis[column] - rolling_mean) > (threshold * rolling_std)
        
        # Get anomaly details
        anomaly_indices = np.where(anomalies)[0]
        anomaly_dates = df_analysis.index[anomaly_indices]
        anomaly_values = df_analysis.loc[anomaly_dates, column].values
        anomaly_z_scores = (df_analysis.loc[anomaly_dates, column].values - 
                           rolling_mean.loc[anomaly_dates].values) / rolling_std.loc[anomaly_dates].values
        
        # Create anomaly dataframe
        anomaly_df = pd.DataFrame({
            'date': anomaly_dates,
            'value': anomaly_values,
            'z_score': anomaly_z_scores
        })
        
        return {
            'anomaly_count': len(anomaly_df),
            'anomaly_percentage': len(anomaly_df) / len(df) * 100,
            'anomalies': anomaly_df
        }