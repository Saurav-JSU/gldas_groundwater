import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import calendar

class WellDataProcessor:
    """Class to process well data for comparison with GLDAS."""
    
    def __init__(self, config):
        """
        Initialize the well data processor.
        
        Parameters:
            config (dict): Configuration dictionary
        """
        self.config = config
    
    def resample_to_monthly(self, well_df):
        """
        Resample well data to monthly averages.
        
        Parameters:
            well_df (pandas.DataFrame): Well data
            
        Returns:
            pandas.DataFrame: Monthly resampled data
        """
        # Ensure date column is datetime
        well_df['date'] = pd.to_datetime(well_df['date'])
        
        # Set date as index
        df = well_df.set_index('date')
        
        # Group by year and month and calculate mean
        monthly_df = df.resample('MS').mean()
        
        # Reset index to get date as column
        monthly_df = monthly_df.reset_index()
        
        # Add month-end dates for easier joining with GLDAS
        monthly_df['year'] = monthly_df['date'].dt.year
        monthly_df['month'] = monthly_df['date'].dt.month
        monthly_df['last_day'] = monthly_df.apply(
            lambda row: calendar.monthrange(row['year'], row['month'])[1], axis=1
        )
        monthly_df['month_end'] = pd.to_datetime(
            monthly_df['year'].astype(str) + '-' + 
            monthly_df['month'].astype(str) + '-' + 
            monthly_df['last_day'].astype(str)
        )
        
        return monthly_df
    
    def process_all_wells(self, wells_dir, output_dir):
        """
        Process all well data files.
        
        Parameters:
            wells_dir (str): Directory containing well data CSV files
            output_dir (str): Directory to save processed files
            
        Returns:
            dict: Dictionary mapping site numbers to processed file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all well CSV files
        well_files = glob.glob(os.path.join(wells_dir, "*.csv"))
        well_files = [f for f in well_files if not f.endswith("metadata.csv")]
        
        processed_files = {}
        
        for well_file in well_files:
            try:
                # Extract site number from filename
                site_no = os.path.basename(well_file).split('.')[0]
                
                # Read well data
                well_df = pd.read_csv(well_file)
                
                # Process data
                if not well_df.empty:
                    # Resample to monthly
                    monthly_df = self.resample_to_monthly(well_df)
                    
                    # Save processed data
                    output_file = os.path.join(output_dir, f"{site_no}_monthly.csv")
                    monthly_df.to_csv(output_file, index=False)
                    
                    processed_files[site_no] = output_file
                    
            except Exception as e:
                print(f"Error processing well data {well_file}: {e}")
        
        return processed_files
    
    def calculate_anomalies(self, well_df, method='mean_removal'):
        """
        Calculate groundwater level anomalies.
        
        Parameters:
            well_df (pandas.DataFrame): Well data
            method (str): Method to calculate anomalies ('mean_removal' or 'detrend')
            
        Returns:
            pandas.DataFrame: Data with anomalies
        """
        df = well_df.copy()
        
        if method == 'mean_removal':
            # Simple mean removal
            if 'gw_anomaly_m' not in df.columns and 'depth_m' in df.columns:
                mean_depth = df['depth_m'].mean()
                df['gw_anomaly_m'] = -(df['depth_m'] - mean_depth)
                
        elif method == 'detrend':
            # Linear detrending
            if 'depth_m' in df.columns:
                # Convert date to numeric for regression
                if 'date' in df.columns:
                    df['date_num'] = pd.to_datetime(df['date']).astype(np.int64)
                else:
                    df['date_num'] = pd.to_datetime(df.index).astype(np.int64)
                
                # Fit linear trend
                from scipy import stats
                slope, intercept, _, _, _ = stats.linregress(df['date_num'], df['depth_m'])
                
                # Calculate trend
                trend = intercept + slope * df['date_num']
                
                # Remove trend (negative to convert depth to level)
                df['gw_anomaly_m'] = -(df['depth_m'] - trend)
                
                # Remove temporary column
                df = df.drop('date_num', axis=1)
        
        return df