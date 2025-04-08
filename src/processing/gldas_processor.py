import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

class GLDASProcessor:
    """Class to process GLDAS data for comparison with well data."""
    
    def __init__(self, config):
        """
        Initialize the GLDAS processor.
        
        Parameters:
            config (dict): Configuration dictionary
        """
        self.config = config
    
    def resample_to_monthly(self, gldas_df):
        """
        Resample GLDAS data to monthly averages if needed.
        
        Parameters:
            gldas_df (pandas.DataFrame): GLDAS data
            
        Returns:
            pandas.DataFrame: Monthly resampled data
        """
        # Ensure date column is datetime
        gldas_df['date'] = pd.to_datetime(gldas_df['date'])
        
        # Set date as index
        df = gldas_df.set_index('date')
        
        # Group by year and month and calculate mean
        monthly_df = df.resample('MS').mean()
        
        # Reset index to get date as column
        monthly_df = monthly_df.reset_index()
        
        return monthly_df
    
    def process_all_gldas(self, gldas_dir, output_dir):
        """
        Process all GLDAS data files.
        
        Parameters:
            gldas_dir (str): Directory containing GLDAS data CSV files
            output_dir (str): Directory to save processed files
            
        Returns:
            dict: Dictionary mapping site numbers to processed file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all GLDAS CSV files
        gldas_files = glob.glob(os.path.join(gldas_dir, "*_gldas.csv"))
        
        processed_files = {}
        
        for gldas_file in gldas_files:
            try:
                # Extract site number from filename
                site_no = os.path.basename(gldas_file).split('_')[0]
                
                # Read GLDAS data
                gldas_df = pd.read_csv(gldas_file)
                
                # Process data
                if not gldas_df.empty:
                    # Resample to monthly if not already
                    monthly_df = self.resample_to_monthly(gldas_df)
                    
                    # Save processed data
                    output_file = os.path.join(output_dir, f"{site_no}_gldas_monthly.csv")
                    monthly_df.to_csv(output_file, index=False)
                    
                    processed_files[site_no] = output_file
                    
            except Exception as e:
                print(f"Error processing GLDAS data {gldas_file}: {e}")
        
        return processed_files