import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import calendar

class ParallelProcessor:
    """Class to process well and GLDAS data in parallel using multiprocessing."""
    
    def __init__(self, config, n_workers=None):
        """
        Initialize the parallel processor.
        
        Parameters:
            config (dict): Configuration dictionary
            n_workers (int, optional): Number of worker processes to use
        """
        self.config = config
        # Use up to 180 cores by default (leaving some for system)
        self.n_workers = n_workers or min(180, mp.cpu_count())
        print(f"Using {self.n_workers} CPU cores for parallel processing")
    
    def _process_well_site(self, well_file):
        """
        Process a single well data file.
        
        Parameters:
            well_file (str): Path to well data file
            
        Returns:
            tuple: (site_no, processed_df) or (site_no, None) if processing fails
        """
        try:
            # Extract site number from filename
            site_no = os.path.basename(well_file).split('.')[0]
            
            # Read well data
            well_df = pd.read_csv(well_file)
            
            if well_df.empty:
                return site_no, None
            
            # Convert date column to datetime
            if 'lev_dt' in well_df.columns:
                well_df['date'] = pd.to_datetime(well_df['lev_dt'], errors='coerce')
            elif 'datetime' in well_df.columns:
                well_df['date'] = pd.to_datetime(well_df['datetime'], errors='coerce')
            else:
                # Try to find any column that might be a date
                date_cols = [col for col in well_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    well_df['date'] = pd.to_datetime(well_df[date_cols[0]], errors='coerce')
                else:
                    return site_no, None
            
            # Drop rows with invalid dates
            well_df = well_df.dropna(subset=['date'])
            
            if len(well_df) < 3:  # Need at least 3 data points
                return site_no, None
            
            # Process depth data if available
            if 'lev_va' in well_df.columns:
                # Convert from feet to meters (handle non-numeric values)
                well_df['depth_m'] = pd.to_numeric(well_df['lev_va'], errors='coerce') * 0.3048
            elif 'depth_m' not in well_df.columns:
                # Check for any column that might contain depth data
                depth_cols = [col for col in well_df.columns if 'depth' in col.lower() or 'lev' in col.lower()]
                if depth_cols:
                    well_df['depth_m'] = pd.to_numeric(well_df[depth_cols[0]], errors='coerce')
                    # Check if conversion to meters is needed
                    if 'ft' in depth_cols[0].lower() or 'feet' in depth_cols[0].lower():
                        well_df['depth_m'] = well_df['depth_m'] * 0.3048
            
            # Drop rows with invalid depth
            if 'depth_m' in well_df.columns:
                well_df = well_df.dropna(subset=['depth_m'])
            
            if len(well_df) < 3:  # Need at least 3 data points after cleaning
                return site_no, None
            
            # Select only numeric columns for resampling (plus date)
            numeric_cols = ['date']
            for col in well_df.columns:
                if col != 'date' and pd.api.types.is_numeric_dtype(well_df[col]):
                    numeric_cols.append(col)
            
            well_df = well_df[numeric_cols].copy()
            
            # Resample to monthly
            well_df = well_df.sort_values('date')
            monthly_df = self._resample_to_monthly(well_df)
            
            # Calculate anomalies if not present
            if 'gw_anomaly_m' not in monthly_df.columns and 'depth_m' in monthly_df.columns:
                mean_depth = monthly_df['depth_m'].mean()
                # Invert sign for consistent interpretation (positive = more water)
                monthly_df['gw_anomaly_m'] = -(monthly_df['depth_m'] - mean_depth)
            
            # Add site information
            monthly_df['site_no'] = site_no
            
            return site_no, monthly_df
            
        except Exception as e:
            print(f"Error processing well data {well_file}: {e}")
            return site_no, None
    
    def _process_gldas_site(self, gldas_file):
        """
        Process a single GLDAS data file.
        
        Parameters:
            gldas_file (str): Path to GLDAS data file
            
        Returns:
            tuple: (site_no, processed_df) or (site_no, None) if processing fails
        """
        try:
            # Extract site number from filename
            site_no = os.path.basename(gldas_file).split('_')[0] if '_' in os.path.basename(gldas_file) else os.path.basename(gldas_file).split('.')[0]
            
            # Read GLDAS data
            gldas_df = pd.read_csv(gldas_file)
            
            if gldas_df.empty:
                return site_no, None
            
            # Process system:index to get date
            if 'system:index' in gldas_df.columns:
                # Convert system:index to string (in case it's not already)
                gldas_df['system:index'] = gldas_df['system:index'].astype(str)
                
                # Parse dates with error handling
                gldas_df['date'] = gldas_df['system:index'].apply(
                    lambda x: self._parse_gldas_date(x) if isinstance(x, str) and len(x) >= 8 else np.nan
                )
            elif 'date' not in gldas_df.columns:
                # Try to find any column that might be a date
                date_cols = [col for col in gldas_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    gldas_df['date'] = pd.to_datetime(gldas_df[date_cols[0]], errors='coerce')
                else:
                    return site_no, None
            
            # Drop rows with invalid dates
            gldas_df = gldas_df.dropna(subset=['date'])
            
            if len(gldas_df) < 3:  # Need at least 3 data points
                return site_no, None
            
            # Process GLDAS data - get the right column
            if 'GWS_tavg' in gldas_df.columns:
                gldas_df['gldas_gws'] = pd.to_numeric(gldas_df['GWS_tavg'], errors='coerce')
                
                # Apply scale factor if defined in config
                scale_factor = self.config.get('gldas', {}).get('scale_factor')
                if scale_factor:
                    gldas_df['gldas_gws'] = gldas_df['gldas_gws'] * scale_factor
            else:
                # Try to find any column that might contain GLDAS data
                gws_cols = [col for col in gldas_df.columns if 'gws' in col.lower() or 'groundwater' in col.lower()]
                if gws_cols:
                    gldas_df['gldas_gws'] = pd.to_numeric(gldas_df[gws_cols[0]], errors='coerce')
                else:
                    return site_no, None
            
            # Drop rows with invalid data
            gldas_df = gldas_df.dropna(subset=['gldas_gws'])
            
            if len(gldas_df) < 3:  # Need at least 3 data points after cleaning
                return site_no, None
            
            # Select only numeric columns for resampling (plus date)
            numeric_cols = ['date']
            for col in gldas_df.columns:
                if col != 'date' and pd.api.types.is_numeric_dtype(gldas_df[col]):
                    numeric_cols.append(col)
            
            gldas_df = gldas_df[numeric_cols].copy()
            
            # Resample to monthly
            gldas_df = gldas_df.sort_values('date')
            monthly_df = self._resample_to_monthly(gldas_df)
            
            # Calculate anomalies
            if 'gldas_gws_anomaly' not in monthly_df.columns and 'gldas_gws' in monthly_df.columns:
                mean_gws = monthly_df['gldas_gws'].mean()
                monthly_df['gldas_gws_anomaly'] = monthly_df['gldas_gws'] - mean_gws
            
            # Add site information
            monthly_df['site_no'] = site_no
            
            return site_no, monthly_df
            
        except Exception as e:
            print(f"Error processing GLDAS data {gldas_file}: {e}")
            return site_no, None
    
    def _parse_gldas_date(self, date_str):
        """
        Parse a GLDAS date string to a datetime object.
        
        Parameters:
            date_str (str): Date string in format 'YYYYMMDD'
            
        Returns:
            datetime: Parsed datetime object or NaT if parsing fails
        """
        try:
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                return pd.Timestamp(year=year, month=month, day=day)
            return pd.NaT
        except (ValueError, TypeError):
            return pd.NaT
    
    def _resample_to_monthly(self, df):
        """
        Resample data to monthly averages.
        
        Parameters:
            df (pandas.DataFrame): DataFrame with a 'date' column
            
        Returns:
            pandas.DataFrame: Monthly resampled data
        """
        try:
            # Check if date is already datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            
            # Set date as index
            df_indexed = df.set_index('date')
            
            # Remove any non-numeric columns before resampling
            numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
            df_numeric = df_indexed[numeric_cols]
            
            # Group by year and month and calculate mean
            monthly_df = df_numeric.resample('MS').mean()
            
            # Reset index to get date as column
            monthly_df = monthly_df.reset_index()
            
            # Add month-end dates for easier joining
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
            
        except Exception as e:
            print(f"Error in resampling: {e}")
            return pd.DataFrame()
    
    def process_all_wells(self, wells_dir, output_dir):
        """
        Process all well data files in parallel.
        
        Parameters:
            wells_dir (str): Directory containing well data CSV files
            output_dir (str): Directory to save processed files
            
        Returns:
            dict: Dictionary mapping site numbers to processed file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all well CSV files
        well_files = glob.glob(os.path.join(wells_dir, "*.csv"))
        well_files = [f for f in well_files if not "all_site_metrics" in f]
        
        processed_files = {}
        
        # Process in parallel
        print(f"Processing {len(well_files)} well files in parallel...")
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Map files to processor function with progress bar
            results = list(tqdm(
                executor.map(self._process_well_site, well_files),
                total=len(well_files),
                desc="Processing well data"
            ))
            
            # Process results
            for site_no, processed_df in results:
                if processed_df is not None and not processed_df.empty:
                    # Save processed data
                    output_file = os.path.join(output_dir, f"{site_no}_monthly.csv")
                    processed_df.to_csv(output_file, index=False)
                    processed_files[site_no] = output_file
        
        print(f"Successfully processed {len(processed_files)} well files")
        return processed_files
    
    def process_all_gldas(self, gldas_dir, output_dir):
        """
        Process all GLDAS data files in parallel.
        
        Parameters:
            gldas_dir (str): Directory containing GLDAS data CSV files
            output_dir (str): Directory to save processed files
            
        Returns:
            dict: Dictionary mapping site numbers to processed file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all GLDAS CSV files
        gldas_files = glob.glob(os.path.join(gldas_dir, "*.csv"))
        
        processed_files = {}
        
        # Process in parallel
        print(f"Processing {len(gldas_files)} GLDAS files in parallel...")
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Map files to processor function with progress bar
            results = list(tqdm(
                executor.map(self._process_gldas_site, gldas_files),
                total=len(gldas_files),
                desc="Processing GLDAS data"
            ))
            
            # Process results
            for site_no, processed_df in results:
                if processed_df is not None and not processed_df.empty:
                    # Save processed data
                    output_file = os.path.join(output_dir, f"{site_no}_gldas_monthly.csv")
                    processed_df.to_csv(output_file, index=False)
                    processed_files[site_no] = output_file
        
        print(f"Successfully processed {len(processed_files)} GLDAS files")
        return processed_files

    def _merge_site(self, args):
        """
        Merge well and GLDAS data for a single site.
        
        Parameters:
            args (tuple): (site_no, well_file, gldas_file, output_dir)
            
        Returns:
            tuple: (site_no, merged_file_path) or (site_no, None) if merging fails
        """
        site_no, well_file, gldas_file, output_dir = args
        
        try:
            # Read data
            well_df = pd.read_csv(well_file)
            gldas_df = pd.read_csv(gldas_file)
            
            # Ensure date columns are datetime
            well_df['date'] = pd.to_datetime(well_df['date'], errors='coerce')
            gldas_df['date'] = pd.to_datetime(gldas_df['date'], errors='coerce')
            
            # Drop rows with invalid dates
            well_df = well_df.dropna(subset=['date'])
            gldas_df = gldas_df.dropna(subset=['date'])
            
            # Merge on date
            merged = pd.merge(well_df, gldas_df, on='date', how='inner', 
                            suffixes=('_well', '_gldas'))
            
            if len(merged) < 3:  # Require at least 3 months of overlap
                return site_no, None
            
            # Save merged data
            output_file = os.path.join(output_dir, f"{site_no}_merged.csv")
            merged.to_csv(output_file, index=False)
            
            return site_no, output_file
        
        except Exception as e:
            print(f"Error merging data for site {site_no}: {e}")
            return site_no, None

    def merge_datasets(self, well_files, gldas_files, output_dir):
        """
        Merge well and GLDAS datasets for all common sites in parallel.
        
        Parameters:
            well_files (dict): Dictionary mapping site numbers to well data files
            gldas_files (dict): Dictionary mapping site numbers to GLDAS data files
            output_dir (str): Directory to save merged files
            
        Returns:
            dict: Dictionary mapping site numbers to merged file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find common sites
        common_sites = set(well_files.keys()).intersection(set(gldas_files.keys()))
        print(f"Found {len(common_sites)} common sites for merging")
        
        merged_files = {}
        
        # Check if we have any sites to merge
        if not common_sites:
            return merged_files
        
        # Prepare arguments for mapping
        merge_args = [(site_no, well_files[site_no], gldas_files[site_no], output_dir) 
                    for site_no in common_sites]
            
        # Merge in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Map sites to merge function with progress bar
            results = list(tqdm(
                executor.map(self._merge_site, merge_args),
                total=len(common_sites),
                desc="Merging datasets"
            ))
            
            # Process results
            for site_no, merged_file in results:
                if merged_file is not None:
                    merged_files[site_no] = merged_file
        
        print(f"Successfully merged {len(merged_files)} sites")
        return merged_files