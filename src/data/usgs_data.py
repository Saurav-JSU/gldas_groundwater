import pandas as pd
import os
import time
from dataretrieval import nwis
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from datetime import datetime

class USGSDataRetriever:
    """Class to retrieve USGS groundwater data using the dataretrieval package."""
    
    def __init__(self, config):
        """
        Initialize the USGS data retriever.
        
        Parameters:
            config (dict): Configuration dictionary
        """
        self.config = config
    
    def get_sites_in_state(self, state_code):
        """
        Get groundwater sites for a specific state.
        
        Parameters:
            state_code (str): Two-letter state code
            
        Returns:
            pandas.DataFrame: DataFrame containing site information
        """
        print(f"Searching for groundwater sites in {state_code}...")
        
        try:
            # More targeted query to find sites with likely data
            sites_df, _ = nwis.get_info(
                stateCd=state_code,
                siteType="GW",
                parameterCd=self.config['usgs']['parameter_code'],
                siteStatus="active"
            )
            
            print(f"Found {len(sites_df)} sites in {state_code}")
            return sites_df
        except Exception as e:
            print(f"Error getting sites for state {state_code}: {e}")
            return pd.DataFrame()
    
    def analyze_site_data(self, site_no):
        """
        Analyze data quality for a site and retrieve its geographic coordinates.
        """
        try:
            # First get site info to retrieve coordinates
            site_info_df, _ = nwis.get_info(sites=site_no)
            
            if site_info_df.empty:
                return None
                
            # Extract lat/lon
            lat = site_info_df['dec_lat_va'].iloc[0] if 'dec_lat_va' in site_info_df.columns else None
            lon = site_info_df['dec_long_va'].iloc[0] if 'dec_long_va' in site_info_df.columns else None
            site_name = site_info_df['station_nm'].iloc[0] if 'station_nm' in site_info_df.columns else None
            
            # Skip if coordinates are missing
            if lat is None or lon is None:
                print(f"Site {site_no} is missing coordinates, skipping")
                return None
            
            # Get groundwater level data
            gw_df, _ = nwis.get_gwlevels(
                sites=site_no,
                start=self.config['time']['start_date'],
                end=self.config['time']['end_date'],
                datetime_index=False  # Fix for incomplete dates warning
            )
            
            if gw_df.empty:
                return None
                    
            # Convert to proper datetime if not already
            if 'lev_dt' in gw_df.columns:
                gw_df['datetime'] = pd.to_datetime(gw_df['lev_dt'])
            
            # Basic metrics
            record_count = len(gw_df)
            
            # If doesn't meet minimum requirement, skip further analysis
            if record_count < self.config['usgs']['min_data_points']:
                return None
                    
            # Time coverage analysis
            start_date = gw_df['datetime'].min()
            end_date = gw_df['datetime'].max()
            date_range = (end_date - start_date).days
            
            # Calculate time coverage percentage
            requested_start = pd.to_datetime(self.config['time']['start_date'])
            requested_end = pd.to_datetime(self.config['time']['end_date'])
            requested_range = (requested_end - requested_start).days
            
            coverage_percentage = min(100, (date_range / requested_range) * 100)
            
            # Null value analysis
            if 'lev_va' in gw_df.columns:
                null_percentage = (gw_df['lev_va'].isna().sum() / record_count) * 100
            else:
                null_percentage = 100
            
            # Frequency analysis
            if record_count > 1:
                # Sort by date
                gw_df = gw_df.sort_values('datetime')
                
                # Calculate time differences
                gw_df['time_diff'] = gw_df['datetime'].diff()
                
                # Get median time difference in days
                median_interval = gw_df['time_diff'].median().days if hasattr(gw_df['time_diff'].median(), 'days') else None
                
                # Determine frequency
                if median_interval is not None:
                    if median_interval <= 1:
                        frequency = "daily"
                    elif 1 < median_interval <= 7:
                        frequency = "weekly"
                    elif 7 < median_interval <= 15:
                        frequency = "biweekly"
                    elif 15 < median_interval <= 45:
                        frequency = "monthly"
                    elif 45 < median_interval <= 100:
                        frequency = "quarterly"
                    else:
                        frequency = "infrequent"
                else:
                    frequency = "unknown"
            else:
                frequency = "single"
                median_interval = None
            
            # Calculate data quality score
            quality_score = (
                min(100, record_count / 10) * 0.4 +
                coverage_percentage * 0.3 +
                (100 - null_percentage) * 0.2 +
                (50 if frequency in ["daily", "weekly"] else 
                30 if frequency == "biweekly" else
                20 if frequency == "monthly" else
                10 if frequency == "quarterly" else 5) * 0.1
            )
            
            # Return the metrics with lat/lon included
            return {
                'site_no': site_no,
                'site_name': site_name,
                'latitude': lat,
                'longitude': lon,
                'record_count': record_count,
                'start_date': start_date,
                'end_date': end_date,
                'date_range_days': date_range,
                'coverage_percentage': coverage_percentage,
                'null_percentage': null_percentage,
                'frequency': frequency,
                'median_interval_days': median_interval,
                'quality_score': quality_score,
                'data': gw_df
            }
        except Exception as e:
            print(f"Error analyzing site {site_no}: {e}")
            return None
    
    def process_state_sites(self, state_sites):
        """
        Process a batch of sites from a state.
        
        Parameters:
            state_sites (list): List of site numbers
            
        Returns:
            list: List of site metrics dictionaries
        """
        results = []
        for site_no in state_sites:
            metrics = self.analyze_site_data(site_no)
            if metrics is not None:
                results.append(metrics)
        return results
    
    def download_all_sites_data(self, output_dir):
        """
        Download data with improved site selection based on data quality.
        
        Parameters:
            output_dir (str): Directory to save the data
            
        Returns:
            pandas.DataFrame: Metadata for all sites with valid data
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        
        # Define states in the Mississippi River Basin
        states = ['MN', 'WI', 'IA', 'IL', 'MO', 'KY', 'TN', 'AR', 'MS', 'LA']
        
        # Set up multiprocessing pool
        num_processes = min(128, mp.cpu_count())  # Using up to 128 cores
        print(f"Using {num_processes} CPU cores for processing")
        pool = mp.Pool(processes=num_processes)
        
        all_metrics = []
        
        # Process each state
        for state in states:
            # Get sites for this state
            state_sites_df = self.get_sites_in_state(state)
            
            if state_sites_df.empty:
                continue
                
            # Get site numbers
            state_site_numbers = state_sites_df['site_no'].unique().tolist()
            total_state_sites = len(state_site_numbers)
            
            print(f"Analyzing {total_state_sites} sites in {state}...")
            
            # Calculate number of sites per batch for better load balancing
            batch_size = max(1, min(100, total_state_sites // (num_processes * 2)))
            site_batches = [state_site_numbers[i:i + batch_size] for i in range(0, total_state_sites, batch_size)]
            
            # Process site batches in parallel with progress tracking
            state_results = []
            for result in tqdm(pool.imap_unordered(self.process_state_sites, site_batches), 
                             total=len(site_batches),
                             desc=f"Processing {state}"):
                state_results.extend(result)
            
            print(f"Found {len(state_results)} usable sites in {state}")
            
            # Add to overall results
            all_metrics.extend(state_results)
            
            # Save state metrics for reference
            if state_results:
                state_metrics_df = pd.DataFrame([
                    {k: v for k, v in m.items() if k != 'data'} 
                    for m in state_results
                ])
                state_metrics_df.to_csv(os.path.join(output_dir, "metrics", f"{state}_metrics.csv"), index=False)
        
        pool.close()
        pool.join()
        
        # Compile all metrics
        if not all_metrics:
            print("No sites with sufficient data found")
            return pd.DataFrame()
            
        print(f"Total sites with usable data: {len(all_metrics)}")
        
        # Create a DataFrame for sorting and analysis
        metrics_df = pd.DataFrame([
            {k: v for k, v in m.items() if k != 'data'} 
            for m in all_metrics
        ])
        
        # Save all metrics
        metrics_df.to_csv(os.path.join(output_dir, "all_site_metrics.csv"), index=False)
        
        # Sort by quality score
        metrics_df = metrics_df.sort_values('quality_score', ascending=False)
        
        # Determine how many sites to save based on config
        max_sites = self.config.get('max_sites')
        if max_sites and len(metrics_df) > max_sites:
            print(f"Selecting top {max_sites} sites by quality score")
            metrics_df = metrics_df.head(max_sites)
        
        # Save final selected sites and their data
        metadata = []
        for _, row in metrics_df.iterrows():
            site_no = row['site_no']
            
            # First get site info to retrieve coordinates
            try:
                site_info_df, _ = nwis.get_info(sites=site_no)
                
                # Extract lat/lon
                latitude = site_info_df['dec_lat_va'].iloc[0] if 'dec_lat_va' in site_info_df.columns else None
                longitude = site_info_df['dec_long_va'].iloc[0] if 'dec_long_va' in site_info_df.columns else None
                site_name = site_info_df['station_nm'].iloc[0] if 'station_nm' in site_info_df.columns else None
                
                # Skip sites without coordinates
                if latitude is None or longitude is None:
                    print(f"Site {site_no} is missing coordinates, skipping")
                    continue
            except Exception as e:
                print(f"Error getting coordinates for site {site_no}: {e}")
                continue
                
            # Find the original data
            site_data = next((m['data'] for m in all_metrics if m['site_no'] == site_no), None)
            
            if site_data is not None:
                # Process data
                site_data['site_no'] = site_no
                
                # Convert depths to meters if necessary
                if 'lev_va' in site_data.columns:
                    site_data['depth_m'] = site_data['lev_va'] * 0.3048  # Convert ft to m
                    
                    # Compute anomalies
                    mean_depth = site_data['depth_m'].mean()
                    site_data['gw_anomaly_m'] = -(site_data['depth_m'] - mean_depth)
                
                # Save to CSV
                output_file = os.path.join(output_dir, f"{site_no}.csv")
                site_data.to_csv(output_file, index=False)
                
                # Add to metadata with coordinates
                metadata.append({
                    'site_no': site_no,
                    'site_name': site_name,
                    'latitude': latitude,
                    'longitude': longitude,
                    'record_count': row['record_count'],
                    'start_date': row['start_date'],
                    'end_date': row['end_date'],
                    'coverage_percentage': row['coverage_percentage'],
                    'null_percentage': row['null_percentage'],
                    'frequency': row['frequency'],
                    'quality_score': row['quality_score'],
                    'file_path': output_file
                })