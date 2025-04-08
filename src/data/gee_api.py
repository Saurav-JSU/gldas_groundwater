import ee
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class GLDASDataExtractor:
    """Class to extract GLDAS data from Google Earth Engine."""
    
    def __init__(self, config):
        """
        Initialize the GLDAS data extractor.
        
        Parameters:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Initialize Earth Engine with project_id
        try:
            project_id = config['gldas'].get('project_id', None)
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            project_id = config['gldas'].get('project_id', None)
            ee.Initialize(project=project_id)
    
    def extract_time_series(self, lat, lon, scale=None):
        """
        Extract GLDAS time series for a specific location.
        
        Parameters:
            lat (float): Latitude
            lon (float): Longitude
            scale (float, optional): Scale in meters for extraction
            
        Returns:
            pandas.DataFrame: Time series data
        """
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])
        
        # Get GLDAS collection
        collection = ee.ImageCollection(self.config['gldas']['collection']) \
            .filterDate(self.config['time']['start_date'], self.config['time']['end_date']) \
            .select(self.config['gldas']['variable'])
        
        # Get time series at point
        def extract_value(image):
            date = ee.Date(image.get('system:time_start'))
            value = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=scale or 25000  # GLDAS resolution is ~25km
            ).get(self.config['gldas']['variable'])
            
            return ee.Feature(None, {
                'date': date.format('YYYY-MM-dd'),
                'value': value
            })
        
        features = collection.map(extract_value).getInfo()['features']
        
        # Convert to DataFrame
        data = []
        for feature in features:
            props = feature['properties']
            if props['value'] is not None:
                data.append({
                    'date': datetime.strptime(props['date'], '%Y-%m-%d'),
                    'gldas_gws': float(props['value']) * self.config['gldas']['scale_factor']
                })
        
        df = pd.DataFrame(data)
        
        # Calculate anomalies
        if not df.empty:
            mean_gws = df['gldas_gws'].mean()
            df['gldas_gws_anomaly'] = df['gldas_gws'] - mean_gws
        
        return df
    
    def extract_for_wells(self, wells_metadata, output_dir):
        """
        Extract GLDAS data for all wells in the metadata.
        
        Parameters:
            wells_metadata (pandas.DataFrame): Well metadata
            output_dir (str): Directory to save extracted data
            
        Returns:
            pandas.DataFrame: Updated metadata with GLDAS extraction info
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Add columns for GLDAS data
        wells_metadata['gldas_file_path'] = None
        wells_metadata['gldas_record_count'] = 0
        
        for i, (idx, well) in enumerate(wells_metadata.iterrows()):
            print(f"Extracting GLDAS data for well {i+1}/{len(wells_metadata)}: {well['site_no']}")
            
            try:
                # Extract data
                gldas_df = self.extract_time_series(well['latitude'], well['longitude'])
                
                if not gldas_df.empty:
                    # Save to CSV
                    output_file = os.path.join(output_dir, f"{well['site_no']}_gldas.csv")
                    gldas_df.to_csv(output_file, index=False)
                    
                    # Update metadata
                    wells_metadata.at[idx, 'gldas_file_path'] = output_file
                    wells_metadata.at[idx, 'gldas_record_count'] = len(gldas_df)
                
                # Avoid hitting rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error extracting GLDAS data for well {well['site_no']}: {e}")
        
        # Save updated metadata
        metadata_file = os.path.join(output_dir, "metadata_with_gldas.csv")
        wells_metadata.to_csv(metadata_file, index=False)
        
        return wells_metadata
    
    def extract_spatial_data(self, date, region_geometry=None, scale=None):
        """
        Extract spatial GLDAS data for a specific date.
        
        Parameters:
            date (str): Date in 'YYYY-MM-DD' format
            region_geometry (ee.Geometry, optional): Region to extract
            scale (float, optional): Scale in meters
            
        Returns:
            ee.Image: GLDAS image for the specified date
        """
        if region_geometry is None:
            # Create region geometry from config bounds
            bounds = self.config['region']['bounds']
            region_geometry = ee.Geometry.Rectangle([
                bounds['lon_min'], bounds['lat_min'],
                bounds['lon_max'], bounds['lat_max']
            ])
        
        # Get GLDAS image for the date
        start_date = ee.Date(date)
        end_date = start_date.advance(1, 'day')
        
        image = ee.ImageCollection(self.config['gldas']['collection']) \
            .filterDate(start_date, end_date) \
            .select(self.config['gldas']['variable']) \
            .first()
        
        if image is None:
            raise ValueError(f"No GLDAS data found for {date}")
        
        # Apply scale factor
        image = image.multiply(self.config['gldas']['scale_factor'])
        
        return image
    
    def export_to_geotiff(self, date, output_file, region_geometry=None, scale=None):
        """
        Export GLDAS data to GeoTIFF for a specific date.
        
        Parameters:
            date (str): Date in 'YYYY-MM-DD' format
            output_file (str): Output file path
            region_geometry (ee.Geometry, optional): Region to extract
            scale (float, optional): Scale in meters
            
        Returns:
            str: Path to downloaded file
        """
        image = self.extract_spatial_data(date, region_geometry, scale)
        
        # Export to Drive
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f"GLDAS_GWS_{date.replace('-', '')}",
            folder='GLDAS_Export',
            fileNamePrefix=f"GLDAS_GWS_{date.replace('-', '')}",
            region=region_geometry,
            scale=scale or 25000,
            crs='EPSG:4326',
            maxPixels=1e9
        )
        
        task.start()
        print(f"Started export task for {date}. Check your Google Drive.")
        
        return None  # Since the file is exported to Drive