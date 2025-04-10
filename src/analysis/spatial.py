import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN

class SpatialAnalyzer:
    """Class for spatial analysis of well and GLDAS data."""
    
    def __init__(self):
        """Initialize the spatial analyzer."""
        pass
    
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the haversine distance between two points in kilometers."""
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Calculate haversine distance
        coords_1 = np.array([[lat1_rad, lon1_rad]])
        coords_2 = np.array([[lat2_rad, lon2_rad]])
        
        # Haversine distance in radians
        dist_rad = haversine_distances(coords_1, coords_2)
        
        # Convert to kilometers (Earth radius ~6371 km)
        dist_km = 6371 * dist_rad[0, 0]
        
        return dist_km
    
    def spatial_clustering(self, sites_df, eps=50, min_samples=3, lat_col='latitude', lon_col='longitude'):
        """
        Perform spatial clustering using DBSCAN.
        
        Parameters:
            sites_df (pd.DataFrame): DataFrame with site information
            eps (float): Maximum distance between points in a cluster (km)
            min_samples (int): Minimum number of samples in a neighborhood
            lat_col (str): Column name for latitude
            lon_col (str): Column name for longitude
            
        Returns:
            pd.DataFrame: DataFrame with cluster assignments
        """
        # Create a copy to avoid modifying the original
        df = sites_df.copy()
        
        # Convert lat/lon to radians
        lat_rad = np.radians(df[lat_col].values)
        lon_rad = np.radians(df[lon_col].values)
        
        # Combine coordinates
        coords = np.vstack((lat_rad, lon_rad)).T
        
        # Calculate eps in radians (eps km / Earth radius)
        eps_rad = eps / 6371.0
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
        clusters = clustering.fit_predict(coords)
        
        # Add cluster labels to DataFrame
        df['cluster'] = clusters
        
        # Get cluster statistics
        cluster_stats = df.groupby('cluster').agg({
            'site_no': 'count',
            lat_col: ['mean', 'min', 'max'],
            lon_col: ['mean', 'min', 'max']
        })
        
        # Flatten column names
        cluster_stats.columns = [f"{col[0]}_{col[1]}" for col in cluster_stats.columns]
        cluster_stats = cluster_stats.reset_index()
        
        return df, cluster_stats
    
    def analyze_spatial_correlation(self, metrics_df, lat_col='latitude', lon_col='longitude', 
                                   metric_col='correlation', distance_bins=10):
        """
        Analyze how correlation varies with spatial distance.
        
        Parameters:
            metrics_df (pd.DataFrame): DataFrame with metrics and coordinates
            lat_col (str): Column name for latitude
            lon_col (str): Column name for longitude
            metric_col (str): Column name for metric to analyze
            distance_bins (int): Number of distance bins
            
        Returns:
            dict: Dictionary with spatial correlation analysis results
        """
        # Create distance matrix
        n_sites = len(metrics_df)
        distances = np.zeros((n_sites, n_sites))
        site_ids = metrics_df['site_no'].values
        
        # Calculate pairwise distances
        for i in range(n_sites):
            for j in range(i, n_sites):
                if i == j:
                    continue
                    
                dist = self.calculate_haversine_distance(
                    metrics_df.iloc[i][lat_col], metrics_df.iloc[i][lon_col],
                    metrics_df.iloc[j][lat_col], metrics_df.iloc[j][lon_col]
                )
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Flatten distance matrix for analysis
        flat_data = []
        for i in range(n_sites):
            for j in range(i+1, n_sites):  # Only upper triangle to avoid duplicates
                metric_i = metrics_df.iloc[i][metric_col]
                metric_j = metrics_df.iloc[j][metric_col]
                
                if not np.isnan(metric_i) and not np.isnan(metric_j):
                    flat_data.append({
                        'site_i': site_ids[i],
                        'site_j': site_ids[j],
                        'distance': distances[i, j],
                        f'{metric_col}_i': metric_i,
                        f'{metric_col}_j': metric_j,
                        'metric_diff': abs(metric_i - metric_j)
                    })
        
        # Convert to DataFrame
        pairs_df = pd.DataFrame(flat_data)
        
        # Calculate correlation between distance and metric difference
        distance_metric_corr = pairs_df['distance'].corr(pairs_df['metric_diff'])
        
        # Create distance bins and analyze metric differences in each bin
        pairs_df['distance_bin'] = pd.cut(pairs_df['distance'], bins=distance_bins)
        bin_stats = pairs_df.groupby('distance_bin').agg({
            'metric_diff': ['count', 'mean', 'std', 'min', 'max']
        })
        
        # Flatten column names
        bin_stats.columns = [f"{col[0]}_{col[1]}" for col in bin_stats.columns]
        bin_stats = bin_stats.reset_index()
        
        # Add bin centers
        bin_stats['bin_center'] = bin_stats['distance_bin'].apply(lambda x: (x.left + x.right) / 2)
        
        return {
            'distance_metric_correlation': distance_metric_corr,
            'bin_statistics': bin_stats,
            'pairs_data': pairs_df
        }