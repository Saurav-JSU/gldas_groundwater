import os
import yaml
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Import modules
from src.data.usgs_data import USGSDataRetriever
from src.data.gee_api import GLDASDataExtractor
from src.processing.parallel_processor import ParallelProcessor
from src.analysis.statistics import StatisticalAnalyzer
from src.analysis.visualization import VisualizationGenerator
from src.analysis.temporal import TemporalAnalyzer
from src.analysis.spatial import SpatialAnalyzer

def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def find_latest_results_dir():
    """Find the most recently created results directory."""
    results_dirs = glob.glob(os.path.join('results', 'run_*'))
    
    if not results_dirs:
        return None
    
    # Sort by creation time (newest first)
    results_dirs.sort(key=os.path.getctime, reverse=True)
    
    return results_dirs[0]

def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GLDAS Groundwater Evaluation Framework')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', choices=['all', 'download', 'process', 'analyze',
                                          'visualize', 'spatial', 'temporal'],
                        default='all', help='Execution mode')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of cores to use (default: auto-detect)')
    parser.add_argument('--results-dir', default=None,
                        help='Specific results directory to use (overrides automatic selection)')
    parser.add_argument('--fixed-dir', action='store_true',
                        help='Use a fixed results directory name without timestamp')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Determine results directory
    if args.fixed_dir:
        # Use a fixed directory name
        results_dir = os.path.join('results', 'current_run')
    elif args.results_dir:
        # Use the provided directory
        results_dir = args.results_dir
    elif args.mode == 'all' or args.mode == 'download':
        # Create a new timestamped directory for full runs or downloads
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join('results', f'run_{timestamp}')
    else:
        # For other modes, try to find the latest results directory
        latest_dir = find_latest_results_dir()
        
        if latest_dir:
            results_dir = latest_dir
        else:
            # If no existing directory found, create a new one
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = os.path.join('results', f'run_{timestamp}')
    
    # Define all directories explicitly
    dirs = {
        'wells': os.path.join('data', 'raw', 'wells'),
        'gldas': os.path.join('data', 'raw', 'gldas'),
        'processed_wells': os.path.join('data', 'processed', 'wells'),
        'processed_gldas': os.path.join('data', 'processed', 'gldas'),
        'merged': os.path.join('data', 'merged'),
        'results': results_dir,
        'metrics': os.path.join(results_dir, 'metrics'),
        'plots': os.path.join(results_dir, 'plots'),
        'spatial': os.path.join(results_dir, 'spatial'),
        'temporal': os.path.join(results_dir, 'temporal')
    }
    
    # Create all directories
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Determine number of cores to use
    n_cores = args.cores
    
    print(f"==== GLDAS Evaluation Framework ====")
    print(f"Mode: {args.mode}")
    print(f"Results will be saved to: {dirs['results']}")
    
    # Initialize components
    parallel_processor = ParallelProcessor(config, n_workers=n_cores)
    
    if args.mode in ['all', 'download']:
        print("\n==== Data Acquisition ====")
        
        # Initialize data collectors
        well_collector = USGSDataRetriever(config)
        gldas_extractor = GLDASDataExtractor(config)
        
        # Download well data
        wells_metadata_file = os.path.join(dirs['wells'], 'all_site_metrics.csv')
        
        if os.path.exists(wells_metadata_file):
            print("Loading existing well metadata...")
            wells_metadata = pd.read_csv(wells_metadata_file)
        else:
            print("Downloading well data...")
            wells_metadata = well_collector.download_all_sites_data(dirs['wells'])
        
        if wells_metadata is not None and not wells_metadata.empty:
            print("Extracting GLDAS data...")
            gldas_metadata = gldas_extractor.extract_for_wells(wells_metadata, dirs['gldas'])
        else:
            print("Error: No valid well metadata. Skipping GLDAS extraction.")
    
    if args.mode in ['all', 'process']:
        print("\n==== Data Processing ====")
        
        # Process data in parallel
        print("Processing well data...")
        processed_wells = parallel_processor.process_all_wells(dirs['wells'], dirs['processed_wells'])
        
        print("Processing GLDAS data...")
        processed_gldas = parallel_processor.process_all_gldas(dirs['gldas'], dirs['processed_gldas'])
        
        print("Merging datasets...")
        merged_files = parallel_processor.merge_datasets(processed_wells, processed_gldas, dirs['merged'])
    else:
        # Load processed files from directories
        processed_wells = {}
        processed_gldas = {}
        merged_files = {}
        
        # Get well files
        for file in os.listdir(dirs['processed_wells']):
            if file.endswith('_monthly.csv'):
                site_no = file.split('_')[0]
                processed_wells[site_no] = os.path.join(dirs['processed_wells'], file)
        
        # Get GLDAS files
        for file in os.listdir(dirs['processed_gldas']):
            if file.endswith('_gldas_monthly.csv'):
                site_no = file.split('_')[0]
                processed_gldas[site_no] = os.path.join(dirs['processed_gldas'], file)
        
        # Get merged files
        for file in os.listdir(dirs['merged']):
            if file.endswith('_merged.csv'):
                site_no = file.split('_')[0]
                merged_files[site_no] = os.path.join(dirs['merged'], file)
    
    if args.mode in ['all', 'analyze']:
        print("\n==== Statistical Analysis ====")
        
        # Load site metadata
        metadata_file = os.path.join(dirs['wells'], 'all_site_metrics.csv')
        if os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file)
        else:
            metadata_df = None
        
        # Initialize analyzer
        analyzer = StatisticalAnalyzer(n_workers=n_cores)
        
        # Analyze all sites
        print(f"Analyzing {len(merged_files)} sites...")
        metrics_df = analyzer.analyze_all_sites(
            merged_files, 
            metadata_df=metadata_df,
            output_dir=dirs['metrics']
        )
        
        # Generate summary statistics
        if not metrics_df.empty:
            print("\n== Summary Statistics ==")
            print(f"Total sites analyzed: {len(metrics_df)}")
            print(f"Mean correlation: {metrics_df['correlation'].mean():.3f}")
            print(f"Median correlation: {metrics_df['correlation'].median():.3f}")
            print(f"Mean RMSE: {metrics_df['rmse'].mean():.3f} m")
            print(f"Mean NSE: {metrics_df['nse'].mean():.3f}")
            print(f"Percentage of sites with correlation > 0.5: {(metrics_df['correlation'] > 0.5).mean() * 100:.1f}%")
            print(f"Percentage of sites with NSE > 0: {(metrics_df['nse'] > 0).mean() * 100:.1f}%")
            
            # Save summary
            summary = {
                'total_sites': len(metrics_df),
                'mean_correlation': metrics_df['correlation'].mean(),
                'median_correlation': metrics_df['correlation'].median(),
                'mean_rmse': metrics_df['rmse'].mean(),
                'mean_nse': metrics_df['nse'].mean(),
                'corr_gt_0_5_pct': (metrics_df['correlation'] > 0.5).mean() * 100,
                'nse_gt_0_pct': (metrics_df['nse'] > 0).mean() * 100
            }
            
            import json
            with open(os.path.join(dirs['results'], 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
        else:
            print("No metrics calculated.")
    
    if args.mode in ['all', 'visualize']:
        print("\n==== Visualization ====")
        
        # Load metrics
        metrics_file = os.path.join(dirs['metrics'], 'all_metrics.csv')
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
        else:
            print("No metrics file found. Cannot generate visualizations.")
            return
        
        # Load site metadata
        metadata_file = os.path.join(dirs['wells'], 'all_site_metrics.csv')
        if os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file)
        else:
            metadata_df = None
        
        # Initialize visualization generator
        viz_generator = VisualizationGenerator(
            output_dir=dirs['plots']
        )
        
        # Generate summary plots
        print("Generating summary visualizations...")
        summary_plots = viz_generator.create_summary_metrics_plot(
            metrics_df,
            output_dir=dirs['plots']
        )
        
        # Generate site-specific visualizations for top-performing sites
        print("Generating site-specific visualizations for selected sites...")
        # Sort by correlation
        sorted_sites = metrics_df.sort_values('correlation', ascending=False)
        
        # Get top 20 sites by correlation
        top_sites = sorted_sites.head(20)
        
        # Plot top sites
        for idx, row in top_sites.iterrows():
            site_no = row['site_no']
            
            # Get merged data
            merged_file = os.path.join(dirs['merged'], f"{site_no}_merged.csv")
            if os.path.exists(merged_file):
                try:
                    # Read merged data
                    merged_df = pd.read_csv(merged_file)
                    merged_df['date'] = pd.to_datetime(merged_df['date'])
                    
                    # Get site metadata
                    site_metadata = None
                    if metadata_df is not None:
                        site_rows = metadata_df[metadata_df['site_no'] == site_no]
                        if not site_rows.empty:
                            site_metadata = site_rows.iloc[0].to_dict()
                    
                    # Create visualizations
                    viz_generator.create_time_series_plot(
                        merged_df, site_no, site_metadata, row.to_dict()
                    )
                    
                    viz_generator.create_scatter_plot(
                        merged_df, site_no, site_metadata, row.to_dict()
                    )
                    
                    viz_generator.create_seasonal_plot(
                        merged_df, site_no, site_metadata
                    )
                    
                except Exception as e:
                    print(f"Error generating visualizations for site {site_no}: {e}")
    
    if args.mode in ['all', 'spatial']:
        print("\n==== Spatial Analysis ====")
        
        # Load metrics
        metrics_file = os.path.join(dirs['metrics'], 'all_metrics.csv')
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
        else:
            print("No metrics file found. Cannot perform spatial analysis.")
            return
        
        # Initialize spatial analyzer
        spatial_analyzer = SpatialAnalyzer()
        
        # Perform spatial clustering
        print("Performing spatial clustering...")
        try:
            clustered_df, cluster_stats = spatial_analyzer.spatial_clustering(
                metrics_df, 
                eps=50,  # 50 km cluster radius
                min_samples=3  # Minimum 3 sites per cluster
            )
            
            # Save results
            clustered_df.to_csv(
                os.path.join(dirs['spatial'], 'clustered_sites.csv'),
                index=False
            )
            
            cluster_stats.to_csv(
                os.path.join(dirs['spatial'], 'cluster_statistics.csv'),
                index=False
            )
            
            # Analyze spatial correlation patterns
            print("Analyzing spatial correlation patterns...")
            spatial_corr = spatial_analyzer.analyze_spatial_correlation(
                metrics_df,
                metric_col='correlation',
                distance_bins=10
            )
            
            # Save spatial correlation results
            spatial_corr['pairs_data'].to_csv(
                os.path.join(dirs['spatial'], 'spatial_correlation_pairs.csv'),
                index=False
            )
            
            spatial_corr['bin_statistics'].to_csv(
                os.path.join(dirs['spatial'], 'spatial_correlation_bins.csv'),
                index=False
            )
            
            # Print summary
            print(f"Spatial correlation with distance: {spatial_corr['distance_metric_correlation']:.3f}")
            
        except Exception as e:
            print(f"Error in spatial analysis: {e}")
    
    if args.mode in ['all', 'temporal']:
        print("\n==== Temporal Analysis ====")
        
        # Load metrics
        metrics_file = os.path.join(dirs['metrics'], 'all_metrics.csv')
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
        else:
            print("No metrics file found. Cannot perform temporal analysis.")
            return
        
        # Initialize temporal analyzer
        temporal_analyzer = TemporalAnalyzer()
        
        # Get sites with good correlation
        good_sites = metrics_df[metrics_df['correlation'] > 0.7]['site_no'].tolist()
        if not good_sites:
            good_sites = metrics_df.sort_values('correlation', ascending=False).head(10)['site_no'].tolist()
        
        # Process each good site
        temporal_results = []
        for site_no in good_sites:
            # Get merged data
            merged_file = os.path.join(dirs['merged'], f"{site_no}_merged.csv")
            if os.path.exists(merged_file):
                try:
                    # Read merged data
                    merged_df = pd.read_csv(merged_file)
                    merged_df['date'] = pd.to_datetime(merged_df['date'])
                    
                    # Perform trend analysis
                    trends = temporal_analyzer.calculate_trends(
                        merged_df,
                        columns=['gw_anomaly_m', 'gldas_gws_anomaly']
                    )
                    
                    # Perform lag analysis
                    lag_results = temporal_analyzer.analyze_lag_correlation(
                        merged_df,
                        col1='gw_anomaly_m',
                        col2='gldas_gws_anomaly'
                    )
                    
                    # Perform seasonal analysis
                    seasonal_well = temporal_analyzer.analyze_seasonal_patterns(
                        merged_df,
                        column='gw_anomaly_m'
                    )
                    
                    seasonal_gldas = temporal_analyzer.analyze_seasonal_patterns(
                        merged_df,
                        column='gldas_gws_anomaly'
                    )
                    
                    # Combine results
                    result = {
                        'site_no': site_no,
                        'well_trend_slope': trends['gw_anomaly_m']['slope'],
                        'well_trend_p_value': trends['gw_anomaly_m']['p_value'],
                        'gldas_trend_slope': trends['gldas_gws_anomaly']['slope'],
                        'gldas_trend_p_value': trends['gldas_gws_anomaly']['p_value'],
                        'optimal_lag': lag_results['optimal_lag'],
                        'max_correlation': lag_results['max_correlation'],
                        'well_peak_month': seasonal_well['peak_month'],
                        'well_trough_month': seasonal_well['trough_month'],
                        'gldas_peak_month': seasonal_gldas['peak_month'],
                        'gldas_trough_month': seasonal_gldas['trough_month'],
                        'well_annual_amplitude': seasonal_well['annual_amplitude'],
                        'gldas_annual_amplitude': seasonal_gldas['annual_amplitude']
                    }
                    
                    temporal_results.append(result)
                    
                    # Save monthly statistics
                    seasonal_well['monthly_stats'].to_csv(
                        os.path.join(dirs['temporal'], f"{site_no}_well_monthly.csv"),
                        index=False
                    )
                    
                    seasonal_gldas['monthly_stats'].to_csv(
                        os.path.join(dirs['temporal'], f"{site_no}_gldas_monthly.csv"),
                        index=False
                    )
                    
                except Exception as e:
                    print(f"Error analyzing temporal patterns for site {site_no}: {e}")
        
        # Save temporal results
        if temporal_results:
            pd.DataFrame(temporal_results).to_csv(
                os.path.join(dirs['temporal'], 'temporal_analysis.csv'),
                index=False
            )
            
            # Print summary
            mean_lag = np.nanmean([r['optimal_lag'] for r in temporal_results])
            print(f"Mean optimal lag: {mean_lag:.1f} months")
    
    # Create a marker file to indicate this is the last run directory
    with open(os.path.join(results_dir, 'last_run.marker'), 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    print(f"\nAnalysis complete. Results saved to: {dirs['results']}")

if __name__ == "__main__":
    main()