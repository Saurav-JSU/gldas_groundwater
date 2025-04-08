import os
import yaml
import argparse
from datetime import datetime
import pandas as pd

# Import our modules - UPDATED IMPORT
from src.data.usgs_data import USGSDataRetriever  # Use the new class
from src.data.gee_api import GLDASDataExtractor
from src.data.usgs_data import USGSDataRetriever
from src.data.gldas_data_extractor import GLDASDriveExporter  # <- NEW FILE
from src.data.gee_api import GLDASDataExtractor               # <- Already present
from src.processing.well_processor import WellDataProcessor
from src.processing.gldas_processor import GLDASProcessor
from src.analysis.comparison import DataComparison

def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GLDAS Groundwater Evaluation')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', choices=['all', 'download', 'process', 'analyze', 'export-gldas'],
                        default='all', help='Execution mode')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create output directories
    data_dir = os.path.join('data')
    os.makedirs(data_dir, exist_ok=True)

    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    results_dir = os.path.join('results')

    # Timestamped directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Paths for specific data types
    raw_wells_dir = os.path.join(raw_dir, 'wells')
    raw_gldas_dir = os.path.join(raw_dir, 'gldas')
    proc_wells_dir = os.path.join(processed_dir, 'wells')
    proc_gldas_dir = os.path.join(processed_dir, 'gldas')

    # Create all directories
    for directory in [raw_wells_dir, raw_gldas_dir, proc_wells_dir, proc_gldas_dir]:
        os.makedirs(directory, exist_ok=True)

    # Initialize components
    well_collector = USGSDataRetriever(config)
    gldas_extractor = GLDASDataExtractor(config)
    gldas_exporter = GLDASDriveExporter(config)  # NEW CLASS
    well_processor = WellDataProcessor(config)
    gldas_processor = GLDASProcessor(config)
    data_comparison = DataComparison(config)

    if args.mode in ['all', 'download']:
        wells_metadata_file = os.path.join(raw_wells_dir, 'all_site_metrics.csv')

        if os.path.exists(wells_metadata_file):
            print("=== Loading existing well metadata ===")
            wells_metadata = pd.read_csv(wells_metadata_file)
        else:
            print("=== Downloading well data ===")
            wells_metadata = well_collector.download_all_sites_data(raw_wells_dir)

        if wells_metadata is not None and not wells_metadata.empty:
            print("=== Extracting GLDAS data ===")
            gldas_metadata = gldas_extractor.extract_for_wells(wells_metadata, raw_gldas_dir)
        else:
            print("Error: wells_metadata is None or empty. Skipping GLDAS extraction.")

    if args.mode == 'export-gldas':
        wells_metadata_file = os.path.join(raw_wells_dir, 'all_site_metrics.csv')
        if os.path.exists(wells_metadata_file):
            wells_metadata = pd.read_csv(wells_metadata_file)
            print("=== Exporting GLDAS data to Google Drive ===")
            gldas_exporter.export_to_drive(wells_metadata)
        else:
            print("Error: all_site_metrics.csv not found. Cannot export GLDAS.")

    if args.mode in ['all', 'process']:
        print("=== Processing well data ===")
        processed_wells = well_processor.process_all_wells(raw_wells_dir, proc_wells_dir)

        print("=== Processing GLDAS data ===")
        processed_gldas = gldas_processor.process_all_gldas(raw_gldas_dir, proc_gldas_dir)
    else:
        # Load existing processed files
        processed_wells = {}
        processed_gldas = {}

        for file in os.listdir(proc_wells_dir):
            if file.endswith('_monthly.csv'):
                site_no = file.split('_')[0]
                processed_wells[site_no] = os.path.join(proc_wells_dir, file)

        for file in os.listdir(proc_gldas_dir):
            if file.endswith('_gldas_monthly.csv'):
                site_no = file.split('_')[0]
                processed_gldas[site_no] = os.path.join(proc_gldas_dir, file)

    if args.mode in ['all', 'analyze']:
        print("=== Analyzing data ===")
        metrics = data_comparison.analyze_all_sites(processed_wells, processed_gldas, run_dir)

        if metrics is not None and not metrics.empty:
            print("\n=== Summary Statistics ===")
            print(f"Total sites analyzed: {len(metrics)}")
            print(f"Mean correlation: {metrics['correlation'].mean():.3f}")
            print(f"Mean RMSE: {metrics['rmse'].mean():.3f} m")
            print(f"Mean NSE: {metrics['nse'].mean():.3f}")
            print(f"Percentage of sites with correlation > 0.5: {(metrics['correlation'] > 0.5).mean() * 100:.1f}%")
            print(f"Percentage of sites with NSE > 0: {(metrics['nse'] > 0).mean() * 100:.1f}%")

            summary = {
                'total_sites': len(metrics),
                'mean_correlation': metrics['correlation'].mean(),
                'mean_rmse': metrics['rmse'].mean(),
                'mean_nse': metrics['nse'].mean(),
                'corr_gt_0.5_pct': (metrics['correlation'] > 0.5).mean() * 100,
                'nse_gt_0_pct': (metrics['nse'] > 0).mean() * 100
            }

            import json
            with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)

            print(f"\nResults saved to: {run_dir}")

if __name__ == "__main__":
    main()
