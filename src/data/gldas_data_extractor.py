import ee
import pandas as pd

class GLDASDriveExporter:
    """Export GLDAS time series to Google Drive for a list of wells."""

    def __init__(self, config):
        self.config = config

        # Initialize Earth Engine
        try:
            project_id = config['gldas'].get('project_id', None)
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            project_id = config['gldas'].get('project_id', None)
            ee.Initialize(project=project_id)

        self.collection_id = config['gldas']['collection']
        self.variable = config['gldas']['variable']
        self.scale = 25000  # Default GLDAS scale in meters
        self.start_date = config['time']['start_date']
        self.end_date = config['time']['end_date']

    def export_site_to_drive(self, site_no, lat, lon):
        point = ee.Geometry.Point([lon, lat])

        collection = ee.ImageCollection(self.collection_id) \
            .filterDate(self.start_date, self.end_date) \
            .select(self.variable) \
            .map(lambda image: image.set('system:index', image.date().format('YYYY-MM-dd')))

        def extract_feature(img):
            value = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=self.scale
            ).set('system:index', img.get('system:index'))
            return ee.Feature(None, value)

        features = collection.map(extract_feature)

        task = ee.batch.Export.table.toDrive(
            collection=ee.FeatureCollection(features),
            description=f'gldas_{site_no}',
            folder='gldas_exports',
            fileNamePrefix=f'{site_no}',
            fileFormat='CSV',
            selectors=['system:index', self.variable]
        )

        task.start()
        print(f"Started export for site {site_no}")

    def export_to_drive(self, wells_df):
        for _, row in wells_df.iterrows():
            site_no = str(row['site_no'])
            lat = row['latitude']
            lon = row['longitude']

            if pd.isna(lat) or pd.isna(lon):
                print(f"Skipping site {site_no}: Missing coordinates")
                continue

            try:
                self.export_site_to_drive(site_no, lat, lon)
            except Exception as e:
                print(f"Error exporting site {site_no}: {e}")
