import os
import pandas as pd
import numpy as np

from flask import Flask, render_template, jsonify, request

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# =========================
# Coordinate Conversion
# =========================

def dmm_to_decimal(value):

    value = float(value)

    degrees = int(value // 100)

    minutes = value % 100

    return degrees + (minutes / 60)


def convert_coordinate(value):

    if pd.isna(value):
        return np.nan

    value = float(value)

    if abs(value) > 180:
        return dmm_to_decimal(value)

    return value


# =========================
# Read Excel Files
# =========================

def read_excel_file(path):

    try:
        df = pd.read_excel(path, engine='openpyxl')

    except:

        try:
            df = pd.read_excel(path, engine='xlrd')

        except:
            return None

    df.columns = df.columns.str.strip().str.lower()

    required_cols = ['meterno', 'gps_lat', 'gps_long']

    if not all(col in df.columns for col in required_cols):
        return None

    return df[required_cols]


# =========================
# Load Data
# =========================

print("Loading and processing data...")

final_df = pd.DataFrame()



# =========================
# Home Page
# =========================

@app.route('/')
def index():

    return render_template('index.html')

# =========================
# Upload Files API
# =========================

@app.route('/upload', methods=['POST'])
def upload_files():

    uploaded_files = request.files.getlist('files')

    all_data = []

    for file in uploaded_files:
        if not (
           file.filename.endswith('.xls')
        or
           file.filename.endswith('.xlsx')
        ):
           continue

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            file.filename
        )

        file.save(filepath)

        df = read_excel_file(filepath)

        if df is None:
            continue

        df['source_file'] = file.filename

        all_data.append(df)

    if len(all_data) == 0:

        return jsonify({
            "error": "No valid files uploaded"
        }), 400

    global final_df

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    final_df['gps_lat'] = pd.to_numeric(
        final_df['gps_lat'],
        errors='coerce'
    )

    final_df['gps_long'] = pd.to_numeric(
        final_df['gps_long'],
        errors='coerce'
    )

    final_df['latitude'] = (
        final_df['gps_lat']
        .apply(convert_coordinate)
    )

    final_df['longitude'] = (
        final_df['gps_long']
        .apply(convert_coordinate)
    )

    final_df = final_df.dropna(
        subset=['latitude', 'longitude']
    )

    return jsonify({

        "message":
            "Files uploaded successfully",

        "records":
            int(len(final_df))
    })

# =========================
# Search API
# =========================

@app.route('/api/search')
def search():
    if final_df.empty:
       return jsonify([])

    query = request.args.get('q', '').lower()

    meters = final_df[
        final_df['meterno'].astype(str)
        .str.lower()
        .str.contains(query)
    ]

    unique_meters = meters['meterno'].drop_duplicates()
    
    return jsonify(unique_meters.tolist())


# =========================
# Meter Clustering API
# =========================

@app.route('/api/meter/<meterno>')
def get_meter(meterno):
    if final_df.empty:
    
        return jsonify({
            "error": "No files uploaded"
        }), 400
    meter_data = final_df[
        final_df['meterno'].astype(str) == str(meterno)
    ].copy()

    if len(meter_data) == 0:

        return jsonify({
            "error": "Meter not found"
        }), 404


    # =====================
    # RAW POINTS
    # =====================

    coords = meter_data[
        ['latitude', 'longitude']
    ].dropna().copy()


    # =====================
    # DBSCAN
    # =====================

    dbscan_result = None

    if len(coords) >= 3:

        coords_rad = np.radians(coords)

        db = DBSCAN(
            eps=10/6371000,
            min_samples=3,
            metric='haversine'
        )

        labels = db.fit_predict(coords_rad)

        coords['dbscan_cluster'] = labels

        valid = coords[
            coords['dbscan_cluster'] != -1
        ]

        if len(valid) > 0:

            main_cluster = (
                valid['dbscan_cluster']
                .value_counts()
                .idxmax()
            )

            cluster_points = valid[
                valid['dbscan_cluster'] == main_cluster
            ]

            dbscan_result = {
                "latitude":
                    float(cluster_points['latitude'].median()),

                "longitude":
                    float(cluster_points['longitude'].median()),

                "points_used":
                    int(len(cluster_points))
            }


    # =====================
    # AGGLOMERATIVE
    # =====================

    agg_result = None

    if len(coords) >= 2:

        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.0001
        )

        agg_labels = agg.fit_predict(
        coords[['latitude', 'longitude']]
)

        coords['agg_cluster'] = agg_labels
        
        main_cluster = (
            coords['agg_cluster']
            .value_counts()
            .idxmax()
        )
        
        cluster_points = coords[
            coords['agg_cluster'] == main_cluster
        ]
        
        agg_result = {
        
            "latitude":
                float(cluster_points['latitude'].median()),
        
            "longitude":
                float(cluster_points['longitude'].median()),
        
            "points_used":
                int(len(cluster_points))

        }

    # =====================
    # RETURN RESPONSE
    # =====================
    
    return jsonify({

        "meterno": meterno,

        "raw_points":
            coords[
                ['latitude', 'longitude']
            ].to_dict('records'),

        "dbscan": dbscan_result,

        "agglomerative": agg_result
    })


# =========================
# Stats API
# =========================

@app.route('/api/stats')
def get_stats():
    if final_df.empty:
    
        return jsonify({
            "error": "No files uploaded"
        }), 400
    file_total = (
        final_df.groupby('source_file')['meterno']
        .count()
    )

    file_unique = (
        final_df.groupby('source_file')['meterno']
        .nunique()
    )
    meter_counts = final_df['meterno'].value_counts()
    return jsonify({

        'total_records': int(len(final_df)),

    'total_unique_meters':
        int(final_df['meterno'].nunique()),

    'files': {

        'labels':
            file_total.index.tolist(),

        'total':
            file_total.tolist(),

        'unique':
            file_unique.tolist()
    },

    'meter_counts': {

        'labels': list(range(1, 11)),

        'values': [
            len(meter_counts[meter_counts == i])
            for i in range(1, 11)
        ]
    }
    })

# =========================
# Run App
# =========================

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )