# Squirrel-Red
# Author: James Fothergill (v8255920)

# Imports
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import libpysal as lps
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import shape
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from folium.plugins import HeatMap
from tqdm import tqdm 
from random import choice
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Squirrel-Red Scottish Red Squirrel Sightings",
    page_icon="🐿️",
    layout="wide"
)

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Distribution Map", "Model Metrics", "Temporal Reliability", "About"]
)

def create_spatial_features(boundaries):
    boundaries = boundaries.copy()
    boundaries['area'] = boundaries.geometry.area

    W = lps.weights.Queen.from_dataframe(boundaries)
    W.transform = 'r'
    return boundaries, W

def create_temporal_features(squirrel_docs, boundaries):
    counts_df = pd.DataFrame(index=boundaries.index)
    boundaries = boundaries.to_crs('EPSG:27700')

    for year in range(2011, 2020):
        year_features = [f for f in squirrel_docs if f['properties']['year'] == year]
        if not year_features:
            counts_df[year] = 0
            continue

        year_points = gpd.GeoDataFrame.from_features(year_features)
        year_points.set_crs(epsg=4326, inplace=True)
        year_points = year_points.to_crs('EPSG:27700')

        joined = gpd.sjoin(year_points, boundaries, how='left', predicate='within')
        counts = joined.groupby('index_right').size()
        counts = counts.reindex(boundaries.index, fill_value=0)
        counts_df[year] = counts

    return counts_df

def create_feature_matrix(boundaries, counts_df, year, W):
    X = pd.DataFrame(index=boundaries.index)
    X['area'] = boundaries['area'] / 1e6
    X['density'] = counts_df[year]
    X['prev_year_count'] = counts_df[year-1] if year > 2011 else counts_df[year]
    X['spatial_lag'] = sparse.csr_matrix(W.sparse).dot(counts_df[year])
    return X

load_dotenv()

@st.cache_resource(show_spinner="Collecting acorns 🌰")
def load_data():
    connection_string = os.getenv("MONGODB_URI")
    if not connection_string:
        raise ValueError("MongoDB URI not found. Check your .env file.")

    client = MongoClient(connection_string)

    # Load boundaries
    boundary_collection = client['Scottish_Boundaries']['Boundary_data']
    boundaries = gpd.GeoDataFrame.from_features(list(boundary_collection.find()))
    boundaries.set_crs(epsg=4326, inplace=True)

    # Load squirrel data
    squirrel_collection = client['Squirrel_Spatial']["squirrel_data_2010_2022_filtered"]
    squirrel_docs = list(squirrel_collection.find())

    return boundaries, squirrel_docs

# Load data and prepare model
boundaries, squirrel_docs = load_data()

# Prepare data and train model
boundaries, W = create_spatial_features(boundaries)
counts_df = create_temporal_features(squirrel_docs, boundaries)

# Create feature matrices
data_frames = []
for year in range(2013, 2019):
    X = create_feature_matrix(boundaries, counts_df, year, W)
    y = counts_df[year+1]
    
    year_df = X.copy()
    year_df['year'] = year
    year_df['target'] = y
    year_df['region'] = boundaries['local_authority']
    data_frames.append(year_df)

data_df = pd.concat(data_frames, axis=0)

# Train Random Forest Regressor model
feature_cols = ['area', 'density', 'prev_year_count', 'spatial_lag']
X = data_df[feature_cols]
y = data_df['target']

# Split the data temporally
train_years = range(2013, 2017)
test_years = range(2017, 2019)

train_mask = data_df['year'].isin(train_years)
test_mask = data_df['year'].isin(test_years)

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with optimal settings
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

# Calculate model performance
train_pred = rf_model.predict(X_train_scaled)
test_pred = rf_model.predict(X_test_scaled)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

# Title
st.markdown("# 🐿️ Squirrel-<span style='color:#FF0000'>Red</span> Scottish Red Squirrel Analysis Dashboard", unsafe_allow_html=True)

if page == "Distribution Map":
    # Subheader
    st.header("Squirrel Distribution Analysis")
    
    # Map type selection
    map_type = st.radio(
        "Select Map Type",
        ["Current Distribution", "Future Predictions", "Historical Comparison"]
    )
    
    if map_type == "Current Distribution":
        # Create base map
        m = folium.Map(
            location=[57, -4],
            zoom_start=6,
            tiles='cartodbpositron'
        )
        
        # Create GeoDataFrame for 2019 squirrels
        squirrel_points = []
        point_coords = []
        for doc in squirrel_docs:
            if ('geometry' in doc and 
                'coordinates' in doc['geometry'] and 
                'properties' in doc and 
                'year' in doc['properties'] and 
                doc['properties']['year'] == 2019):
                
                coords = doc['geometry']['coordinates']
                point_coords.append([coords[0], coords[1]])
                squirrel_points.append([coords[1], coords[0]])
        
        # Create the GeoDataFrame for the spatial join
        points_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([p[0] for p in point_coords], [p[1] for p in point_coords]),
            crs="EPSG:4326"
        )
        
        # Spatially join to count points in each area
        joined = gpd.sjoin(points_gdf, boundaries, how='right', predicate='within')
        area_counts = joined.groupby('local_authority').size().fillna(0)
        
        # Add the boundary overlay with hover information
        folium.GeoJson(
            boundaries.__geo_interface__,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'black',
                'weight': 1
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['local_authority'],
                aliases=['Area:'],
                labels=True,
                sticky=True
            )
        ).add_to(m)
        
        # Add heatmap
        from folium.plugins import HeatMap
        HeatMap(
            squirrel_points,
            min_opacity=0.5,
            radius=15,
            blur=10,
            max_zoom=1
        ).add_to(m)
        
        # Display map
        folium_static(m)
        
        # Statistics section
        st.subheader("Current Distribution Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Sightings", len(squirrel_points))
        
        with col2:
            st.metric("Areas with Sightings", len(area_counts[area_counts > 0]))
        
        # Show area breakdown
        st.subheader("Sightings by Area")
        area_df = pd.DataFrame(area_counts).reset_index()
        area_df.columns = ['Area', 'Sightings']
        area_df = area_df.sort_values('Sightings', ascending=False)
        st.dataframe(area_df, hide_index=True)

    elif map_type == "Future Predictions":
        # Year selection for predictions
        year = st.slider(
            "Select Prediction Year",
            min_value=2020,
            max_value=2025,
            value=2020,
            step=1
        )
        
        # Create the base map
        m = folium.Map(
            location=[57, -4],
            zoom_start=6,
            tiles='cartodbpositron'
        )
        
        # Make the predictions
        future_predictions = {}
        current_counts = counts_df.copy()
        
        # Generate the predictions for selected year
        for pred_year in range(2020, year + 1):
            X_future = create_feature_matrix(boundaries, current_counts, pred_year-1, W)
            X_future_scaled = scaler.transform(X_future)
            
            # Get the predictions and uncertainty estimates from all trees
            predictions = []
            for estimator in rf_model.estimators_:
                predictions.append(estimator.predict(X_future_scaled))
            predictions = np.array(predictions)
            
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            future_predictions[pred_year] = {
                'mean': mean_pred,
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred
            }
            
            # Update the counts for next year's prediction
            current_counts[pred_year] = mean_pred
        
        # Create the choropleth map
        def get_color(count):
            if count > 100:
                return '#8B0000'  # Dark red
            elif count > 50:
                return '#FF0000'  # Red
            elif count > 20:
                return '#FF4444'  # Light red
            elif count > 0:
                return '#FFB6C1'  # Pink
            return '#808080'      # Gray
        
        # Add the choropleth layer
        folium.GeoJson(
            boundaries.__geo_interface__,
            style_function=lambda x: {
                'fillColor': get_color(future_predictions[year]['mean'][
                    boundaries[boundaries['local_authority'] == x['properties']['local_authority']].index[0]
                ]),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['local_authority'],
                aliases=['Area:'],
                labels=True,
                sticky=True
            )
        ).add_to(m)
        
        # Add the legend
        legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 150px; height: 130px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color:white;
                        padding: 10px;
                        border-radius: 5px;
                        ">
                <p style="margin-bottom: 10px;"><strong>Predicted Sightings</strong></p>
                <p><span style="color:#8B0000;">■</span> >100</p>
                <p><span style="color:#FF0000;">■</span> 51-100</p>
                <p><span style="color:#FF4444;">■</span> 21-50</p>
                <p><span style="color:#FFB6C1;">■</span> 1-20</p>
                <p><span style="color:#808080;">■</span> 0</p>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Display map
        folium_static(m)
        
        # Statistics section
        st.subheader(f"Predicted Distribution for {year}")
        
        col1, col2, col3 = st.columns(3)
        
        total_current = counts_df[2019].sum()
        total_predicted = future_predictions[year]['mean'].sum()
        
        with col1:
            st.metric(
                "Current Total (2019)",
                f"{int(total_current):,}"
            )
        
        with col2:
            st.metric(
                "Predicted Total",
                f"{int(total_predicted):,}",
                f"{int(total_predicted - total_current):+,}"
            )
        
        with col3:
            change_percent = ((total_predicted - total_current) / total_current * 100 
                            if total_current > 0 else 0)
            st.metric(
                "Predicted Change",
                f"{change_percent:+.1f}%"
            )
        
        # Show detailed predictions table
        st.subheader("Regional Predictions")
        predictions_df = pd.DataFrame({
            'Region': boundaries['local_authority'],
            'Current (2019)': counts_df[2019].round().astype(int),
            f'Predicted ({year})': future_predictions[year]['mean'].round().astype(int),
            'Lower CI': future_predictions[year]['lower'].round().astype(int),
            'Upper CI': future_predictions[year]['upper'].round().astype(int)
        })
        
        predictions_df['Change'] = predictions_df[f'Predicted ({year})'] - predictions_df['Current (2019)']
        predictions_df['% Change'] = (predictions_df['Change'] / predictions_df['Current (2019)'] * 100).round(1)
        
        # Sort by predicted values
        predictions_df = predictions_df.sort_values(f'Predicted ({year})', ascending=False)
        
        # Format and display the dataframe
        st.dataframe(
            predictions_df.style.format({
                'Current (2019)': '{:,}',
                f'Predicted ({year})': '{:,}',
                'Lower CI': '{:,}',
                'Upper CI': '{:,}',
                'Change': '{:+,}',
                '% Change': '{:+.1f}%'
            }),
            hide_index=True
        )
        
        # Add prediction disclaimer
        st.info("""
        **Note:** These predictions are based on:
        - Historical patterns (2011-2019)
        - Spatial relationships between regions
        - Regional characteristics
        
        The confidence intervals (CI) represent the model's uncertainty. Actual populations 
        may vary due to external factors such as climate change, habitat modification, 
        or conservation efforts.
        """)

    elif map_type == "Historical Comparison":
        # Year selection
        years = st.select_slider(
            "Select Years to Compare",
            options=list(range(2011, 2020)),
            value=(2015, 2019)
        )
        
        # Create base map
        m = folium.Map(
            location=[57, -4],
            zoom_start=6,
            tiles='cartodbpositron'
        )
        
        # Create GeoDataFrames for both years
        def get_year_points(year):
            points = []
            coords = []
            processed = 0
            total = len(squirrel_docs)
            
            progress_text = st.empty()
            
            for doc in squirrel_docs:
                processed += 1
                progress_text.text(f"Processing {year} data: {processed}/{total}")
                
                if ('geometry' in doc and 
                    'coordinates' in doc['geometry'] and 
                    'properties' in doc and 
                    'year' in doc['properties'] and 
                    doc['properties']['year'] == year):
                    
                    c = doc['geometry']['coordinates']
                    points.append([c[1], c[0]])  # For heatmap
                    coords.append([c[0], c[1]])  # For GeoDataFrame
            
            gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy([p[0] for p in coords], [p[1] for p in coords]),
                crs="EPSG:4326"
            )
            progress_text.empty()
            return points, gdf
        
        # Get data for both years
        points_old, gdf_old = get_year_points(years[0])
        points_new, gdf_new = get_year_points(years[1])
        
        # Calculate counts for both years
        joined_old = gpd.sjoin(gdf_old, boundaries, how='right', predicate='within')
        joined_new = gpd.sjoin(gdf_new, boundaries, how='right', predicate='within')
        
        counts_old = joined_old.groupby('local_authority').size().fillna(0)
        counts_new = joined_new.groupby('local_authority').size().fillna(0)
        
        # Calculate changes
        changes = counts_new - counts_old
        
        # Colour the boundaries based on change
        def get_color(change):
            if change > 0:
                return '#ff4444'  # Red for increase
            elif change < 0:
                return '#4444ff'  # Blue for decrease
            return '#808080'      # Gray for no change
        
        # Add choropleth layer
        folium.GeoJson(
            boundaries.__geo_interface__,
            style_function=lambda x: {
                'fillColor': get_color(changes.get(x['properties']['local_authority'], 0)),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.5
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['local_authority'],
                aliases=['Area:'],
                labels=True,
                sticky=True
            )
        ).add_to(m)
        
        # Display map
        folium_static(m)
        
        # Statistics section
        st.subheader(f"Comparison: {years[0]} vs {years[1]}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Sightings in {years[0]}", 
                     len(points_old))
        
        with col2:
            st.metric(f"Sightings in {years[1]}", 
                     len(points_new),
                     len(points_new) - len(points_old))
        
        with col3:
            st.metric("Areas with Changes",
                     len(changes[changes != 0]))
        
        # Show detailed changes
        st.subheader("Changes by Area")
        changes_df = pd.DataFrame({
            'Area': changes.index,
            f'Sightings {years[0]}': counts_old.round().astype(int),
            f'Sightings {years[1]}': counts_new.round().astype(int),
            'Change': changes.round().astype(int)
        })
        changes_df = changes_df.sort_values('Change', ascending=False)
        
        # Colour code the changes
        def color_changes(val):
            if val > 0:
                return 'color: #ff4444'
            elif val < 0:
                return 'color: #4444ff'
            return ''
        
        st.dataframe(
            changes_df.style.applymap(color_changes, subset=['Change']),
            hide_index=True
        )

elif page == "Model Metrics":
    st.header("Model Performance Metrics")
    
    # Check if data is loaded
    if not squirrel_docs:
        st.error("No data loaded. Please check the database connection.")
        st.stop()
    
    metric_type = st.radio(
        "Select Metric Type",
        ["Prediction Accuracy", "Regional Performance"]  # Removed "Temporal Reliability"
    )
    
    if metric_type == "Prediction Accuracy":
        with st.spinner("Processing prediction accuracy..."):
            st.subheader("Prediction vs Actual Sightings")
            
            # Process historical data for Random Forest
            X = []  # Features
            y = []  # Target
            years_data = {}
            
            # Prepare data for Random Forest (only up to 2019)
            for doc in squirrel_docs:
                if ('properties' in doc and 
                    'year' in doc['properties'] and 
                    'month' in doc['properties'] and
                    isinstance(doc['properties']['year'], int) and
                    doc['properties']['year'] <= 2019):
                    
                    year = doc['properties']['year']
                    if year not in years_data:
                        years_data[year] = {
                            'count': 0,
                            'months': [0] * 12
                        }
                    
                    years_data[year]['count'] += 1
                    month = doc['properties']['month']
                    if isinstance(month, int) and 1 <= month <= 12:
                        years_data[year]['months'][month - 1] += 1
            
            if years_data:
                # Create features and target
                years = sorted(years_data.keys())
                for i in range(len(years) - 1):
                    year = years[i]
                    next_year = years[i + 1]
                    
                    features = [years_data[year]['count']] + years_data[year]['months']
                    X.append(features)
                    y.append(years_data[next_year]['count'])
                
                if X and y:
                    # Convert to numpy arrays and train model
                    X = np.array(X)
                    y = np.array(y)
                    
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X, y)
                    
                    # Make predictions
                    actual_data = {}
                    predicted_data = {}
                    
                    for i in range(len(years) - 1):
                        year = years[i]
                        next_year = years[i + 1]
                        
                        features = [years_data[year]['count']] + years_data[year]['months']
                        
                        actual_data[next_year] = years_data[next_year]['count']
                        predicted_data[next_year] = int(rf_model.predict([features])[0])
                    
                    # Create comparison plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(actual_data.keys()),
                        y=list(actual_data.values()),
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='blue', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(predicted_data.keys()),
                        y=list(predicted_data.values()),
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='red', width=3, dash='dash'),
                        marker=dict(size=10)
                    ))
                    
                    fig.update_layout(
                        title="Random Forest: Predicted vs Actual Sightings",
                        xaxis_title="Year",
                        yaxis_title="Number of Sightings",
                        hovermode='x unified',
                        height=500,
                        template="plotly_dark",
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate error metrics
                    mse = mean_squared_error(
                        list(actual_data.values()),
                        list(predicted_data.values())
                    )
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(
                        list(actual_data.values()),
                        list(predicted_data.values())
                    )
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Squared Error", f"{mse:,.0f}")
                    with col2:
                        st.metric("Root Mean Squared Error", f"{rmse:,.0f}")
                    with col3:
                        st.metric("Mean Absolute Error", f"{mae:,.0f}")
                else:
                    st.error("Insufficient data for training.")
            else:
                st.error("No valid data found. Please check the data structure.")

    elif metric_type == "Regional Performance":
        with st.spinner("Processing regional performance..."):
            st.subheader("Regional Performance Analysis")
            
            # Process data by region
            region_data = {}
            
            # Initialize regions from boundaries
            for idx, row in boundaries.iterrows():
                region_name = row['local_authority']
                region_data[region_name] = {
                    'years': {},
                    'total_sightings': 0,
                    'prediction_accuracy': 0,
                    'geometry': row['geometry']
                }
            
            # Match squirrel points to regions
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_points = len(squirrel_docs)
            
            for idx, doc in enumerate(squirrel_docs):
                # Update progress bar
                if idx % 100 == 0:
                    progress = idx / total_points
                    progress_bar.progress(progress)
                    status_text.text(f"Processing points... {idx:,} of {total_points:,} ({progress:.1%})")
                
                if ('properties' in doc and 
                    'year' in doc['properties'] and 
                    'geometry' in doc and
                    isinstance(doc['properties']['year'], int) and
                    doc['properties']['year'] <= 2019):
                    
                    point = shape(doc['geometry'])
                    year = doc['properties']['year']
                    
                    # Find which region contains this point
                    for region_name, region_info in region_data.items():
                        if region_info['geometry'].contains(point):
                            if year not in region_info['years']:
                                region_info['years'][year] = 0
                            region_info['years'][year] += 1
                            region_info['total_sightings'] += 1
                            break
            
            # Update progress bar to completion
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Clear progress indicators
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Remove geometry after matching
            for region in region_data:
                del region_data[region]['geometry']
            
            # Calculate prediction accuracy for regions with sufficient data
            status_text = st.empty()
            status_text.text("Calculating regional predictions...")
            
            for region in region_data:
                if region_data[region]['total_sightings'] >= 50: # Only analyse regions with enough data
                    years = sorted(region_data[region]['years'].keys())
                    if len(years) > 1:
                        X = []
                        y = []
                        
                        # Create training data
                        for i in range(len(years) - 1):
                            X.append([region_data[region]['years'][years[i]]])
                            y.append(region_data[region]['years'][years[i + 1]])
                        
                        if X and y:
                            # Train model for this region
                            rf = RandomForestRegressor(n_estimators=50, random_state=42)
                            rf.fit(np.array(X), np.array(y))
                            
                            # Calculate accuracy
                            predictions = rf.predict(np.array(X))
                            accuracy = 100 * (1 - np.mean(np.abs(predictions - y) / y))
                            region_data[region]['prediction_accuracy'] = accuracy
            
            status_text.empty()
            
            # Create performance metrics table
            st.subheader("Regional Statistics")
            
            metrics_data = []
            for region, data in region_data.items():
                if data['total_sightings'] >= 50:  # Only include regions with sufficient data
                    metrics_data.append({
                        'Region': region,
                        'Sightings': f"{data['total_sightings']:,}",
                        'Years of Data': len(data['years']),
                        'Prediction Accuracy': f"{data['prediction_accuracy']:.1f}%"
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data).sort_values('Sightings', ascending=False)
                
                # Style the dataframe
                def color_accuracy(val):
                    try:
                        accuracy = float(val.strip('%'))
                        if accuracy >= 80:
                            return 'background-color: #c6efce; color: #006100'  # Light green
                        elif accuracy >= 60:
                            return 'background-color: #ffeb9c; color: #9c6500'  # Light yellow
                        else:
                            return 'background-color: #ffc7ce; color: #9c0006'  # Light red
                    except:
                        return ''
                
                st.dataframe(
                    metrics_df.style.applymap(
                        color_accuracy,
                        subset=['Prediction Accuracy']
                    ),
                    hide_index=True,
                    use_container_width=True
                )
                
                # Create map visualization
                m = folium.Map(
                    location=[57, -4],
                    zoom_start=6,
                    tiles='cartodbpositron'
                )
                
                # Colour regions by prediction accuracy
                folium.GeoJson(
                    boundaries,
                    style_function=lambda x: {
                        'fillColor': f'#{int(255 * min(max(region_data.get(x["properties"]["local_authority"], {}).get("prediction_accuracy", 0), 0)/100, 1)):02x}0000',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['local_authority'],
                        aliases=['Region:'],
                        labels=True,
                        sticky=True
                    )
                ).add_to(m)
                
                # Display map
                st.write("Regional Prediction Accuracy")
                folium_static(m)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_accuracy = np.mean([
                        data['prediction_accuracy']
                        for data in region_data.values()
                        if data['total_sightings'] >= 50
                    ])
                    st.metric(
                        "Average Regional Accuracy",
                        f"{avg_accuracy:.1f}%"
                    )
                
                with col2:
                    best_region = max(
                        [(r, d) for r, d in region_data.items() if d['total_sightings'] >= 50],
                        key=lambda x: x[1]['prediction_accuracy']
                    )
                    st.metric(
                        "Best Performing Region",
                        f"{best_region[0]}\n({best_region[1]['prediction_accuracy']:.1f}%)"
                    )
                
                with col3:
                    total_sightings = sum(
                        data['total_sightings']
                        for data in region_data.values()
                        if data['total_sightings'] >= 50
                    )
                    st.metric(
                        "Total Sightings Analyzed",
                        f"{total_sightings:,}"
                    )

elif page == "Temporal Reliability":
    st.header("Temporal Reliability Analysis")
    
    # Process temporal data
    monthly_data = {}
    yearly_data = {}
    
    # Collect temporal patterns
    for doc in squirrel_docs:
        if ('properties' in doc and 
            'year' in doc['properties'] and 
            'month' in doc['properties'] and
            isinstance(doc['properties']['year'], int) and
            isinstance(doc['properties']['month'], int) and
            doc['properties']['year'] <= 2019):
            
            year = doc['properties']['year']
            month = doc['properties']['month']
            
            # Yearly aggregation
            if year not in yearly_data:
                yearly_data[year] = 0
            yearly_data[year] += 1
            
            # Monthly aggregation
            if month not in monthly_data:
                monthly_data[month] = 0
            monthly_data[month] += 1
    
    # Create monthly pattern visualization
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_counts = [monthly_data.get(m, 0) for m in months]
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=month_names,
        y=monthly_counts,
        marker_color='darkred'
    ))
    
    fig_monthly.update_layout(
        title="Monthly Distribution of Squirrel Sightings",
        xaxis_title="Month",
        yaxis_title="Number of Sightings",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Calculate monthly reliability metrics
    monthly_stats = pd.DataFrame({
        'Month': month_names,
        'Sightings': monthly_counts,
        'Percentage': [count/sum(monthly_counts)*100 for count in monthly_counts]
    })
    
    # Style the monthly stats table
    def color_percentage(val):
        if val >= 15:
            return 'background-color: #c6efce; color: #006100'  # High activity
        elif val >= 5:
            return 'background-color: #ffeb9c; color: #9c6500'  # Medium activity
        else:
            return 'background-color: #ffc7ce; color: #9c0006'  # Low activity
    
    st.dataframe(
        monthly_stats.style.format({
            'Sightings': '{:,}',
            'Percentage': '{:.1f}%'
        }).applymap(
            color_percentage,
            subset=['Percentage']
        ),
        hide_index=True,
        use_container_width=True
    )
    
    # Create yearly trend visualisation
    years = sorted(yearly_data.keys())
    yearly_counts = [yearly_data[year] for year in years]
    
    fig_yearly = go.Figure()
    fig_yearly.add_trace(go.Scatter(
        x=years,
        y=yearly_counts,
        mode='lines+markers',
        line=dict(color='darkred', width=3),
        marker=dict(size=10)
    ))
    
    fig_yearly.update_layout(
        title="Yearly Trend of Squirrel Sightings",
        xaxis_title="Year",
        yaxis_title="Number of Sightings",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig_yearly, use_container_width=True)
    
    # Calculate year-over-year changes
    yoy_changes = []
    for i in range(1, len(years)):
        prev_year = yearly_counts[i-1]
        curr_year = yearly_counts[i]
        change = ((curr_year - prev_year) / prev_year * 100)
        yoy_changes.append({
            'Year': years[i],
            'Sightings': curr_year,
            'Change': change
        })
    
    # Display year-over-year changes
    if yoy_changes:
        yoy_df = pd.DataFrame(yoy_changes)
        
        def color_change(val):
            if val >= 10:
                return 'background-color: #c6efce; color: #006100'  # Significant increase
            elif val <= -10:
                return 'background-color: #ffc7ce; color: #9c0006'  # Significant decrease
            else:
                return 'background-color: #ffeb9c; color: #9c6500'  # Stable
        
        st.subheader("Year-over-Year Changes")
        st.dataframe(
            yoy_df.style.format({
                'Sightings': '{:,}',
                'Change': '{:+.1f}%'
            }).applymap(
                color_change,
                subset=['Change']
            ),
            hide_index=True,
            use_container_width=True
        )
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            peak_month = month_names[monthly_counts.index(max(monthly_counts))]
            st.metric(
                "Peak Activity Month",
                peak_month
            )
        
        with col2:
            avg_monthly = sum(monthly_counts) / len([c for c in monthly_counts if c > 0])
            st.metric(
                "Average Monthly Sightings",
                f"{avg_monthly:,.0f}"
            )
        
        with col3:
            avg_change = np.mean([c['Change'] for c in yoy_changes])
            st.metric(
                "Average Annual Change",
                f"{avg_change:+.1f}%"
            )

elif page == "About":
    st.header("About This Project")
    
    st.subheader("Data Source")
    st.markdown("""
    This project uses red squirrel sighting data from the [National Biodiversity Network (NBN) Atlas](https://nbnatlas.org/). 
    The dataset includes verified red squirrel sightings across Scotland from 2010 to 2019.
    
    Data collection and verification is made possible through the efforts of:
    - Scottish Wildlife Trust
    - Saving Scotland's Red Squirrels project
    - Local community volunteers
    - Citizen scientists
    """)
    
    st.subheader("Data Processing")
    st.markdown("""
    The raw data underwent several processing steps to ensure quality and reliability:
    
    1. **Duplicate Removal**: We identified and removed duplicate sightings using:
        - Spatial proximity (within 50 metres)
        - Temporal proximity (same day)
        - Coordinate uncertainty analysis
        - Original dataset: 75,068 records
        - After duplicate removal: 70,838 records
        - Duplicates removed: 4,230 (5.63%)
    
    2. **Spatial Analysis**:
        - Sightings were mapped to Scottish local authorities
        - Used geometric boundaries for regional analysis
        - Applied spatial containment algorithms
    
    3. **Temporal Processing**:
        - Data from 2020 onwards was excluded due to COVID-19 impact on:
            - Volunteer mobility restrictions
            - Reduced monitoring capacity
            - Data collection inconsistencies
            - Potential reporting biases
    """)
    
    st.subheader("Analysis Methods")
    st.markdown("""
    The project employs several analytical approaches:
    
    1. **Predictive Modelling**:
        - Random Forest Regression for sighting predictions
        - Year-over-year trend analysis
        - Regional performance metrics
    
    2. **Temporal Analysis**:
        - Monthly distribution patterns
        - Seasonal variation study
        - Year-over-year change calculation
    
    3. **Regional Analysis**:
        - Local authority-based aggregation
        - Geographic distribution mapping
        - Regional prediction accuracy assessment
    """)
    
    st.subheader("Technologies Used")
    st.markdown("""
    This application is built using:
    - **Streamlit**: Web application framework
    - **MongoDB**: Data storage and retrieval
    - **Python**: Primary programming language
    - **Libraries**:
        - Pandas & GeoPandas: Data manipulation
        - Folium: Interactive mapping
        - Scikit-learn: Machine learning
        - Plotly: Data visualisation
    """)
    
    st.subheader("Acknowledgements")
    st.markdown("""
    Special thanks to:
    - The NBN Atlas for providing the data
    - Saving Scotland's Red Squirrels project team
    - All volunteers and citizen scientists who contributed to data collection
    - The Scottish Wildlife Trust for their conservation efforts
    
    This project is for educational purposes and aims to support red squirrel conservation efforts in Scotland.
    """)

