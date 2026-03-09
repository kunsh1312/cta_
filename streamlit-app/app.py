import streamlit as st
import pandas as pd
import geopandas as gpd
from pathlib import Path
import altair as alt
import requests, io

#Paths
user_path = Path(__file__).parent.resolve()
raw_data_path = user_path / "data" / "raw-data" 
derived_data_path = user_path / "data" / "derived-data"

@st.cache_resource
def load_derived_crime():
    url = "https://www.dropbox.com/scl/fi/90xiyc5az9kyuroa8xmg8/derived_crime.zip?rlkey=u3d9t60gno2z1m1kiq5cjg63f&st=fpp2qi55&dl=1"
    response = requests.get(url)
    response.raise_for_status()
    return gpd.read_file(io.BytesIO(response.content))

derived_crime = load_derived_crime()

#File Identifiers
RIDERSHIP_CSV  = "CTA_Ridership_L_Station_Entries_Daily_Totals_2022-2026.csv"
DERIVED_SHP    = "derived_crime.shp"

#CRS
SOURCE_CRS = "EPSG:3435"   # Illinois State Plane East (feet)
TARGET_CRS = "EPSG:4326"   # WGS-84

# Shapefile column names 
COL_STATION      = "stationnam"  
COL_LONGNAME     = "LONGNAME_x"  
COL_LINES        = "LINES_x"     
COL_YEAR         = "Year_x"      
COL_MONTH        = "Month"
COL_RIDES        = "rides"
COL_CRIME_ID     = "ID"
COL_PRIMARY_TYPE = "Primary Ty"  

LINE_SUFFIXES = [
    " Line", " (O'Hare)", " (Congress)", " (Lake)", " (Englewood)", " (Express)"
]


def load_violent_classify(df):
    df = df[df[COL_CRIME_ID].notna()]
    violent_types = [
        'HOMICIDE', 
        'CRIMINAL SEXUAL ASSAULT', 
        'SEX OFFENSE', 
        'ROBBERY', 
        'ASSAULT', 
        'BATTERY', 
        'KIDNAPPING'
    ]
    df['Crime_Category'] = df['Primary Ty'].apply(
        lambda x: 'Violent' if str(x).upper() in violent_types else 'Non-Violent'
    )
    return df

def year_filter(df,start_yr = 2022,end_yr = 2025):
    df = df[df["Year_x"] >= start_yr]
    df = df[df["Year_x"] <= end_yr]
    return df

def crime_filter(df,crime_type = "All"):
    if crime_type == "All":
        return df
    elif crime_type == "Violent":
        df = df[df['Crime_Category']=="Violent"]
        return df
    elif crime_type == "Non-Violent":
        df = df[df['Crime_Category']=="Non-Violent"]
        return df
    else:
        df = df[df[COL_PRIMARY_TYPE] == crime_type.upper()]
        return df

@st.cache_data
def aggregator(crime_type = "All",start_yr = 2022,end_yr = 2025):
    crime_df = load_violent_classify(derived_crime)
    crime_df = crime_filter(crime_df,crime_type)
    crime_df = year_filter(crime_df,start_yr,end_yr)
    
    ridership_df = year_filter(derived_crime,start_yr,end_yr)

    rides_monthly = (
    ridership_df.drop_duplicates(subset=[COL_STATION, "date"])
    [[COL_STATION, COL_LONGNAME, COL_YEAR, COL_MONTH, COL_RIDES]]
    .groupby([COL_STATION, COL_LONGNAME, COL_YEAR, COL_MONTH])[COL_RIDES]
    .sum()
    .reset_index()
    )
    
    crime_monthly = (
    crime_df
    .groupby([COL_STATION, COL_YEAR, COL_MONTH])
    .size()
    .reset_index(name="crime_count")
    )

    monthly = rides_monthly.merge(crime_monthly, on=[COL_STATION, COL_YEAR, COL_MONTH], how="left")
    monthly["crime_count"] = monthly["crime_count"].fillna(0)
    monthly.rename(columns={
        COL_LONGNAME: "stationname_mapped",
        COL_YEAR:     "year",
        COL_MONTH:    "month",
    }, inplace=True)

    monthly = monthly.groupby("stationname_mapped").agg(crime_count=("crime_count", "sum"), rides=("rides", "sum")).reset_index()

    return monthly

final_data = aggregator(crime_type="Violent",start_yr=2025)

def make_chart(final_data):
    Chart = alt.Chart(final_data).mark_point(filled=True).transform_calculate(
        crime_per_100000 = "datum.crime_count/(datum.rides/100000)"
    ).encode(
        alt.X("crime_per_100000:Q", title="Crime incidents per 100,000 riders"),
        alt.Y("rides:Q", title="Total Rides"),
        tooltip=["stationname_mapped:N", 
                alt.Tooltip("crime_per_100000:Q", format=".0f"),
                "rides:Q"]
        )
    return Chart

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="CTA Crime & Ridership Dashboard", layout="wide")

st.title("🚇 CTA Station Analysis: Crime vs. Ridership")

# Create the Tabs first
tab_analysis, tab_map = st.tabs(["📊 Statistical Analysis", "🗺️ Geographic Map"])

# 1. Global Sidebar Elements (Common to both tabs)
st.sidebar.header("Global Filters")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=2022,
    max_value=2025,
    value=(2022, 2025)
)
start_yr, end_yr = year_range

# 2. Tab-Specific Sidebar & Content
with tab_analysis:
    # Sidebar elements for Analysis
    st.sidebar.divider()
    st.sidebar.subheader("Scatterplot Filters")
    valid_crimes = sorted(derived_crime[COL_PRIMARY_TYPE].dropna().unique().tolist())
    crime_options = ["All", "Violent", "Non-Violent"] + [c.title() for c in valid_crimes]
    selected_crime = st.sidebar.selectbox(
        "Select Crime Category/Type", 
        options=crime_options, 
        key="analysis_crime_sel"
    )

    # Data Processing
    with st.spinner("Aggregating analysis data..."):
        filtered_data = aggregator(
            crime_type=selected_crime, 
            start_yr=start_yr, 
            end_yr=end_yr
        )

    # UI Layout for Analysis
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Analysis: {selected_crime} Crimes ({start_yr}-{end_yr})")
        chart = make_chart(filtered_data)
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.subheader("Summary Metrics")
        total_rides = filtered_data['rides'].sum()
        total_crimes = filtered_data['crime_count'].sum()
        st.metric("Total Ridership", f"{total_rides:,}")
        st.metric("Total Incidents", f"{int(total_crimes):,}")
        avg_rate = (total_crimes / (total_rides / 100000)) if total_rides > 0 else 0
        st.metric("Avg Crime Rate", f"{avg_rate:.2f}", help="Incidents per 100k riders")

with tab_map:
    # Sidebar elements for Map
    st.sidebar.divider()
    st.sidebar.subheader("Map Controls")
    
    map_metric = st.sidebar.radio(
        "Circle Size Reflects:",
        options=["Total Crimes", "Crimes per 100k Riders", "Total Riders"],
        key="map_metric_choice"
    )
    
    # Use aggregator to get the ridership and crime totals per station
    map_filtered_data = aggregator(crime_type="All", start_yr=start_yr, end_yr=end_yr)

    # Prepare spatial data
    import pydeck as pdk
    # We drop the 'rides' column from map_df if it exists to prevent merge conflicts (rides_x/rides_y)
    map_df = derived_crime.to_crs("EPSG:4326").copy()
    if 'rides' in map_df.columns:
        map_df = map_df.drop(columns=['rides'])
        
    map_df['lat'] = map_df.geometry.y
    map_df['lon'] = map_df.geometry.x
    
    # Merge spatial points with the aggregated data
    map_plot_df = map_df.drop_duplicates(subset=[COL_STATION]).merge(
        map_filtered_data, 
        left_on=COL_STATION, 
        right_on="stationname_mapped"
    )

    # Now 'rides' is guaranteed to be the column from map_filtered_data
    map_plot_df["crime_rate"] = (map_plot_df["crime_count"] / (map_plot_df["rides"] / 100000)).fillna(0)
    map_plot_df['crime_rate_str'] = map_plot_df['crime_rate'].map('{:.2f}'.format)
    map_plot_df['rides_str'] = map_plot_df['rides'].map('{:,}'.format)

    # Determine Radius based on Selection
    if map_metric == "Total Crimes":
        # Using a multiplier of 2 for visible pixel scaling
        map_plot_df["radius_val"] = 5 + (map_plot_df["crime_count"]/400)
    elif map_metric == "Crimes per 100k Riders":
        map_plot_df["radius_val"] = 5 + (map_plot_df["crime_rate"]/10)
    else: # Total Riders
        # Ridership numbers are large, so we scale them down significantly for pixel radius
        map_plot_df["radius_val"] = 5 + (map_plot_df["rides"] / 500000)

    # Map Layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        map_plot_df,
        get_position=["lon", "lat"],
        get_color="[200, 30, 0, 160]",
        radius_units="pixels",
        get_radius="radius_val",
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=11)

    st.subheader(f"CTA Station Hotspots (By {map_metric})")

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Station:</b> {stationname_mapped}<br/>"
                    "<b>Crimes:</b> {crime_count}<br/>"
                    "<b>Rate:</b> {crime_rate_str}<br/>"
                    "<b>Total Rides:</b> {rides_str}",
            "style": {"color": "white"}
        }
    ))    


    
