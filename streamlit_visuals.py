import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import os
import plotly.express as px
from PIL import Image
import io

# --- Config ---
st.set_page_config(layout="wide", page_title="Pavement Marking Condition Map", page_icon="🛣️")

# Custom CSS to improve visuals
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: rgba(71, 71, 71, 0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .condition-good {
        background-color: #9AE6B4;
        padding: 0.5rem;
        border-radius: 5px;
        color: #276749;
    }
    .condition-damaged {
        background-color: #FBD38D;
        padding: 0.5rem;
        border-radius: 5px;
        color: #975A16;
    }
    .condition-missing {
        background-color: #FEB2B2;
        padding: 0.5rem;
        border-radius: 5px;
        color: #9B2C2C;
    }
    .condition-unknown {
        background-color: #E2E8F0;
        padding: 0.5rem;
        border-radius: 5px;
        color: #4A5568;
    }
    .stSidebar {
        background-color: rgba(71, 71, 71, 0.1);
        border-right: 1px solid #e2e8f0;
    }
    .map-container {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(37, 37, 37, 0.1);
        margin-top: 1rem;
    }
    /* Fix map dimming issues */
    .folium-map {
        width: 100% !important;
        height: 100% !important;
        opacity: 1 !important;
        z-index: 1 !important;
    }
    
    /* Ensure map tiles render at full opacity */
    .leaflet-tile-pane {
        opacity: 1 !important;
    }
    
    /* Fix any overlay issues */
    .leaflet-control-container {
        z-index: 999 !important;
        opacity: 1 !important;
    }

</style>
""", unsafe_allow_html=True)

IMAGE_FOLDER = "mapillary_images"

# --- Load CSV ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('crops_metadata_with_coords.csv')
        # Convert condition strings to lowercase for consistency
        if 'predicted_condition' in df.columns:
            df['predicted_condition'] = df['predicted_condition'].str.lower()
        df = df.dropna(subset=['latitude', 'longitude'])
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# --- Get marker color and icon ---
def get_marker_style(condition):
    condition = str(condition).lower()
    if condition == 'good':
        return 'green', 'check-circle'
    elif condition == 'damaged':
        return 'orange', 'warning'
    elif condition == 'missing':
        return 'red', 'times-circle'
    else:
        return 'blue', 'question-circle'

# --- Create custom icon ---
def create_custom_icon(condition):
    color, icon_name = get_marker_style(condition)
    return folium.Icon(
        color=color,
        icon=icon_name,
        prefix='fa',
        shadow=True
    )

# --- Create map with markers ---

def create_map(data, use_clusters=True):
    # Calculate map center (weighted towards more important markers)
    weights = {'missing': 3, 'damaged': 2, 'good': 1}
    
    # Default center if calculations fail
    center_lat, center_lng = data['latitude'].mean(), data['longitude'].mean()
    
    # Initial map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, 
                  tiles="CartoDB positron")  # Use a cleaner map style
    
    # Add different base map layers
    folium.TileLayer(
        'CartoDB positron', 
        name='Light Map',
        attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'
    ).add_to(m)

    folium.TileLayer(
        'CartoDB dark_matter', 
        name='Dark Map',
        attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'
    ).add_to(m)

    folium.TileLayer(
        'OpenStreetMap', 
        name='Street Map',
        attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    ).add_to(m)

    folium.TileLayer(
        'Stamen Terrain', 
        name='Terrain Map',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)
    
    # Create marker cluster if enabled
    if use_clusters:
        marker_cluster = MarkerCluster(name="All Markers").add_to(m)
        
    # Create condition-based groups
    good_group = folium.FeatureGroup(name="Good Condition", show=True)
    damaged_group = folium.FeatureGroup(name="Damaged Condition", show=True)
    missing_group = folium.FeatureGroup(name="Missing Condition", show=True)
    
    # Add markers
    for _, row in data.iterrows():
        lat, lon = row['latitude'], row['longitude']
        condition = row.get('predicted_condition', 'unknown')
        image_name = row.get('original_image', '')
        
        # Create detailed popup with HTML styling
        popup_content = f"""
        <div style="width: 250px; font-family: Arial; padding: 5px;">
            <h4 style="margin-bottom: 10px; color: #2c3e50;">Pavement Marking Details</h4>
            <p><b>Condition:</b> <span style="color: {get_marker_style(condition)[0]}; font-weight: bold;">{condition.upper()}</span></p>
            <p><b>Coordinates:</b> {lat:.6f}, {lon:.6f}</p>
            <p><b>Image:</b> {image_name}</p>
        </div>
        """
        
        # Create marker with tooltip (hover text)
        marker = folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{condition.title()} - Click for details",
            icon=create_custom_icon(condition)
        )
        
        # Add marker to appropriate group
        if condition == 'good':
            marker.add_to(good_group)
        elif condition == 'damaged':
            marker.add_to(damaged_group)
        elif condition == 'missing':
            marker.add_to(missing_group)
        else:
            # If using clusters, add directly to cluster
            if use_clusters:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(m)
    
    # Add all groups to map
    good_group.add_to(m)
    damaged_group.add_to(m)
    missing_group.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    
    return m

# --- Create analytics components ---
def create_condition_chart(data):
    # Count conditions
    condition_counts = data['predicted_condition'].value_counts().reset_index()
    condition_counts.columns = ['Condition', 'Count']
    
    # Create color map for consistency
    color_map = {
        'good': '#38A169',
        'damaged': '#DD6B20',
        'missing': '#E53E3E',
        'unknown': '#718096'
    }
    
    # Create chart
    fig = px.pie(
        condition_counts, 
        values='Count', 
        names='Condition',
        color='Condition',
        color_discrete_map=color_map,
        title="Distribution of Pavement Marking Conditions",
        hole=0.4,
    )
    
    fig.update_layout(
        legend_title="Condition",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",  # Transparent background (or set to your page bg)
        font=dict(color="white") ),
        margin=dict(t=30, b=0, l=0, r=0),
        height=300,
    )
    
    return fig

# --- Handle image loading with error handling ---
def get_image(image_path):
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            # Create a placeholder image if actual image is missing
            placeholder = Image.new('RGB', (400, 300), color=(240, 240, 240))
            d = io.BytesIO()
            placeholder.save(d, format='PNG')
            return d.getvalue()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# --- Main Streamlit App ---
st.title("🛣️ Pavement Marking Condition Viewer")
st.markdown("### Interactive visualization of pavement marking conditions")

# Load data
data = load_data()
if data.empty:
    st.warning("No data available. Please check that your CSV file exists and contains valid data.")
    st.stop()

# Sidebar filters and options
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/flat-winding-road_1284-52216.jpg?semt=ais_hybrid&w=740", width=100)
    st.header("Control Panel")
    
    st.subheader("🔍 Data Filters")
    selected_conditions = st.multiselect(
        "Filter by Condition",
        options=['good', 'damaged', 'missing'],
        default=['good', 'damaged', 'missing']
    )
    
    st.subheader("⚙️ Map Settings")
    use_clustering = st.checkbox("Enable marker clustering", value=True)
    map_height = st.slider("Map height", min_value=400, max_value=800, value=600, step=50)
    
    # Quick stats for sidebar
    st.subheader("📊 Quick Stats")
    total_markings = len(data)
    st.metric("Total Markings", total_markings)
    
    # Quick counts by condition
    for condition in ['good', 'damaged', 'missing']:
        count = len(data[data['predicted_condition'] == condition])
        percentage = (count / total_markings) * 100 if total_markings > 0 else 0
        st.metric(
            f"{condition.title()} Markings", 
            f"{count} ({percentage:.1f}%)"
        )

# Filter data based on selections
filtered_data = data[data['predicted_condition'].str.lower().isin([c.lower() for c in selected_conditions])]

# Create page layout with columns
col1, col2 = st.columns([7, 3])

with col1:
    st.subheader("🗺️ Interactive Map")
    st.markdown("Explore pavement markings across the area. Click on markers for details.  \n **Map colors:** 🟢 Good | 🟠 Damaged | 🔴 Missing")
    
    # Create and display map in a styled container
    with st.container():
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        map_obj = create_map(filtered_data, use_clusters=use_clustering)
        map_data = st_folium(map_obj, width="100%", height=map_height)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Add analytics chart
    st.subheader("📊 Condition Analysis")
    chart = create_condition_chart(filtered_data)
    st.plotly_chart(chart, use_container_width=True)
    
    # Show selected point details + image
    if map_data and map_data.get("last_object_clicked"):
        clicked = map_data["last_object_clicked"]
        lat, lon = clicked['lat'], clicked['lng']
        
        # Find closest match in dataframe (with improved tolerance)
        match = filtered_data.loc[
            ((filtered_data['latitude'] - lat).abs() < 0.0001) &
            ((filtered_data['longitude'] - lon).abs() < 0.0001)
        ]
        
        if not match.empty:
            row = match.iloc[0]
            condition = row['predicted_condition']
            
            # Style based on condition
            condition_class = f"condition-{condition}" if condition in ['good', 'damaged', 'missing'] else "condition-unknown"
            
            # Display selected point info in a styled box
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("### 📍 Selected Marking")
            
            # Show condition with colored badge
            st.markdown(f'<div class="{condition_class}">{condition.upper()}</div>', unsafe_allow_html=True)
            
            # Show other details
            st.write(f"**Coordinates:** {row['latitude']:.6f}, {row['longitude']:.6f}")
            
            # Display image with proper error handling
            image_path = os.path.join(IMAGE_FOLDER, row['original_image'])
            image = get_image(image_path)
            
            if image is not None:
                st.image(image, use_container_width=True)
            else:
                st.warning("Image preview not available.")
                
            st.markdown('</div>', unsafe_allow_html=True)

# Footer with explanatory text
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.8rem;">
    <p>Map colors: 🟢 Good | 🟠 Damaged | 🔴 Missing</p>
    <p>Click on markers to view detailed information and images</p>
</div>
""", unsafe_allow_html=True)

# Run with: streamlit run streamlit_visuals.py