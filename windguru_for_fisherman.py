"""
    A small script to calculate a good day for fishing based on same data provided bu windguru
    Created on:      23-Oct-2025
    Original author: Adriano Jordão (adriano.jordao@gmail.com)
"""

import io
from zoneinfo import ZoneInfo
import sys
import streamlit as st
from windguru_scraper import WindguruScraper

# Page configuration
st.set_page_config(
    page_title="Windguru Fishing Forecast",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make sidebar wider
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎣 Windguru Fishing Forecast")
st.markdown("Find the best days for fishing based on weather conditions")

# Initialize session state for form values
if 'wave_height_min' not in st.session_state:
    st.session_state.wave_height_min = 0.0
if 'wave_height_max' not in st.session_state:
    st.session_state.wave_height_max = 1.6
if 'wave_period_min' not in st.session_state:
    st.session_state.wave_period_min = 0.0
if 'wave_period_max' not in st.session_state:
    st.session_state.wave_period_max = 12.0
if 'wind_speed_min' not in st.session_state:
    st.session_state.wind_speed_min = 0.0
if 'wind_speed_max' not in st.session_state:
    st.session_state.wind_speed_max = 20.0
if 'wind_gust_min' not in st.session_state:
    st.session_state.wind_gust_min = 0.0
if 'wind_gust_max' not in st.session_state:
    st.session_state.wind_gust_max = 25.0
if 'spot_url' not in st.session_state:
    st.session_state.spot_url = "https://www.windguru.cz/65000"
if 'spot_name' not in st.session_state:
    st.session_state.spot_name = "Sesimbra"

# Sidebar with form
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.form("fishing_params"):
        st.subheader("Location")
        spot_name = st.text_input(
            "Spot Name",
            value=st.session_state.spot_name,
            help="Name for your fishing spot"
        )

        spot_url = st.text_input(
            "Windguru URL",
            value=st.session_state.spot_url,
            help="Full URL from windguru.cz (e.g., https://www.windguru.cz/65000)"
        )

        st.divider()
        st.subheader("🌊 Wave Conditions")

        col1, col2 = st.columns(2)
        with col1:
            wave_height_min = st.number_input(
                "Min Height (m)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.wave_height_min,
                step=0.1
            )
        with col2:
            wave_height_max = st.number_input(
                "Max Height (m)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.wave_height_max,
                step=0.1
            )

        col1, col2 = st.columns(2)
        with col1:
            wave_period_min = st.number_input(
                "Min Period (s)",
                min_value=0.0,
                max_value=30.0,
                value=st.session_state.wave_period_min,
                step=1.0
            )
        with col2:
            wave_period_max = st.number_input(
                "Max Period (s)",
                min_value=0.0,
                max_value=30.0,
                value=st.session_state.wave_period_max,
                step=1.0
            )

        st.divider()
        st.subheader("💨 Wind Conditions")

        col1, col2 = st.columns(2)
        with col1:
            wind_speed_min = st.number_input(
                "Min Speed (km/h)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.wind_speed_min,
                step=1.0
            )
        with col2:
            wind_speed_max = st.number_input(
                "Max Speed (km/h)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.wind_speed_max,
                step=1.0
            )

        col1, col2 = st.columns(2)
        with col1:
            wind_gust_min = st.number_input(
                "Min Gust (km/h)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.wind_gust_min,
                step=1.0
            )
        with col2:
            wind_gust_max = st.number_input(
                "Max Gust (km/h)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.wind_gust_max,
                step=1.0
            )

        # Submit button
        submitted = st.form_submit_button("🎣 Check Forecast", use_container_width=True)

# Main content area
if submitted:
    # Update session state
    st.session_state.wave_height_min = wave_height_min
    st.session_state.wave_height_max = wave_height_max
    st.session_state.wave_period_min = wave_period_min
    st.session_state.wave_period_max = wave_period_max
    st.session_state.wind_speed_min = wind_speed_min
    st.session_state.wind_speed_max = wind_speed_max
    st.session_state.wind_gust_min = wind_gust_min
    st.session_state.wind_gust_max = wind_gust_max
    st.session_state.spot_url = spot_url
    st.session_state.spot_name = spot_name

    # Build ranges dictionary
    user_ranges = {
        "wave_height_m": (wave_height_min, wave_height_max),
        "wave_period_s": (wave_period_min, wave_period_max),
        "wind_speed_kmh": (wind_speed_min, wind_speed_max),
        "wind_gust_kmh": (wind_gust_min, wind_gust_max),
    }

    # Show loading spinner
    with st.spinner(f"🌊 Fetching forecast for {spot_name}..."):
        try:
            # Capture print output
            output_buffer = io.StringIO()
            sys.stdout = output_buffer

            # Create scraper and fetch data
            scraper = WindguruScraper(spot_url, spot_title_override=spot_name)
            df = scraper.fetch_data()
            
            # Restore stdout
            sys.stdout = sys.__stdout__
            
            if df.empty:
                st.error("❌ No forecast data retrieved. Please check the URL and try again.")
                with st.expander("📋 Debug Output"):
                    st.text(output_buffer.getvalue())
            else:
                st.success(f"✅ Successfully fetched {len(df)} forecast records!")
                
                # Filter conditions
                filtered = scraper.filter_conditions(
                    wave_height_range=user_ranges["wave_height_m"],
                    wave_period_range=user_ranges["wave_period_s"],
                    wind_speed_range=user_ranges["wind_speed_kmh"],
                    wind_gust_range=user_ranges["wind_gust_kmh"],
                )
                
                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Good Conditions", len(filtered))
                with col3:
                    percentage = (len(filtered) / len(df) * 100) if len(df) > 0 else 0
                    st.metric("Match Rate", f"{percentage:.1f}%")
                
                # Generate plot
                st.subheader("📊 Forecast Visualization")
                with st.spinner("Generating chart..."):
                    try:
                        # Generate plot (don't show, save to buffer)
                        import matplotlib
                        matplotlib.use('Agg')  # Use non-interactive backend
                        
                        scraper.plot_forecast(
                            ranges=user_ranges,
                            show_week_from=None,
                            figsize=(20, 11),
                            marker_size=3.0,
                            line_width=1.0,
                            save_path="temp_forecast.png",
                            show=False  # Don't block
                        )
                        
                        # Display the image
                        st.image("temp_forecast.png", use_container_width=True)
                        
                        # Offer download
                        with open("temp_forecast.png", "rb") as file:
                            st.download_button(
                                label="📥 Download Chart",
                                data=file,
                                file_name=f"windguru_{spot_name.lower().replace(' ', '_')}.png",
                                mime="image/png"
                            )
                    except Exception as e:
                        st.error(f"Error generating plot: {e}")
                
                # Show filtered data table
                if not filtered.empty:
                    with st.expander("📋 View Good Fishing Hours"):
                        # Convert to local timezone for display
                        display_df = filtered.copy()
                        tz_pt = ZoneInfo("Europe/Lisbon")
                        display_df['local_time'] = display_df['datetime'].dt.tz_convert(tz_pt)
                        display_df = display_df[['local_time', 'model', 'wave_height_m', 
                                                   'wave_period_s', 'wind_speed_kmh', 'wind_gust_kmh']]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Debug output
                with st.expander("🔍 Debug Output"):
                    st.text(output_buffer.getvalue())
                    
        except Exception as e:
            sys.stdout = sys.__stdout__
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
else:
    # Welcome screen
    st.info("👈 Configure your fishing conditions in the sidebar and click '🎣 Check Forecast' to get started!")
    
    st.markdown("""
    ### How to use:
    1. **Enter your Windguru URL** - Find your spot on windguru.cz and copy the URL
    2. **Set your ideal conditions** - Define acceptable ranges for waves and wind
    3. **Click Check Forecast** - Get a visual forecast with color-coded good/bad periods
    
    ### Color coding:
    - 🟢 **Green**: All conditions within your ranges - perfect for fishing!
    - 🟡 **Yellow**: One condition slightly outside range - still manageable
    - 🔴 **Red**: Multiple conditions outside range - not recommended
    
    ### Popular spots:
    - Sesimbra: `https://www.windguru.cz/65000`
    - Cascais: `https://www.windguru.cz/574`
    - Costa da Caparica: `https://www.windguru.cz/48963`
    """)

# Footer
st.divider()
st.markdown("Made with ❤️ for fishermen by Adriano Jordão & Manuel Monteiro | Data from [Windguru](https://www.windguru.cz)")