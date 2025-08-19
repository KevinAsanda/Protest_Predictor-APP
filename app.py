import streamlit as st
import pandas as pd
import joblib
import geopandas as gpd
from sklearn.pipeline import Pipeline
import folium
from streamlit_folium import st_folium
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#Load Model and Data
@st.cache_data
def load_data():
    try:
        shapefile = gpd.read_file('za_municipalities/MDB_Local_Municipal_Boundary_2018.shp')
        model = joblib.load('protest_risk_model.joblib')
        data = pd.read_csv('merged_data.csv')
        return model, data, shapefile
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please ensure all required files are present.")
        return None, None, None

model, df, municipalities_gdf = load_data()

if model is None:
    st.stop()

FEATURES = ['Total Population', 'Province name',
    'Piped (tap) water inside dwelling', 'Piped (tap) water inside yard',
       'Piped (tap) water on community stand',
       'No access to piped (tap) water', 'Formal Dwelling',
       'Traditional Dwelling', 'Informal Dwelling', 'Other Dwelling',
       'Flush toilet', 'Chemical toilet', 'Pit toilet', 'Bucket toilet',
       'Other Toilet', 'No toilet',
       'Removed by local authority/private company/community members at least once a week',
       'Removed by local authority/private company/community members less often',
       'Communal refuse dump', 'Communal container/central collection point',
       'Own refuse dump',
       'Dump or leave rubbish anywhere (no rubbish disposal)', 'Other Dump',
       'Electricity for Light', 'Gas for Light', 'Paraffin for Light',
       'Candles for Light', 'Solar for Light', 'Other Source of Light',
       'None Source of Light', 'Electricity for Cooking', 'Gas for Cooking',
       'Paraffin for Cooking', 'Wood for Cooking', 'Coal for Cooking',
       'Animal dung', 'Solar for Cooking', 'Other Source of Cooking',
       'None Source of Cooking'
]

@st.cache_data(ttl=3600, hash_funcs={Pipeline: lambda _: None, gpd.GeoDataFrame: lambda _: None})
def prepare_and_predict(_df, _model, _municipalities_gdf):
    """Prepare data, run predictions and merge with geodata.

    Note: leading underscores on `_model` and `_municipalities_gdf` tell
    Streamlit not to hash these complex, unhashable objects for the cache key.
    """
    # Work on copies to avoid side-effects
    df = _df.copy()

    # Clean Total Population column
    if 'Total Population' in df.columns and df['Total Population'].dtype == 'object':
        df['Total Population'] = df['Total Population'].str.replace(',', '').astype(float)

    # Prepare features and predictions
    X_full = df[FEATURES]
    df['risk_probability'] = _model.predict_proba(X_full)[:, 1]
    df['risk_prediction'] = _model.predict(X_full)

    # Merge predictions with a copy of the geodataframe
    df['municipality_clean'] = df['District/Local municipality name'].str.strip().str.title()
    municipalities_gdf_copy = _municipalities_gdf.copy()
    municipalities_gdf_copy['municipality_clean'] = municipalities_gdf_copy['MUNICNAME'].str.strip().str.title()

    merged_gdf = municipalities_gdf_copy.merge(
        df, on='municipality_clean', how='left'
    )

    merged_gdf['risk_probability'] = merged_gdf['risk_probability'].fillna(0)

    for col in merged_gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(merged_gdf[col]):
            merged_gdf[col] = merged_gdf[col].astype(str)
        elif merged_gdf[col].dtype == 'object' and not pd.api.types.is_string_dtype(merged_gdf[col]):
            merged_gdf[col] = merged_gdf[col].astype(str)

    return merged_gdf

geodata = prepare_and_predict(df, model, municipalities_gdf)

st.set_page_config(page_title="Protest Risk Dashboard", layout="wide")
st.title('🇿🇦 Service Delivery Protest Risk Dashboard')

tab1, tab2 = st.tabs(["Manual Prediction", "Risk Map"])

with tab1:
    st.header("Predict Risk for a Custom Municipality")

    if 'step' not in st.session_state:
        st.session_state.step = 1
        st.session_state.user_inputs = {}

    def next_step():
        st.session_state.step += 1
    def prev_step():
        st.session_state.step -= 1
    def next():
        st.session_state.step += 1

    if st.session_state.step == 1:
        st.subheader("Step 1: General Information")
        with st.container():
            st.session_state.user_inputs['Total Population'] = st.number_input('Total Population', min_value=0, value=st.session_state.user_inputs.get('Total Population', 1000), step=100)
            st.session_state.user_inputs['Province name'] = st.selectbox('Province', options=sorted(geodata['PROVINCE'].unique().tolist()), index=0)
            st.button("Next", on_click= next, type="primary")

    elif st.session_state.step == 2:
        st.subheader("Step 2: Water Access")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.user_inputs['Piped (tap) water inside dwelling'] = st.number_input('% Water in Dwelling', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Piped (tap) water inside dwelling', 0.5), step=0.01)
                st.session_state.user_inputs['Piped (tap) water inside yard'] = st.number_input('% Water in yard', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Piped (tap) water inside yard', 0.1), step=0.01)
                st.session_state.user_inputs['Piped (tap) water on community stand'] = st.number_input('% Water on community stand', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Piped (tap) water on community stand', 0.7), step=0.01)
                st.session_state.user_inputs['No access to piped (tap) water'] = st.number_input('% No access to clean water', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get(' No access to piped (tap) water', 0.2), step=0.01)
        c1, c2 = st.columns(2)
        c1.button("Previous", on_click=prev_step)
        c2.button("Next", on_click=next_step(), type="primary")

    elif st.session_state.step == 3:
        st.subheader("Step 3: Dwelling")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.user_inputs['Formal Dwelling'] = st.number_input('% Formal housing', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Formal Dwelling', 0.5), step=0.01)
                st.session_state.user_inputs['Traditional Dwelling'] = st.number_input('% Traditional housing', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Traditional Dwelling', 0.1), step=0.01)
                st.session_state.user_inputs['Other Dwelling'] = st.number_input('% Other housing', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Other Dwelling', 0.2), step=0.01)
                st.session_state.user_inputs['Informal Dwelling'] = st.number_input('% Informal housing', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Informal Dwelling', 0.7), step=0.01)
            c1, c2 = st.columns(2)
            c1.button("Previous", on_click=prev_step)
            c2.button("Next", on_click=next_step, type="primary")

    elif st.session_state.step == 4:
        st.subheader("Step 4: Toilet System")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.user_inputs['Flush toilet'] = st.number_input('% Flush toilet', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Flush toilet', 0.5), step=0.01)
                st.session_state.user_inputs['Chemical toilet'] = st.number_input('% Chemical toilet', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Chemical toilet', 0.1), step=0.01)
                st.session_state.user_inputs['Pit toilet'] = st.number_input('% Pit toilet', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Pit toilet', 0.7), step=0.01)
                st.session_state.user_inputs['Bucket toilet'] = st.number_input('% Bucket toilet', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Bucket toilet', 0.2), step=0.01)
                st.session_state.user_inputs['Other Toilet'] = st.number_input('% Other Toilet', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Other Toilet', 0.2), step=0.01)
                st.session_state.user_inputs['No toilet'] = st.number_input('% No toilet', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('No toilet', 0.1), step=0.01)
            c1, c2 = st.columns([1, 1])
            c1.button("Previous", on_click=prev_step)
            c2.button("Next", on_click=next_step, type="primary")

    elif st.session_state.step == 5:
        st.subheader("Step 5: Trash removal")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.user_inputs['Removed by local authority/private company/community members at least once a week'] = st.number_input('% Removed by local authority/private company/community members at least once a week', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Removed by local authority/private company/community members at least once a week', 0.5), step=0.01)
                st.session_state.user_inputs['Removed by local authority/private company/community members less often'] = st.number_input('% Removed by local authority/private company/community members less often', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Removed by local authority/private company/community members less often', 0.1), step=0.01)
                st.session_state.user_inputs['Communal refuse dump'] = st.number_input('% Communal refuse dump', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Communal refuse dump', 0.7), step=0.01)
                st.session_state.user_inputs['Communal container/central collection point'] = st.number_input('% Communal container/central collection point', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Communal container/central collection point', 0.2), step=0.01)
                st.session_state.user_inputs['Own refuse dump'] = st.number_input('% Own refuse dump', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Own refuse dump', 0.5), step=0.01)
                st.session_state.user_inputs['Dump or leave rubbish anywhere (no rubbish disposal)'] = st.number_input('% Dump or leave rubbish anywhere (no rubbish disposal)', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Dump or leave rubbish anywhere (no rubbish disposal)', 0.1), step=0.01)
                st.session_state.user_inputs['Other Dump'] = st.number_input('% Other Dump', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Other Dump', 0.2), step=0.01)
            c1, c2 = st.columns([1, 1])
            c1.button("Previous", on_click=prev_step)
            c2.button("Next", on_click=next_step, type="primary")

    elif st.session_state.step == 6:
        st.subheader("Step 6: Source of Light")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.user_inputs['Electricity for light'] = st.number_input('% Electricity for light', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Electricity for light', 0.5), step=0.01)
                st.session_state.user_inputs['Gas for light'] = st.number_input('% Gas for light ', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Gas for light', 0.1), step=0.01)
                st.session_state.user_inputs['Paraffin for light'] = st.number_input('% Paraffin for light', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Paraffin for light', 0.7), step=0.01)
                st.session_state.user_inputs['Other Source of light'] = st.number_input('% Other Source of light', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Other Source of light', 0.2), step=0.01)
                st.session_state.user_inputs['Candles for light'] = st.number_input('% Candles for light', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Candles for light', 0.2), step=0.01)
                st.session_state.user_inputs['Solar for light'] = st.number_input('% Solar for light', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Solar for light', 0.5), step=0.01)
                st.session_state.user_inputs['None Source of light'] = st.number_input('% No Source of light', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('None Source of light', 0.1), step=0.01)
            c1, c2 = st.columns([1, 1])
            c1.button("Previous", on_click=prev_step)
            c2.button("Next", on_click=next_step, type="primary")

    elif st.session_state.step == 7:
        st.subheader("Step 7: Source of Cooking")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.user_inputs['Paraffin for cooking'] = st.number_input('% Paraffin for cooking', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Paraffin for cooking', 0.5), step=0.01)
                st.session_state.user_inputs['Wood for cooking'] = st.number_input('% Wood for cooking', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Wood for cooking', 0.1), step=0.01)
                st.session_state.user_inputs['Coal for cooking'] = st.number_input('% Coal for cooking', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Coal for cooking', 0.7), step=0.01)
                st.session_state.user_inputs['Gas for cooking'] = st.number_input('% Gas for cooking', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Gas for cooking', 0.2), step=0.01)
                st.session_state.user_inputs['Electricity for cooking'] = st.number_input('% Electricity for cooking', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Electricity for cooking', 0.5), step=0.01)
                st.session_state.user_inputs['Animal dung'] = st.number_input('% Animal dung', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Animal dung', 0.2), step=0.01)
                st.session_state.user_inputs['Other Source of cooking'] = st.number_input('% Other Source of cooking', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Other Source of cooking', 0.2), step=0.01)
                st.session_state.user_inputs['Solar for cooking'] = st.number_input('% Solar for cooking', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('Solar for cooking', 0.5), step=0.01)
                st.session_state.user_inputs['None source of cooking'] = st.number_input('% No Source of cooking', min_value=0.0, max_value=1.0, value=st.session_state.user_inputs.get('None Source of cooking', 0.1), step=0.01)
            c1, c2 = st.columns([1, 1])
            c1.button("Previous", on_click=prev_step)
            
            if c2.button("Predict Risk", type="primary"):
                # Validate that the numeric inputs provided by the user (only those matching FEATURES)
                # sum to 1. This avoids including non-feature fields like 'Province name'.
                numeric_keys = [k for k in st.session_state.user_inputs.keys() if k in FEATURES]
                numeric_values = [st.session_state.user_inputs[k] for k in numeric_keys if isinstance(st.session_state.user_inputs[k], (int, float))]
                if not numeric_values:
                    st.error("No numeric feature inputs found to validate. Please fill the form before predicting.")
                else:
                        # Create a DataFrame from the user's input in the correct order
                        input_df = pd.DataFrame([st.session_state.user_inputs])

                        # Ensure the column order matches the training data
                        final_input_df = input_df

                        # Make predictions
                        prediction_code = model.predict(final_input_df)[0]
                        prediction_prob = model.predict_proba(final_input_df)[0][1] # Probability of a riot

                        # Display the results
                        st.header("Prediction Result")
                        if prediction_code == 1:
                            st.error("High Risk of Riot")
                        else:
                            st.success("Low Risk of Riot")

                        st.metric(label="Probability of a Riot", value=f"{prediction_prob*100:.2f}%")
                        
                        if prediction_prob > 0.5:
                            st.warning("Intervention is recommended based on the provided socio-economic indicators.")
                        else:
                            st.info("Continue monitoring. The risk is currently low.")
