import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Blended regressor class - this matches the one used during model training
class BlendedRegressor:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        
    def predict(self, X):
        final_pred = 0
        for name, weight in self.weights.items():
            final_pred += weight * self.models[name].predict(X)
        return final_pred

    def fit(self, X, y):
        pass

# Page configuration
st.set_page_config(
    page_title="Ames Housing AI Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Make metrics look like cards */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    
    /* Subtle tab styling */
    div[data-baseweb="tab-list"] {gap: 10px;}
    div[data-baseweb="tab"] {
        border-radius: 5px 5px 0 0; 
        padding-top: 5px; 
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing resources
@st.cache_resource
def load_resources():
    """Load the trained model and required column names"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "best_model.pkl")
    columns_path = os.path.join(current_dir, "models", "model_columns.pkl")
    
    # Fallback for different folder structures
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, "..", "models", "best_model.pkl")
        columns_path = os.path.join(current_dir, "..", "models", "model_columns.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None, None
    
    try:
        model = joblib.load(model_path)
        model_cols = joblib.load(columns_path)
        return model, model_cols
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

model, model_cols = load_resources()

# Header section
st.title("🏡 Ames Housing Price Estimator")
st.markdown("### Professional AI Valuation Tool")
st.markdown("Enter the property details below to generate a market value prediction based on historical data from Ames, Iowa.")

# User input form organized in tabs
def user_input_features():
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 Location & General", 
        "📐 Areas & Layout", 
        "✨ Quality & Condition", 
        "🛁 Utilities & Extras"
    ])
    
    # Tab 1: Location and general info
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            neighborhood = st.selectbox("Neighborhood", 
                ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 
                 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 
                 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 
                 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'],
                 index=16)
            
            ms_zoning = st.selectbox("Zoning Classification", 
                                     ['RL', 'RM', 'FV', 'RH', 'C (all)'],
                                     help="RL: Residential Low Density, RM: Medium Density, FV: Village")
            
            bldg_type = st.selectbox("Building Type", ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'])

        with col2:
            sale_type = st.selectbox("Sale Type", 
                                     ['WD', 'New', 'COD', 'Con', 'Other'],
                                     help="Select 'New' if the house is just built.")
            
            house_style = st.selectbox("House Style", ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer'])

    # Tab 2: Areas and layout (moved lot frontage here)
    with tab2:
        st.info("ℹ️ **Tip:** Living Area is usually the sum of 1st and 2nd Floor.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Living Spaces (sq ft)**")
            first_flr_sf = st.number_input("1st Floor Area", 0, 5000, 1200)
            second_flr_sf = st.number_input("2nd Floor Area", 0, 4000, 0)
            
            # Auto-calculate suggestion for GrLivArea
            calc_grliv = first_flr_sf + second_flr_sf
            gr_liv_area = st.number_input("Total Living Area (GrLivArea)", 0, 8000, calc_grliv)
        
        with c2:
            st.markdown("**Basement & Lot**")
            total_bsmt_sf = st.number_input("Total Basement Area", 0, 5000, 1000)
            lot_area = st.number_input("Lot Area (sq ft)", 0, 100000, 9000)
            lot_frontage = st.number_input("Lot Frontage (ft)", 0, 500, 70)
        
        with c3:
            st.markdown("**Garage & Exterior**")
            garage_area = st.number_input("Garage Area", 0, 2000, 500)
            wood_deck = st.number_input("Wood Deck Area", 0, 1000, 0)
            open_porch = st.number_input("Open Porch Area", 0, 1000, 50)

    # Tab 3: Quality and condition
    with tab3:
        c4, c5 = st.columns(2)
        with c4:
            overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6, help="10 is Very Excellent")
            year_built = st.number_input("Year Built", 1870, 2025, 2005)
            
        with c5:
            overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5)
            year_remod = st.number_input("Year Remodeled", 1950, 2025, 2005)

    # Tab 4: Utilities and extras
    with tab4:
        c6, c7 = st.columns(2)
        with c6:
            st.write("**Above Ground Bathrooms**")
            full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
            half_bath = st.number_input("Half Bathrooms", 0, 3, 1)
        with c7:
            st.write("**Basement Bathrooms**")
            bsmt_full = st.number_input("Bsmt Full Bathrooms", 0, 3, 0)
            bsmt_half = st.number_input("Bsmt Half Bathrooms", 0, 2, 0)
        
        st.write("**Additional Features**")
        fireplaces = st.slider("Number of Fireplaces", 0, 4, 1)

    # Compile all data into dictionary
    data = {
        # Categorical features
        'MSZoning': ms_zoning, 'SaleType': sale_type, 'Neighborhood': neighborhood,
        'BldgType': bldg_type, 'HouseStyle': house_style,
        
        # Year information
        'YearBuilt': year_built, 'YearRemodAdd': year_remod,
        
        # Area measurements (this is important for log transformations later)
        'LotFrontage': lot_frontage, 'LotArea': lot_area,
        '1stFlrSF': first_flr_sf, '2ndFlrSF': second_flr_sf, 'GrLivArea': gr_liv_area,
        'TotalBsmtSF': total_bsmt_sf, 'GarageArea': garage_area,
        'OpenPorchSF': open_porch, 'WoodDeckSF': wood_deck,
        
        # Quality and counts
        'OverallQual': overall_qual, 'OverallCond': overall_cond,
        'FullBath': full_bath, 'HalfBath': half_bath, 
        'BsmtFullBath': bsmt_full, 'BsmtHalfBath': bsmt_half, 'Fireplaces': fireplaces,
        
        # Default values for hidden parameters required by model
        'PoolArea': 0, 'EnclosedPorch': 0, 'ScreenPorch': 0, '3SsnPorch': 0,
        'MSSubClass': '20', 'Functional': 'Typ', 'Heating': 'GasA', 
        'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd', 'SaleCondition': 'Normal',
        'KitchenAbvGr': 1, 'TotRmsAbvGrd': 6, 'GarageCars': 2,
        'BsmtUnfSF': 0, 'BsmtFinSF1': 0
    }
    
    return pd.DataFrame([data])

input_df = user_input_features()

st.markdown("---")

# Prediction button 
col_space_1, col_btn, col_space_2 = st.columns([2, 1, 2])
with col_btn:
    predict_btn = st.button("Calculate Price", type="primary", use_container_width=True)

if predict_btn:
    if model and model_cols is not None:
        try:
            with st.spinner('Analyzing market data...'):
                
                # Step 1: Feature engineering on raw data
                # Calculate total bathrooms (no need to log these)
                input_df['Total_Bathrooms'] = (
                    input_df['FullBath'] + 
                    (0.5 * input_df['HalfBath']) + 
                    input_df['BsmtFullBath'] + 
                    (0.5 * input_df['BsmtHalfBath'])
                )

                # Create binary features based on raw data
                input_df['haspool'] = 0
                input_df['has2ndfloor'] = input_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
                input_df['hasgarage'] = input_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
                input_df['hasbsmt'] = input_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
                input_df['hasfireplace'] = input_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

                # Step 2: Apply log transformations to skewed features
                skewed_vars = [
                    'TotalBsmtSF', 
                    'GarageArea', 
                    'GrLivArea', 
                    '1stFlrSF', 
                    '2ndFlrSF', 
                    'LotArea',
                    'LotFrontage',
                    'OpenPorchSF',
                    'WoodDeckSF'
                ]
                
                for col in skewed_vars:
                    if col in input_df.columns:
                        input_df[col] = np.log1p(input_df[col])

                # Step 3: Create derived features in log space
                # TotalSF - sum in logarithmic space 
                input_df['TotalSF'] = (
                    input_df['TotalBsmtSF'] + 
                    input_df['1stFlrSF'] + 
                    input_df['2ndFlrSF']
                )
                
                # TotalPorch - sum in logarithmic space
                input_df['TotalPorch'] = (
                    input_df['OpenPorchSF'] + 
                    input_df['WoodDeckSF']
                )

                # Step 4: Drop multicollinear columns
                cols_to_drop = ['GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars']
                input_df = input_df.drop(
                    [c for c in cols_to_drop if c in input_df.columns], 
                    axis=1
                )

                # Step 5: Encode categorical variables and align columns
                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=model_cols, fill_value=0)

                # Step 6: Make prediction
                log_pred = model.predict(input_df)
                final_price = np.expm1(log_pred)[0]

            # Display results
            st.success("Analysis Complete!")
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.metric(label="Estimated Value", value=f"${final_price:,.2f}")
            
            with c_res2:
                # Calculate price per square foot (reverse log for display)
                raw_grliv = np.expm1(input_df['GrLivArea'].values[0])
                if raw_grliv > 0:
                    ppsq = final_price / raw_grliv
                    st.metric(label="Price per Sq Ft", value=f"${ppsq:.2f}")

            # Show contextual information
            if 'SaleType_New' in input_df.columns and input_df['SaleType_New'].values[0] == 1:
                st.info("💎 Premium applied for New Construction.")
            
            if ('Neighborhood_StoneBr' in input_df.columns and input_df['Neighborhood_StoneBr'].values[0] == 1) or \
               ('Neighborhood_NridgHt' in input_df.columns and input_df['Neighborhood_NridgHt'].values[0] == 1):
                st.info("📈 High-value neighborhood detected.")

        except Exception as e:
            st.error(f"Prediction Failed: {e}")
            st.warning("Please check if 'model_columns.pkl' is generated from the latest notebook run.")
            import traceback
            st.code(traceback.format_exc())