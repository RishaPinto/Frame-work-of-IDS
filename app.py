import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import plotly.express as px  # For our charts
from sklearn.preprocessing import StandardScaler
import time # To make spinners more visible

# --- 0. Page Configuration ---
st.set_page_config(
    page_title="IDS Analysis Dashboard",
    page_icon="🛡",
    layout="wide"
)


# --- 1. Load Model, Scaler, and Required Columns ---
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load("intrusion_detection_model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        # --- THIS IS THE FIX ---
        # We get the required column list directly from the scaler.
        # This GUARANTEES the order is 100% correct.
        # The 'training_columns.json' file is no longer needed.
        REQUIRED_COLUMNS_LIST = list(scaler.feature_names_in_)
        
        return model, scaler, REQUIRED_COLUMNS_LIST
    except FileNotFoundError:
        st.error("FATAL ERROR: intrusion_detection_model.pkl or scaler.pkl not found.", icon="🚨")
        st.info("Please make sure all model files are in the same folder as your app.")
        return None, None, None
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load model assets. Error: {e}")
        return None, None, None

model, scaler, REQUIRED_COLUMNS_LIST = load_model_assets()

# Check if model loading failed
if model is None:
    st.stop() # Stop the app if files are missing

REQUIRED_COLUMNS_SET = set(REQUIRED_COLUMNS_LIST)
attack_labels = {
    0: 'BENIGN', 1: 'Bot', 2: 'BruteForce',
    3: 'DoS', 4: 'Infiltration', 5: 'PortScan', 6: 'WebAttack'
}

# ===================================================================
# --- 2. Custom CSS Styling (Same as before) ---
# ===================================================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1E2E;
    }
    [data-testid="stSidebar"] .st-emotion-cache-16txtl3 { /* Sidebar title */
        font-size: 2.5rem;
        color: #00C09B;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #262730;
        border-radius: 8px 8px 0 0;
        gap: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0E1117;
        border-bottom: 2px solid #00C09B;
    }

    /* Metric boxes */
    [data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 20px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
    }
    
    /* Custom button */
    .stDownloadButton button {
        background-color: #00C09B;
        color: #FFFFFF;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stDownloadButton button:hover {
        background-color: #00A080;
    }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# --- 3. Login & Session State Management ---
# ===================================================================

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'is_compatible' not in st.session_state:
    st.session_state.is_compatible = False
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'normal_count' not in st.session_state:
    st.session_state.normal_count = 0
if 'attack_count' not in st.session_state:
    st.session_state.attack_count = 0
if 'missing_cols' not in st.session_state:
    st.session_state.missing_cols = []

# --- Simple Login Function ---
def login_form():
    st.markdown(f"""
    <style>
        [data-testid="stVerticalBlock"] {{
            max-width: 450px;
            margin: auto;
            padding: 40px;
            background-color: #1E1E2E;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.title("🛡 Welcome to the IDS Dashboard")
        st.subheader("Please log in to continue")
        
        # For your major project, you can change these to any username/password
        PROJECT_USERNAME = "admin"
        PROJECT_PASSWORD = "password123"

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Hint: admin")
            password = st.text_input("Password", type="password", placeholder="Hint: password123")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if username == PROJECT_USERNAME and password == PROJECT_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun() # Rerun the script to show the main app
                else:
                    st.error("Invalid username or password", icon="❌")

# --- Main App Function (what to show after login) ---
def main_dashboard():
    # ===================================================================
    # --- 3. Sidebar ---
    # ===================================================================
    st.sidebar.title("🛡 IDS Project")
    
    # --- NEW: Logout Button ---
    if st.sidebar.button("Log Out", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun() # Rerun to show the login screen
        
    st.sidebar.info(
        "This is a prototype of a *Behavior-Based Intrusion Detection System* "
        "using a LightGBM model. It's designed as a *Diagnostic Tool* "
        "for analyzing network traffic logs."
    )

    st.sidebar.header("How to Use")
    st.sidebar.markdown("""
    1.  *Upload a CSV file* in the correct format.
    2.  The app performs a *Smart Validation Check*.
    3.  If compatible, the *Analysis Dashboard* will show:
        * Key metrics (Total, Normal, Attack).
        * A pie chart of the results.
        * A breakdown of all attacks found.
    4.  You can view the full, predicted data in the *Raw Data* tab.
    """)

    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV in the CICIDS2017 format.", type=["csv"])

    with st.sidebar.expander("See Required Columns"):
        st.json(REQUIRED_COLUMNS_LIST[:10])
    
    # ===================================================================
    # --- 4. Main Page (The Dashboard with TABS) ---
    # ===================================================================
    st.title("Advanced Network Traffic Analysis Dashboard")

    tab_welcome, tab_dashboard, tab_data = st.tabs(["🏠 Welcome", "📊 Analysis Dashboard", "📄 Full Data Report"])

    # This runs immediately when a file is uploaded, not inside a tab
    if uploaded_file is not None:
        # Check if we've already analyzed this specific file
        if 'last_file_name' not in st.session_state or st.session_state.last_file_name != uploaded_file.name:
            st.session_state.last_file_name = uploaded_file.name
            st.session_state.analysis_complete = False
            
        if not st.session_state.analysis_complete:
            with st.spinner('Analyzing file compatibility...'):
                time.sleep(1) 
                df_raw = pd.read_csv(uploaded_file)
                
                # --- The Smart Validation Step ---
                categorical_cols = df_raw.select_dtypes(include=['object']).columns
                df_encoded = pd.get_dummies(df_raw, columns=categorical_cols)
                uploaded_columns = set(df_encoded.columns)
                
                matching_cols = REQUIRED_COLUMNS_SET.intersection(uploaded_columns)
                missing_cols = list(REQUIRED_COLUMNS_SET.difference(uploaded_columns))
                match_percentage = (len(matching_cols) / len(REQUIRED_COLUMNS_SET)) * 100

            if match_percentage < 90:
                st.session_state.is_compatible = False
                st.session_state.missing_cols = missing_cols
                st.sidebar.error(f"Incompatible File: {match_percentage:.0f}% match.", icon="❌")
            else:
                st.session_state.is_compatible = True
                st.sidebar.success("File is compatible!", icon="✅")
                
                with st.spinner('Running AI model on all packets... This may take a moment.'):
                    time.sleep(1) 
                    # --- The Prediction Pipeline ---
                    
                    # A. Align columns to match the training dictionary
                    #    We use REQUIRED_COLUMNS_LIST to ensure the order is correct!
                    df_aligned = df_encoded.reindex(columns=REQUIRED_COLUMNS_LIST, fill_value=0)
                    
                    # B. Scale the data using the loaded scaler
                    df_scaled = scaler.transform(df_aligned)
                    
                    # C. Make predictions on the scaled data
                    preds = model.predict(df_scaled)
                    
                    df_raw['Predicted_Label'] = [attack_labels.get(int(p), 'Unknown') for p in preds]
                    
                    st.session_state.df_raw = df_raw
                    st.session_state.normal_count = (df_raw['Predicted_Label'] == 'BENIGN').sum()
                    st.session_state.attack_count = len(df_raw) - st.session_state.normal_count
                
                st.session_state.analysis_complete = True
                st.sidebar.success("Analysis Complete! ✅")
                st.sidebar.markdown("### 👉 Click the '📊 Analysis Dashboard' tab to see your results!")

    # --- Tab 1: Welcome Page ---
    with tab_welcome:
        st.header("Welcome to the AI-Powered IDS")
        st.write(
            "This tool is a demonstration of how Machine Learning can be used to detect cyberattacks "
            "that traditional, rule-based systems might miss. It's an example of a *'Proposed System'* "
            "that uses behavioral analysis to find threats."
        )
        st.image("https://placehold.co/1200x400/0E1117/FFFFFF?text=Network+Security+AI&font=inter", use_column_width=True)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("The 'Existing System' (Signature-Based)")
            st.markdown("""
            Traditional systems are like a security guard with a photo album of known criminals.
            - *Method:* Matches a "fingerprint" from a list.
            - *Weakness:* *Completely BLIND* to new, "zero-day" attacks.
            - *Analogy:* A simple virus scanner.
            """)
        with col2:
            st.subheader("Our 'Proposed System' (Behavior-Based AI)")
            st.markdown("""
            Our system is a smart detective who can profile suspects based on behavior.
            - *Method:* Understands patterns using an AI model.
            - *Strength:* *Can detect* new attacks based on suspicious behavior.
            - *AnalLogy:* An AI-powered profiling expert.
            """)
        st.divider()
        st.subheader("The Smart Validation Pipeline")
        st.write(
            "A key feature of this application is its *robustness*. Our model was trained to read a specific data format (like 'German'). "
            "This app first checks the 'language' of the uploaded file."
        )
        st.markdown("""
        - *If the file is compatible (100% match):* It runs the prediction.
        - *If the file is incompatible (0% match):* It stops and warns the user.
        """)
        st.warning("This 'Incompatible File' message is a *critical success feature*, not a bug. It prevents the AI from being fooled by bad data and giving a false 'BENIGN' result.")

    # --- Tab 2: The Main Dashboard ---
    with tab_dashboard:
        if uploaded_file is None:
            st.info("Please upload your network traffic CSV file using the sidebar to begin analysis.", icon="⬆")
        elif not st.session_state.is_compatible:
            st.error(f"*Error: Incompatible File.* This model cannot read this file.", icon="❌")
            st.write("The model was trained on a different data format and is missing critical features.")
            st.write(f"*Example missing columns:*")
            st.json(st.session_state.missing_cols[:10])
        elif st.session_state.analysis_complete:
            st.header("Analysis Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Packets Analyzed", f"{len(st.session_state.df_raw):,}")
            col2.metric("✅ Normal Packets", f"{st.session_state.normal_count:,}")
            col3.metric("🚨 Attacks Detected", f"{st.session_state.attack_count:,}")
            st.divider()
            col_pie, col_bar = st.columns(2)
            with col_pie:
                st.subheader("Normal vs. Attack")
                if st.session_state.attack_count > 0 or st.session_state.normal_count > 0:
                    pie_data = pd.DataFrame({'Category': ['Normal', 'Attack'], 'Count': [st.session_state.normal_count, st.session_state.attack_count]})
                    fig_pie = px.pie(pie_data, names='Category', values='Count', 
                                     color='Category', 
                                     color_discrete_map={'Normal':'#00C09B', 'Attack':'#FF4B4B'},
                                     hole=0.3)
                    fig_pie.update_layout(legend_title_text='Category', paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="#FAFAFA")
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No data to display in chart.")
            with col_bar:
                st.subheader("Attack Type Breakdown")
                if st.session_state.attack_count > 0:
                    attack_df = st.session_state.df_raw[st.session_state.df_raw['Predicted_Label'] != 'BENIGN']
                    attack_counts = attack_df['Predicted_Label'].value_counts().reset_index()
                    attack_counts.columns = ['Attack Type', 'Count']
                    st.dataframe(attack_counts, use_container_width=True)
                    fig_bar = px.bar(attack_counts, x='Attack Type', y='Count', 
                                     title="Count by Attack Type", color='Attack Type')
                    fig_bar.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="#FAFAFA")
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No attacks were detected to break down.")
        else:
            st.info("File uploaded, but analysis is not yet complete or failed.", icon="⌛")

    # --- Tab 3: The Full Data Report ---
    with tab_data:
        st.header("Full Data with Predictions")
        if uploaded_file is None:
            st.info("Upload a file in the sidebar to see the full data report here.", icon="⬆")
        elif not st.session_state.is_compatible:
            st.error("Cannot display data from an incompatible file.", icon="❌")
        elif st.session_state.analysis_complete:
            st.dataframe(st.session_state.df_raw)
            csv = st.session_state.df_raw.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Results as CSV", csv, "prediction_results.csv", "text/csv", use_container_width=True)
        else:
            st.info("File uploaded, but analysis is not yet complete or failed.", icon="⌛")

    # --- THIS BLOCK IS NOW MOVED FROM GLOBAL SCOPE TO INSIDE THE FUNCTION ---
    # This helps reset the app if the file is changed
    if uploaded_file is None and 'last_file_name' in st.session_state:
        st.session_state.analysis_complete = False
        st.session_state.is_compatible = False
        st.session_state.df_raw = None
        st.session_state.normal_count = 0
        st.session_state.attack_count = 0
        st.session_state.missing_cols = []
        del st.session_state.last_file_name

# ===================================================================
# --- 5. Main App Controller ---
# ===================================================================
if st.session_state.logged_in:
    # If user is logged in, run the main dashboard function
    main_dashboard()
else:
    # If user is not logged in, show the login form
    login_form()

#python app.py
#python -m streamlit run app.py
#admin
#password123
    #python app.py
#python -m streamlit run app.py
#admin
#password123