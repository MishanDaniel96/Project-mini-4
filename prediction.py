import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Simplified Disease Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_data
def load_data():
    data = {}
    try:
        data['Liver'] = pd.read_pickle('liver_data_final.pkl')
        data['Parkinsons'] = pd.read_pickle('parkinsons.pkl')
        data['Kidney'] = pd.read_pickle('kidney_disease.pkl')
        
        data['Kidney'].columns = [col.strip().lower().replace(' ', '_').replace('?', '') for col in data['Kidney'].columns]
    except Exception as e:
        st.sidebar.error("🚨 Error loading one or more data files. Please check .pkl files.")
        st.stop() 
    return data

def mock_parkinsons_predict(fo_hz, jitter):
    score = 0.0
    if fo_hz < 130.0: score += 0.5
    if jitter > 0.00008: score += 0.5
    probability = min(0.9, max(0.1, score * 0.7 + 0.1))
    prediction = 1 if probability >= 0.5 else 0
    return prediction, probability

def mock_liver_predict(total_bilirubin, albumin):
    score = 0.0
    if total_bilirubin > 2.0: score += 0.6
    if albumin < 3.0: score += 0.4
    probability = min(0.95, max(0.05, score * 0.8))
    prediction = 1 if probability >= 0.5 else 2
    return prediction, probability

def mock_kidney_predict(bu, hemo):
    score = 0.0
    if bu > 60.0: score += 0.6
    if hemo < 10.0: score += 0.4
    probability = min(0.9, max(0.1, score * 0.8))
    prediction = 1 if probability >= 0.5 else 0
    return prediction, probability

def show_parkinsons_form(df):
    st.markdown("## 🗣️ Parkinson's Prediction")
    st.markdown("---")
    
 
    min_fo = df['MDVP:Fo(Hz)'].min()
    max_fo = df['MDVP:Fo(Hz)'].max()
    mean_fo = df['MDVP:Fo(Hz)'].mean()
    min_jitter = df['MDVP:Jitter(Abs)'].min()
    max_jitter = df['MDVP:Jitter(Abs)'].max()
    mean_jitter = df['MDVP:Jitter(Abs)'].mean()

    with st.form("parkinsons_form"):
        st.subheader("Vocal Metrics Input")
        fo_hz = st.slider('Avg Vocal Frequency (MDVP:Fo(Hz))', float(min_fo), float(max_fo), float(mean_fo), step=0.1)
        jitter_abs = st.slider('Absolute Jitter (MDVP:Jitter(Abs))', float(min_jitter), float(max_jitter), float(mean_jitter), step=0.000001, format="%.6e")

        st.markdown("---")
        submitted = st.form_submit_button('PREDICT STATUS', type="primary")

    if submitted:
        prediction, probability = mock_parkinsons_predict(fo_hz, jitter_abs)
        st.markdown("### Test Result:")
        
        col_res, col_score = st.columns(2)
        if prediction == 1:
            col_res.error(f"⚠️ HIGH RISK: Parkinson's Patient")
            col_score.metric("Risk Confidence", f"{probability*100:.1f}%", delta="Action Required")
        else:
            col_res.success(f"✅ LOW RISK: Healthy Subject")
            col_score.metric("Risk Confidence", f"{(1-probability)*100:.1f}%", delta="Stable")

def show_liver_form(df):
    st.markdown("## 🩸 Liver Prediction")
    st.info("Uses Total Bilirubin and Albumin levels to determine risk.")
    st.markdown("---")
    
    with st.form("liver_form"):
        st.subheader("Biomarkers Input")
        
        total_bilirubin = st.slider('Total Bilirubin', df['Total_Bilirubin'].min(), df['Total_Bilirubin'].max(), df['Total_Bilirubin'].mean(), step=0.1)
        albumin = st.slider('Albumin', df['Albumin'].min(), df['Albumin'].max(), df['Albumin'].mean(), step=0.1)


        st.markdown("---")
        submitted = st.form_submit_button('PREDICT STATUS', type="primary")

    if submitted:
        prediction, probability = mock_liver_predict(total_bilirubin, albumin)
        st.markdown("### Test Result:")
        
        col_res, col_score = st.columns(2)
        if prediction == 1:
            col_res.error(f"⚠️ HIGH RISK: Liver Patient")
            col_score.metric("Risk Confidence", f"{probability*100:.1f}%", delta="Action Required")
        else:
            col_res.success(f"✅ LOW RISK: Non-Liver Patient")
            col_score.metric("Risk Confidence", f"{(1-probability)*100:.1f}%", delta="Stable")

def show_kidney_form(df):
    st.markdown("## 💧 Kidney Prediction")
    st.markdown("---")
    
    with st.form("kidney_form"):
        st.subheader("Clinical Metrics Input")
        col1, col2 = st.columns(2)
        
        with col1:
            bu = st.slider('Blood Urea (bu)', df['bu'].min(), df['bu'].max(), df['bu'].mean(), step=1.0)
        
        with col2:
            hemo = st.slider('Hemoglobin (hemo)', df['hemo'].min(), df['hemo'].max(), df['hemo'].mean(), step=0.1)

        st.markdown("---")
        submitted = st.form_submit_button('PREDICT STATUS', type="primary")

    if submitted:
        prediction, probability = mock_kidney_predict(bu, hemo)
        st.markdown("### Test Result:")
        
        col_res, col_score = st.columns(2)
        if prediction == 1:
            col_res.error(f"⚠️ HIGH RISK: Chronic Kidney Disease")
            col_score.metric("Risk Confidence", f"{probability*100:.1f}%", delta="Action Required")
        else:
            col_res.success(f"✅ LOW RISK: No CKD Indicated")
            col_score.metric("Risk Confidence", f"{(1-probability)*100:.1f}%", delta="Stable")

    st.markdown("## 📚 Data Overview")
    st.markdown("---")
    
    dataset_name = st.selectbox("Select Dataset for Review:", list(data_frames.keys()))
    df = data_frames[dataset_name]

    col1, col2 = st.columns(2)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    
    st.markdown("---")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)

def main():
    data_frames = load_data()
    
    st.sidebar.title("Multiple Disease Prediction System 🏥")
    
    mode = st.sidebar.radio(
        "Select an Action:",
        ( 'Data Overview','Parkinsons Prediction', 'Liver Prediction', 'Kidney Prediction')
    )
    st.markdown("# Predictive Analytics Dashboard")
    st.markdown("---")

    if mode == 'Parkinsons Prediction':
        show_parkinsons_form(data_frames['Parkinsons'])
    
    elif mode == 'Liver Prediction':
        show_liver_form(data_frames['Liver'])

    elif mode == 'Kidney Prediction':
        show_kidney_form(data_frames['Kidney'])

    elif mode == 'Data Overview':
        show_data_exploration(data_frames)

if __name__ == "__main__":
    main()