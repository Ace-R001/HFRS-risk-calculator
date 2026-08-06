#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Severe HFRS Risk Prediction Web Tool (第二次分析定稿版)
=========================================================
基于第二次分析的最终随机森林模型 (9 因子, θ = 0.6119)。
风格参考第一次分析部署, 实际模型/结果以本次为准。

Usage: streamlit run app.py
"""
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ Load model (cached) ============
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'final_model.pkl')


@st.cache_resource(show_spinner='Loading model...')
def load_model():
    return pickle.load(open(MODEL_PATH, 'rb'))


@st.cache_resource(show_spinner='Preparing explainer...')
def load_explainer(model):
    return shap.TreeExplainer(model)


fm = load_model()
model = fm['model']
threshold = fm['threshold']

FEATURES = ['CK', 'CK_MB', 'CR', 'DD', 'Hypotension', 'MYO', 'NEU%', 'NPR', 'Urine_output_24h']
# 训练范围 (development set, n=160 实际 min-max), 用于外推警告
TRAIN_RANGE = {
    'CK': (27.6, 3928.0), 'CK_MB': (3.2, 235.2), 'CR': (48.2, 1223.9),
    'DD': (452.0, 22069.0), 'Hypotension': (0, 1), 'MYO': (12.1, 1565.8),
    'NEU%': (12.8, 90.1), 'NPR': (0.0, 3.6), 'Urine_output_24h': (10.0, 7430.0),
}
FEAT_NAMES = {
    'CK': 'CK (U/L)', 'CK_MB': 'CK-MB (U/L)', 'CR': 'CR (\u03bcmol/L)',
    'DD': 'D-dimer (ng/mL)', 'Hypotension': 'Hypotension',
    'MYO': 'MYO (ng/mL)', 'NEU%': 'NEU% (%)', 'NPR': 'NPR',
    'Urine_output_24h': '24-h urine output (mL)',
}
display_names = [FEAT_NAMES[f] for f in FEATURES]

# ============ Page config ============
st.set_page_config(page_title='Severe HFRS Risk Prediction', page_icon='🩸', layout='wide')

st.markdown("""
    <style>
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            font-size: 17px !important;
        }
        .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
            font-size: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center;">Severe HFRS Risk Calculator for Patients with Hantavirus Infection</h1>', unsafe_allow_html=True)

# ============ Input section ============
st.header('Patient Inputs')

col1, col2, col3 = st.columns(3)
with col1:
    ck = st.number_input('CK (U/L)', min_value=0.0, max_value=5000.0, value=146.1, step=5.0)
    ck_mb = st.number_input('CK-MB (U/L)', min_value=0.0, max_value=500.0, value=37.5, step=1.0)
    cr = st.number_input('CR (\u03bcmol/L)', min_value=20.0, max_value=1500.0, value=223.7, step=5.0)
with col2:
    dd = st.number_input('D-dimer (ng/mL)', min_value=0.0, max_value=25000.0, value=2000.7, step=10.0)
    hypotension = st.selectbox('Hypotension', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    myo = st.number_input('MYO (ng/mL)', min_value=0.0, max_value=5000.0, value=88.7, step=5.0)
with col3:
    neu_pct = st.number_input('NEU% (%)', min_value=10.0, max_value=100.0, value=63.8, step=0.5)
    npr = st.number_input('NPR', min_value=0.0, max_value=20.0, value=0.3, step=0.1,
                          help='Neutrophil-to-platelet ratio = NEU (×10⁹/L) / PLT (×10⁹/L)')
    urine = st.number_input('24-h urine output (mL)', min_value=0.0, max_value=8000.0, value=570.0, step=50.0)

feature_values = [ck, ck_mb, cr, dd, hypotension, myo, neu_pct, npr, urine]

# ============ 训练范围外推警告 ============
out_of_range = []
for f, v in zip(FEATURES, feature_values):
    lo, hi = TRAIN_RANGE[f]
    if v < lo or v > hi:
        out_of_range.append(f'{FEAT_NAMES[f]} ({v:.1f}; training range {lo:.1f}\u2013{hi:.1f})')
if out_of_range:
    st.warning('**Input outside the training range:** ' + '; '.join(out_of_range) +
               '. The prediction is extrapolated beyond the development cohort and should be interpreted with caution.')

# ============ Prediction ============
if st.button('Predict', type='primary'):
    features = np.array([feature_values])
    features_df = pd.DataFrame(features, columns=FEATURES)

    predicted_class = int(model.predict(features)[0])
    prob_severe = model.predict_proba(features)[0, 1]

    # ============ Results display ============
    st.header('Prediction Result')

    c1, c2 = st.columns(2)
    with c1:
        st.metric('Predicted Class',
                  'Severe HFRS' if predicted_class == 1 else 'Non-severe HFRS',
                  delta='High risk' if predicted_class == 1 else 'Low risk')
    with c2:
        st.metric('Probability of Severe HFRS', f'{prob_severe*100:.1f}%')

    st.progress(min(prob_severe, 1.0))
    st.caption(f'Predicted probability of severe HFRS: {prob_severe*100:.1f}% | '
               f'Fixed threshold \u03b8 = {threshold:.4f} (Youden index)')

    if predicted_class == 1:
        st.error(
            f'**High risk:** According to the model, this patient has a {prob_severe*100:.1f}% '
            'probability of developing severe HFRS. Intensive monitoring of hemodynamics, '
            'renal function, coagulation, and fluid balance is recommended. Early transfer '
            'to intensive care should be considered.'
        )
    else:
        st.success(
            f'**Low risk:** According to the model, this patient has a {(1-prob_severe)*100:.1f}% '
            'probability of non-severe HFRS. Standard monitoring of renal function and '
            'platelet count is recommended.'
        )

    # ============ SHAP Force Plot ============
    st.subheader('SHAP Force Plot Explanation')
    explainer_shap = load_explainer(model)
    shap_values = explainer_shap.shap_values(features_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    base_val = explainer_shap.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    plt.figure(figsize=(15, 6))
    shap.force_plot(base_val, shap_values[0], features_df.iloc[0],
                    feature_names=display_names, matplotlib=True,
                    show=False, text_rotation=0)
    plt.gcf().set_facecolor('white')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    st.pyplot(plt.gcf())
    plt.close()
    st.caption(
        f'Base value: {base_val:.3f} | Predicted probability f(x): {prob_severe:.3f}. '
        'Red segments push the prediction toward severe HFRS; blue segments toward non-severe.'
    )

# ============ Footer ============
st.markdown('---')
st.markdown(
    '<p style="text-align: center; font-size: 14px;">'
    '<b>Disclaimer:</b> This tool uses a random forest model and nine routinely available clinical '
    'variables at admission to provide an individualized estimate of severe HFRS risk for early '
    'clinical decision support; it is not intended to replace professional medical judgment.</p>',
    unsafe_allow_html=True
)
