import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model & features (refined version)
@st.cache_resource
def load_model():
    with open('rf_model_refined.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('features_refined.txt', 'r') as f:
        features = eval(f.read())
    return model, features

model, features = load_model()

st.title("🚗 CNG PdM App: Predict Engine Faults")
st.write("Input symptoms/driving data → Get maintenance alert (RF Model, ~80% F1 from 129 driver responses)")

# Edit 1: More Inputs (Now 6 symptoms in columns + other causal features)
col1, col2 = st.columns(2)
with col1:
    has_overheat = st.slider("Overheat Symptom? (0=No, 1=Yes)", 0, 1, 0)
    has_knock = st.slider("Knock Symptom? (0=No, 1=Yes)", 0, 1, 0)
    has_shutdown = st.slider("Shutdown Symptom? (0=No, 1=Yes)", 0, 1, 0)
    has_emission_smell = st.slider("Emission Smell Symptom? (0=No, 1=Yes)", 0, 1, 0)  # New: Injector fouling
    has_poor_accel = st.slider("Poor Acceleration Symptom? (0=No, 1=Yes)", 0, 1, 0)  # New: Power loss in retrofits
    has_difficult_start = st.slider("Difficult Start Symptom? (0=No, 1=Yes)", 0, 1, 0)  # New: Ignition issues
with col2:
    symptom_freq = st.selectbox("Symptom Frequency", [0,1,2,3], help="0=Rarely, 3=Frequently")  # Encoded
    maint_freq = st.selectbox("Maint Freq (km)", [0,1,2,3], help="0=1000km, 3=Fault only")  # Encoded
terrain_options = {0: "City roads", 1: "Rough terrain", 2: "Highways", 3: "Hilly terrain", 4: "City & hilly"}
terrain = st.selectbox("Terrain", list(terrain_options.keys()), format_func=lambda x: f"{x} - {terrain_options[x]}")
idling = st.slider("Frequent Idling? (0=No, 1=Yes)", 0, 1, 0)
used = st.slider("Used Conversion? (0=No, 1=Yes)", 0, 1, 0)

if st.button("Predict Risk"):
    # Build input (defaults from data means)
    inputs = {feat: np.mean(model.feature_importances_) for feat in features}  # Avg defaults
    inputs['has_overheat'] = has_overheat
    inputs['has_knock'] = has_knock
    inputs['has_shutdown'] = has_shutdown
    inputs['has_emission_smell'] = has_emission_smell  # New
    inputs['has_poor_accel'] = has_poor_accel  # New
    inputs['has_difficult_start'] = has_difficult_start  # New
    inputs['Symptom Freq'] = symptom_freq
    inputs['Maint Freq (km)'] = maint_freq
    inputs['Terrain'] = terrain
    inputs['Idling'] = idling
    inputs['Used?'] = used
    input_df = pd.DataFrame([inputs], columns=features)
    
    prob = model.predict_proba(input_df)[0][1]
    st.metric("Fault Risk", f"{prob:.1%}")
    
    # Edit 2: Alternate Alert (Based on prominent symptoms—check top 6, pick 1-2 for rec)
    prominent_symptoms = []
    if has_overheat: prominent_symptoms.append("overheat")
    if has_knock: prominent_symptoms.append("knock")
    if has_shutdown: prominent_symptoms.append("shutdown")
    if has_emission_smell: prominent_symptoms.append("emission smell")
    if has_poor_accel: prominent_symptoms.append("poor acceleration")
    if has_difficult_start: prominent_symptoms.append("difficult start")
    
    if prob > 0.5:
        if prominent_symptoms:
            # Tailored rec based on 1-2 prominent (CNG-specific)
            if "overheat" in prominent_symptoms and "knock" in prominent_symptoms:
                rec = "Check valves and cooling system for thermal wear."
            elif "emission smell" in prominent_symptoms or "poor acceleration" in prominent_symptoms:
                rec = "Inspect injectors/regulators for fouling or leaks."
            elif "difficult start" in prominent_symptoms or "shutdown" in prominent_symptoms:
                rec = "Examine spark plugs/ignition for dry gas issues."
            else:
                rec = "Schedule general CNG retrofit inspection."
            st.error(f"🚨 High Risk: {rec} in <1000km.")
        else:
            st.error("🚨 High Risk: Schedule general valve/spark check in <1000km (CNG wear).")
    else:
        st.success("✅ Low Risk: Monitor weekly.")

st.caption("Prototype for low-resource CNG fleets – Open-source via Streamlit.")