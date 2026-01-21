"""Streamlit Frontend (Zustandslos) - VWL Simulation Interface"""
import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import json

# Page Config
st.set_page_config(
    page_title="VWL-RL Simulation",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Volkswirtschafts-Simulation mit RL")
st.markdown("**Multi-Agent RL System:** 10 Firmen + 50 Haushalte + 1 Regierung")

st.divider()

# Sidebar: Simulation Parameters
st.sidebar.header("⚙️ Simulations-Parameter")

tax_rate = st.sidebar.slider(
    "Steuersatz",
    min_value=0.0,
    max_value=0.5,
    value=0.3,
    step=0.05,
    help="Einkommenssteuersatz (0% - 50%)"
)

gov_spending = st.sidebar.slider(
    "Staatsausgaben",
    min_value=0.0,
    max_value=1000.0,
    value=500.0,
    step=50.0,
    help="Monatliche Staatsausgaben"
)

interest_rate = st.sidebar.slider(
    "Zinssatz",
    min_value=0.0,
    max_value=0.2,
    value=0.05,
    step=0.01,
    help="Zentralbank-Zinssatz"
)

scenario = st.sidebar.selectbox(
    "Szenario",
    ["Normal", "Rezession", "Boom", "Inflation"],
    help="Wähle ein wirtschaftliches Szenario"
)

num_steps = st.sidebar.number_input(
    "Simulations-Schritte",
    min_value=10,
    max_value=200,
    value=100,
    step=10
)

st.sidebar.divider()

use_rl = st.sidebar.checkbox(
    "🧠 RL-Agent nutzen",
    value=False,
    help="Wenn aktiv: RL-Agent steuert Wirtschaftspolitik. Sonst: Manuelle Parameter."
)

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Simulation starten")
    
    if st.button("▶️ Simulieren", type="primary", use_container_width=True):
        with st.spinner("Simulation läuft..."):
            # Mock Simulation (bis Backend bereit)
            # TODO: Replace mit Backend API Call
            # response = requests.post("http://backend:8000/simulate", json={...})
            
            # Mock Data
            steps = np.arange(num_steps)
            
            # Szenario-abhängige Simulation
            if scenario == "Rezession":
                bip = 1000 - steps * 5 + np.random.normal(0, 20, num_steps)
                unemployment = 0.05 + steps * 0.003 + np.random.normal(0, 0.01, num_steps)
                inflation = 0.02 - steps * 0.0001 + np.random.normal(0, 0.005, num_steps)
            elif scenario == "Boom":
                bip = 1000 + steps * 10 + np.random.normal(0, 30, num_steps)
                unemployment = 0.05 - steps * 0.0002 + np.random.normal(0, 0.005, num_steps)
                inflation = 0.02 + steps * 0.0005 + np.random.normal(0, 0.01, num_steps)
            elif scenario == "Inflation":
                bip = 1000 + steps * 2 + np.random.normal(0, 25, num_steps)
                unemployment = 0.05 + np.random.normal(0, 0.01, num_steps)
                inflation = 0.02 + steps * 0.002 + np.random.normal(0, 0.015, num_steps)
            else:  # Normal
                bip = 1000 + steps * 3 + np.random.normal(0, 20, num_steps)
                unemployment = 0.05 + np.random.normal(0, 0.01, num_steps)
                inflation = 0.02 + np.random.normal(0, 0.005, num_steps)
            
            bip = np.clip(bip, 100, 5000)
            unemployment = np.clip(unemployment, 0, 0.5)
            inflation = np.clip(inflation, -0.1, 0.3)
            
            # Store in session state
            st.session_state['sim_data'] = {
                'steps': steps,
                'bip': bip,
                'unemployment': unemployment,
                'inflation': inflation
            }
            
        st.success("✅ Simulation abgeschlossen!")

with col2:
    st.subheader("📊 Aktuelle Parameter")
    st.metric("Steuersatz", f"{tax_rate:.1%}")
    st.metric("Staatsausgaben", f"{gov_spending:.0f}€")
    st.metric("Zinssatz", f"{interest_rate:.1%}")
    st.metric("Szenario", scenario)

# Results Visualization
if 'sim_data' in st.session_state:
    st.divider()
    st.subheader("📉 Simulations-Ergebnisse")
    
    data = st.session_state['sim_data']
    
    # BIP Chart
    fig_bip = go.Figure()
    fig_bip.add_trace(go.Scatter(
        x=data['steps'],
        y=data['bip'],
        mode='lines',
        name='BIP',
        line=dict(color='#1f77b4', width=2)
    ))
    fig_bip.update_layout(
        title="Bruttoinlandsprodukt (€)",
        xaxis_title="Zeitschritt",
        yaxis_title="BIP",
        height=300
    )
    st.plotly_chart(fig_bip, use_container_width=True)
    
    # Arbeitslosigkeit + Inflation
    col1, col2 = st.columns(2)
    
    with col1:
        fig_unemp = go.Figure()
        fig_unemp.add_trace(go.Scatter(
            x=data['steps'],
            y=data['unemployment'] * 100,
            mode='lines',
            name='Arbeitslosigkeit',
            line=dict(color='#d62728', width=2)
        ))
        fig_unemp.update_layout(
            title="Arbeitslosenquote (%)",
            xaxis_title="Zeitschritt",
            yaxis_title="Prozent",
            height=300
        )
        st.plotly_chart(fig_unemp, use_container_width=True)
    
    with col2:
        fig_infl = go.Figure()
        fig_infl.add_trace(go.Scatter(
            x=data['steps'],
            y=data['inflation'] * 100,
            mode='lines',
            name='Inflation',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig_infl.update_layout(
            title="Inflationsrate (%)",
            xaxis_title="Zeitschritt",
            yaxis_title="Prozent",
            height=300
        )
        st.plotly_chart(fig_infl, use_container_width=True)
    
    # Summary Stats
    st.divider()
    st.subheader("📊 Zusammenfassung")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "End-BIP",
            f"{data['bip'][-1]:.0f}€",
            delta=f"{data['bip'][-1] - data['bip'][0]:.0f}€"
        )
    
    with col2:
        st.metric(
            "BIP-Wachstum",
            f"{((data['bip'][-1] / data['bip'][0]) - 1) * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Durchschn. Arbeitslosigkeit",
            f"{np.mean(data['unemployment']) * 100:.1f}%"
        )
    
    with col4:
        st.metric(
            "Durchschn. Inflation",
            f"{np.mean(data['inflation']) * 100:.1f}%"
        )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    🎓 DHSH - Fortgeschrittene KI-Anwendungen & Cloud & Big Data<br>
    Multi-Agent RL mit Ray RLlib | Januar 2026
    </div>
    """,
    unsafe_allow_html=True
)
