"""Streamlit Frontend - VWL Simulation Interface

Zustandslos: Jeder Request ist unabhängig, alle Daten kommen vom Backend.
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import os
from typing import Dict, List

# === CONFIGURATION ===

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://vwl-rl-backend-698656921826.europe-west1.run.app"
)

# === PAGE CONFIG ===

st.set_page_config(
    page_title="VWL-RL Simulation",
    page_icon="🏬",
    layout="wide"
)

# === HELPER FUNCTIONS ===

def check_backend_health() -> Dict:
    """Prüfe Backend Verfügbarkeit"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "unreachable", "env_available": False, "model_loaded": False}
    except Exception as e:
        return {"status": "error", "env_available": False, "model_loaded": False, "error": str(e)}


def run_simulation(params: Dict) -> Dict:
    """Führe Simulation im Backend aus"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/simulate",
            json=params,
            timeout=60
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"Backend Error {response.status_code}: {response.text}"
            }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Backend Timeout (>60s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# === UI LAYOUT ===

st.title("🏬 Volkswirtschafts-Simulation mit RL")
st.markdown("**Cloud-basiertes RL-System:** Frontend (zustandslos) + Backend (zustandsbehaftet)")

st.divider()

# Backend Status
health = check_backend_health()

if health["status"] == "healthy":
    st.success(f"✅ Backend erreichbar | Environment: {'Ja' if health['env_available'] else 'Nein'} | RL-Model: {'Geladen' if health['model_loaded'] else 'Nicht geladen'}")
else:
    st.error(f"❌ Backend nicht erreichbar: {health.get('error', 'Unknown')}")
    st.warning("Simulation nicht möglich ohne Backend-Verbindung")
    st.stop()

# === SIDEBAR: PARAMETERS ===

st.sidebar.header("⚙️ Simulations-Parameter")

# Environment (aktuell nur eines verfügbar)
environment = st.sidebar.selectbox(
    "Environment",
    ["FullEconomy-v0"],
    help="Wirtschafts-Environment für Simulation"
)

# Szenario
scenario = st.sidebar.selectbox(
    "Szenario",
    ["Normal", "Rezession", "Boom", "Inflation"],
    help="Wirtschaftliches Start-Szenario"
)

# Anzahl Steps
num_steps = st.sidebar.number_input(
    "Simulations-Schritte",
    min_value=10,
    max_value=200,
    value=100,
    step=10,
    help="Anzahl der Tage (Steps) für die Simulation"
)

st.sidebar.divider()

# RL-Agent Toggle
use_rl = st.sidebar.checkbox(
    "🧠 RL-Agent nutzen",
    value=False,
    help="Wenn aktiv: RL-Agent steuert Wirtschaftspolitik. Sonst: Manuelle Parameter."
)

if not use_rl:
    st.sidebar.subheader("🔧 Manuelle Parameter")
    
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
        help="Tägliche Staatsausgaben in Euro"
    )
    
    interest_rate = st.sidebar.slider(
        "Zinssatz",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Zentralbank-Zinssatz"
    )
    
    manual_params = {
        "tax_rate": tax_rate,
        "gov_spending": gov_spending,
        "interest_rate": interest_rate
    }
else:
    manual_params = None
    st.sidebar.info("🧠 RL-Agent übernimmt Steuerung")

# === MAIN AREA ===

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Simulation starten")
    
    if st.button("▶️ Simulieren", type="primary", use_container_width=True):
        
        # Simulation Request vorbereiten
        sim_params = {
            "environment": environment,
            "num_steps": num_steps,
            "scenario": scenario,
            "use_rl_agent": use_rl,
            "manual_params": manual_params
        }
        
        # Simulation ausführen
        with st.spinner("🔄 Simulation läuft im Backend..."):
            result = run_simulation(sim_params)
        
        if result["success"]:
            st.success("✅ Simulation abgeschlossen!")
            st.session_state['sim_result'] = result["data"]
        else:
            st.error(f"❌ Simulation fehlgeschlagen: {result['error']}")

with col2:
    st.subheader("📊 Aktuelle Parameter")
    if not use_rl:
        st.metric("Steuersatz", f"{manual_params['tax_rate']:.1%}")
        st.metric("Staatsausgaben", f"{manual_params['gov_spending']:.0f}€")
        st.metric("Zinssatz", f"{manual_params['interest_rate']:.1%}")
    else:
        st.info("🧠 RL-Agent aktiv")
    st.metric("Szenario", scenario)
    st.metric("Steps", num_steps)

# === RESULTS VISUALIZATION ===

if 'sim_result' in st.session_state:
    st.divider()
    st.subheader("📉 Simulations-Ergebnisse")
    
    data = st.session_state['sim_result']
    steps = data['steps']
    summary = data['summary']
    
    # Daten extrahieren
    step_numbers = [s['step'] for s in steps]
    bip_values = [s['bip'] for s in steps]
    inflation_values = [s['inflation'] * 100 for s in steps]  # in %
    unemployment_values = [s['unemployment'] * 100 for s in steps]  # in %
    debt_values = [s['debt'] for s in steps]
    
    # BIP Chart
    fig_bip = go.Figure()
    fig_bip.add_trace(go.Scatter(
        x=step_numbers,
        y=bip_values,
        mode='lines',
        name='BIP',
        line=dict(color='#1f77b4', width=2)
    ))
    fig_bip.update_layout(
        title="Bruttoinlandsprodukt (€)",
        xaxis_title="Zeitschritt (Tag)",
        yaxis_title="BIP",
        height=300
    )
    st.plotly_chart(fig_bip, use_container_width=True)
    
    # Inflation + Arbeitslosigkeit
    col1, col2 = st.columns(2)
    
    with col1:
        fig_infl = go.Figure()
        fig_infl.add_trace(go.Scatter(
            x=step_numbers,
            y=inflation_values,
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
    
    with col2:
        fig_unemp = go.Figure()
        fig_unemp.add_trace(go.Scatter(
            x=step_numbers,
            y=unemployment_values,
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
    
    # Summary Stats
    st.divider()
    st.subheader("📊 Zusammenfassung")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "End-BIP",
            f"{summary['final_bip']:.0f}€",
            delta=f"{summary['bip_growth']:.1f}%"
        )
    
    with col2:
        st.metric(
            "BIP-Wachstum",
            f"{summary['bip_growth']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Durchschn. Inflation",
            f"{summary['avg_inflation']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Durchschn. Arbeitslosigkeit",
            f"{summary['avg_unemployment']:.1f}%"
        )

# === FOOTER ===

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    🎓 DHSH - Fortgeschrittene KI-Anwendungen & Cloud & Big Data<br>
    Reinforcement Learning + Cloud Architecture | Januar 2026<br>
    Backend: <code>""" + BACKEND_URL + """</code>
    </div>
    """,
    unsafe_allow_html=True
)
