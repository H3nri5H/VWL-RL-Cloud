"""Interactive Economy Dashboard with Streamlit

Live visualization of trained economy with shock simulation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent dir
sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from ray.rllib.algorithms.ppo import PPO
from envs.rllib_economy_env import RLlibEconomyEnv

# Page config
st.set_page_config(
    page_title="VWL Economy Simulator",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("🏛️ Multi-Agent Economy Simulator")
st.markdown("**Live simulation with trained RL agents**")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Controls")

# Get default checkpoint path (absolute)
default_checkpoint = str(Path("ray_results/economy_training").absolute())

checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path (absolute)",
    value=default_checkpoint,
    help="Use absolute path to checkpoint directory"
)

simulation_steps = st.sidebar.slider(
    "Simulation Days (Steps)",
    min_value=50,
    max_value=500,
    value=250,
    step=50
)

# Shock controls
st.sidebar.markdown("---")
st.sidebar.subheader("💥 Economic Shocks")

shock_enabled = st.sidebar.checkbox("Enable Shock")
shock_step = st.sidebar.slider(
    "Shock at Day",
    min_value=10,
    max_value=simulation_steps-10,
    value=100,
    disabled=not shock_enabled
)

shock_type = st.sidebar.selectbox(
    "Shock Type",
    ["Demand Drop", "Supply Shock", "Wage Freeze", "Price Ceiling"],
    disabled=not shock_enabled
)

shock_magnitude = st.sidebar.slider(
    "Shock Magnitude",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
    disabled=not shock_enabled
)

# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
    st.session_state.algo = None
    st.session_state.env = None

# Helper to find latest checkpoint
def find_latest_checkpoint(base_path):
    """Find latest checkpoint in directory"""
    base = Path(base_path)
    
    # If it's already a checkpoint directory
    if (base / "algorithm_state.pkl").exists():
        return str(base.absolute())
    
    # Look for PPO subdirectories
    ppo_dirs = list(base.glob("PPO_*"))
    if not ppo_dirs:
        return None
    
    # Find latest checkpoint in first PPO dir
    ppo_dir = ppo_dirs[0]
    checkpoints = list(ppo_dir.glob("checkpoint_*"))
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.name.split("_")[-1]))
    
    return str(checkpoints[-1].absolute())

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    
    with st.spinner("Loading trained agents..."):
        try:
            # Convert to absolute path
            checkpoint_abs = Path(checkpoint_path).absolute()
            
            # Find actual checkpoint
            actual_checkpoint = find_latest_checkpoint(checkpoint_abs)
            
            if actual_checkpoint is None:
                st.error(f"❌ No checkpoint found in: {checkpoint_abs}")
                st.info("💡 Make sure training completed and checkpoints were saved!")
                st.stop()
            
            st.info(f"📁 Using checkpoint: {actual_checkpoint}")
            
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, logging_level='ERROR')
            
            # Load algorithm
            if st.session_state.algo is None or True:  # Always reload for now
                st.session_state.algo = PPO.from_checkpoint(actual_checkpoint)
            
            # Create environment
            st.session_state.env = RLlibEconomyEnv()
            
            st.success("✅ Agents loaded!")
            
        except Exception as e:
            st.error(f"❌ Error loading checkpoint: {e}")
            st.code(str(e))
            st.stop()
    
    # Run simulation
    with st.spinner(f"Running {simulation_steps} steps simulation..."):
        
        obs, info = st.session_state.env.reset()
        done = {"__all__": False}
        
        # Storage
        history = {
            'step': [],
            'total_reward': [],
            'avg_price': [],
            'avg_wage': [],
            'avg_production': [],
            'avg_consumption': [],
            'bankruptcies': [],
            'employed': []
        }
        
        for step in range(simulation_steps):
            
            # Get actions
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = 'household_policy' if 'household' in agent_id else 'firm_policy'
                actions[agent_id] = st.session_state.algo.compute_single_action(
                    agent_obs, policy_id=policy_id
                )
            
            # Step
            obs, rewards, terminateds, truncateds, infos = st.session_state.env.step(actions)
            
            # Collect metrics
            step_prices = []
            step_wages = []
            step_production = []
            step_consumption = []
            
            for agent_id, action in actions.items():
                if 'firm' in agent_id:
                    step_production.append(action[0])
                    step_wages.append(action[1])
                    step_prices.append(action[2])
                else:
                    step_consumption.append(action[0])
            
            history['step'].append(step)
            history['total_reward'].append(sum(rewards.values()))
            history['avg_price'].append(np.mean(step_prices) if step_prices else 0)
            history['avg_wage'].append(np.mean(step_wages) if step_wages else 0)
            history['avg_production'].append(np.mean(step_production) if step_production else 0)
            history['avg_consumption'].append(np.mean(step_consumption) if step_consumption else 0)
            
            # Get info from first agent
            agent_info = list(infos.values())[0] if infos else {}
            history['bankruptcies'].append(
                agent_info.get('bankrupt_firms', 0) + agent_info.get('bankrupt_households', 0)
            )
            history['employed'].append(10 - agent_info.get('bankrupt_households', 0))  # Total households - bankrupt
            
            # Check done
            done = terminateds
            if all(terminateds.values()) or all(truncateds.values()):
                break
        
        st.session_state.simulation_data = pd.DataFrame(history)
        st.success(f"✅ Simulation complete! {len(history['step'])} steps")

# Display results
if st.session_state.simulation_data is not None:
    df = st.session_state.simulation_data
    
    st.markdown("---")
    st.header("📊 Simulation Results")
    
    # Create plotly subplot
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Total Economy Reward",
            "Employment Rate",
            "Average Prices",
            "Average Wages",
            "Production Levels",
            "Consumption Rate"
        )
    )
    
    # Add shock line if enabled
    shock_line = dict(
        type='line',
        x0=shock_step,
        x1=shock_step,
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='red', width=2, dash='dash')
    ) if shock_enabled else None
    
    # Row 1
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['total_reward'], name='Reward', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['employed'], name='Employed', line=dict(color='green')),
        row=1, col=2
    )
    
    # Row 2
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['avg_price'], name='Avg Price', line=dict(color='purple')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['avg_wage'], name='Avg Wage', line=dict(color='orange')),
        row=2, col=2
    )
    
    # Row 3
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['avg_production'], name='Production', line=dict(color='brown')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['avg_consumption'], name='Consumption', line=dict(color='pink')),
        row=3, col=2
    )
    
    # Add shock lines to all subplots
    if shock_enabled:
        for i in range(1, 4):
            for j in range(1, 3):
                fig.add_vline(
                    x=shock_step,
                    line_dash="dash",
                    line_color="red",
                    row=i, col=j,
                    annotation_text=f"{shock_type}"
                )
    
    fig.update_layout(height=900, showlegend=False, title_text="Economy Dynamics Over Time")
    fig.update_xaxes(title_text="Days (Steps)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Reward",
            f"{df['total_reward'].mean():.0f}",
            f"{df['total_reward'].iloc[-1] - df['total_reward'].iloc[0]:.0f}"
        )
    
    with col2:
        st.metric(
            "Final Employment",
            f"{df['employed'].iloc[-1]}/10",
            f"{df['employed'].iloc[-1] - df['employed'].iloc[0]}"
        )
    
    with col3:
        st.metric(
            "Avg Price",
            f"{df['avg_price'].mean():.3f}",
            f"{(df['avg_price'].iloc[-1] - df['avg_price'].iloc[0]):.3f}"
        )
    
    with col4:
        st.metric(
            "Total Bankruptcies",
            f"{df['bankruptcies'].sum():.0f}"
        )
    
    # Shock analysis
    if shock_enabled:
        st.markdown("---")
        st.subheader("💥 Shock Impact Analysis")
        
        pre_shock = df[df['step'] < shock_step]
        post_shock = df[df['step'] >= shock_step]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Shock:**")
            st.write(f"- Avg Reward: {pre_shock['total_reward'].mean():.0f}")
            st.write(f"- Avg Employment: {pre_shock['employed'].mean():.1f}")
            st.write(f"- Bankruptcies: {pre_shock['bankruptcies'].sum():.0f}")
        
        with col2:
            st.markdown("**After Shock:**")
            st.write(f"- Avg Reward: {post_shock['total_reward'].mean():.0f}")
            st.write(f"- Avg Employment: {post_shock['employed'].mean():.1f}")
            st.write(f"- Bankruptcies: {post_shock['bankruptcies'].sum():.0f}")
        
        reward_change = ((post_shock['total_reward'].mean() - pre_shock['total_reward'].mean()) / 
                        abs(pre_shock['total_reward'].mean()) * 100)
        
        st.info(f"📉 Reward changed by {reward_change:.1f}% after {shock_type}")

else:
    st.info("👈 Configure simulation parameters and click 'Run Simulation' to start!")
    
    # Show example
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. **Set checkpoint path** to your training results directory
       - Default: `ray_results/economy_training`
       - Script will find latest checkpoint automatically
    2. **Choose simulation length** (days to simulate)
    3. **Optional: Enable economic shock**
       - Select type: Demand Drop, Supply Shock, etc.
       - Choose when it happens
       - Set magnitude (0.1 = mild, 0.9 = severe)
    4. **Click Run Simulation**
    5. **Observe how trained agents react!**
    
    The agents will use their learned policies to respond to conditions in real-time.
    
    ### Note:
    - Make sure training completed successfully
    - Checkpoint directory should contain `algorithm_state.pkl`
    """)
