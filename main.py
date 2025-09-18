import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import load_ground_motion_data, octave_spacing, ieee693_high_rrs_with_tolerances
import os

# --- Core Calculation Functions (Unchanged) ---
def calculate_response_spectrum(accel, dt, frequencies, damping=0.05):
    """
    Calculates the pseudo-acceleration response spectrum of a ground motion.
    Uses the Newmark-Beta method for linear systems.
    """
    sa = np.zeros(len(frequencies))
    g = 9.81 # Acceleration due to gravity (m/s^2)

    for i, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq
        m = 1.0
        k = omega**2 * m
        c = 2 * damping * omega * m
        gamma, beta = 0.5, 0.25
        u, v, a = 0.0, 0.0, 0.0
        u_max = 0.0
        p = -m * (accel * g)
        a1 = (m / (beta * dt**2)) + (c * gamma / (beta * dt))
        a2 = (m / (beta * dt)) + c * (gamma / beta - 1)
        a3 = m * (1 / (2 * beta) - 1) + c * dt * (gamma / (2 * beta) - 1)
        k_hat = k + a1
        for p_i in p:
            p_hat = p_i + a1 * u + a2 * v + a3 * a
            u_new = p_hat / k_hat
            delta_u = u_new - u
            v_new = v + (1 - gamma) * dt * a + gamma * dt * (delta_u / (beta * dt**2) - v / (beta * dt) - ((1 / (2 * beta)) - 1) * a)
            a_new = (delta_u / (beta * dt**2)) - v / (beta * dt) - ((1 / (2 * beta)) - 1) * a
            u, v, a = u_new, v_new, a_new
            if abs(u) > u_max:
                u_max = abs(u)
        sa[i] = u_max * omega**2 / g
    return sa

# --- NEW INTERACTIVE PLOTTING FUNCTIONS (using Plotly) ---
def _add_ieee_curves_to_fig(fig, ieee_data):
    """Helper function to add IEEE curves to a Plotly figure."""
    if ieee_data:
        f_ieee, sa_ieee, sa_upper, sa_lower = ieee_data
        fig.add_trace(go.Scatter(x=f_ieee, y=sa_ieee, mode='lines', name='IEEE 693 RRS', line=dict(color='black', dash='dash')))
        fig.add_trace(go.Scatter(x=f_ieee, y=sa_upper, mode='lines', name='+50% Tolerance', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=f_ieee, y=sa_lower, mode='lines', name='-10% Tolerance', line=dict(color='green', dash='dot')))

def plot_single_spectrum_interactive(frequencies, sa, label, ieee_data=None):
    """Creates an interactive Plotly chart for a single spectrum."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies, y=sa, mode='lines', name=label, line=dict(color='blue')))
    _add_ieee_curves_to_fig(fig, ieee_data)
    fig.update_layout(
        title='Response Spectrum',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Spectral Acceleration (g)',
        xaxis_type='log',
        legend_title='Trace'
    )
    return fig

def plot_comparison_spectra_interactive(frequencies, sa1, sa2, label1, label2, ieee_data=None):
    """Creates an interactive Plotly chart for comparing two spectra."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies, y=sa1, mode='lines', name=label1, line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=frequencies, y=sa2, mode='lines', name=label2, line=dict(color='green')))
    _add_ieee_curves_to_fig(fig, ieee_data)
    fig.update_layout(
        title='Response Spectra Comparison',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Spectral Acceleration (g)',
        xaxis_type='log',
        legend_title='Trace'
    )
    return fig


st.title('Response Spectra Analysis Tool')

# --- Sidebar (Unchanged) ---
with st.sidebar.expander("⚙️ Analysis & Plotting Options", expanded=True):
    damping = st.slider('Damping Ratio (%)', 1.0, 10.0, 5.0, 0.5) / 100.0
    st.caption("Frequency Range")
    col1, col2, col3 = st.columns(3)
    with col1:
        f_min = st.number_input('Min (Hz)', 0.1, 10.0, 0.2, key='f_min')
    with col2:
        f_max = st.number_input('Max (Hz)', 10.0, 100.0, 50.0, key='f_max')
    with col3:
        n_per_octave = st.number_input('Pts/Oct', 10, 50, 20, key='n_oct')
    st.caption("Reference Spectrum")
    plot_ieee = st.checkbox("Plot IEEE 693 Spectrum", key='plot_ieee')
    if plot_ieee:
        perf_level_option = st.selectbox("Performance Level", ["High", "Moderate"])
        perf_level = perf_level_option.lower()
        is_vertical = st.checkbox("Vertical Spectrum", key='is_vertical')

with st.sidebar.expander("📁 Upload Data", expanded=True):
    file1 = st.file_uploader("Upload First File", type=['txt', 'tim'], key="file1")
    col1, col2 = st.columns(2)
    with col1:
        rate1 = st.number_input('Rate (Hz)', min_value=1, value=512, key='rate1')
    with col2:
        channel1 = 0
        if file1 and file1.name.lower().endswith('.tim'):
            channel1 = st.number_input('Channel', min_value=0, step=1, value=0, key='ch1', help="0-based index")
    file2 = st.file_uploader("Upload Second File", type=['txt', 'tim'], key="file2")
    col1, col2 = st.columns(2)
    with col1:
        rate2 = st.number_input('Rate (Hz)', min_value=1, value=256, key='rate2')
    with col2:
        channel2 = 0
        if file2 and file2.name.lower().endswith('.tim'):
            channel2 = st.number_input('Channel', min_value=0, step=1, value=0, key='ch2', help="0-based index")

# --- Main Logic (Updated to call Plotly functions) ---
if st.button('Calculate and Plot Spectra'):
    if not file1 and not file2:
        st.warning('Please upload at least one file to analyze.')
    else:
        try:
            accel1, accel2, sa1, sa2 = None, None, None, None
            frequencies = octave_spacing(f_min, f_max, n_per_octave)
            ieee_data_to_plot = None
            if plot_ieee:
                damping_percent = damping * 100
                f_ieee, sa_ieee, sa_upper, sa_lower = ieee693_high_rrs_with_tolerances(
                    damping_percent=damping_percent, performance_level=perf_level, vertical=is_vertical)
                ieee_data_to_plot = (f_ieee, sa_ieee, sa_upper, sa_lower)

            with st.spinner('Calculating...'):
                if file1:
                    with open(file1.name, "wb") as f: f.write(file1.getbuffer())
                    accel1 = load_ground_motion_data(file1.name, channel_index=channel1)
                    os.remove(file1.name)
                    sa1 = calculate_response_spectrum(accel1, 1.0/rate1, frequencies, damping)
                if file2:
                    with open(file2.name, "wb") as f: f.write(file2.getbuffer())
                    accel2 = load_ground_motion_data(file2.name, channel_index=channel2)
                    os.remove(file2.name)
                    sa2 = calculate_response_spectrum(accel2, 1.0/rate2, frequencies, damping)

            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            if accel1 is not None:
                with col1:
                    st.metric(label=f"PGA (File 1: {file1.name})", value=f"{np.max(np.abs(accel1)):.4f} g")
            if accel2 is not None:
                with col2:
                    st.metric(label=f"PGA (File 2: {file2.name})", value=f"{np.max(np.abs(accel2)):.4f} g")

            # --- Plotting logic now uses st.plotly_chart ---
            if sa1 is not None and sa2 is not None:
                fig = plot_comparison_spectra_interactive(frequencies, sa1, sa2, f'{file1.name}', f'{file2.name}', ieee_data=ieee_data_to_plot)
                st.plotly_chart(fig, use_container_width=True)
            elif sa1 is not None:
                fig = plot_single_spectrum_interactive(frequencies, sa1, f'{file1.name}', ieee_data=ieee_data_to_plot)
                st.plotly_chart(fig, use_container_width=True)
            elif sa2 is not None:
                fig = plot_single_spectrum_interactive(frequencies, sa2, f'{file2.name}', ieee_data=ieee_data_to_plot)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure the channel index and sampling rates are correct for the uploaded files.")