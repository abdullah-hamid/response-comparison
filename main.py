import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import load_ground_motion_data, octave_spacing
import os


def calculate_response_spectrum(accel, dt, frequencies, damping=0.05):
    """
    Calculates the pseudo-acceleration response spectrum of a ground motion.
    Uses the Newmark-Beta method for linear systems.
    """
    sa = np.zeros(len(frequencies))
    g = 9.81  # Acceleration due to gravity (m/s^2)

    for i, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq
        m = 1.0
        k = omega ** 2 * m
        c = 2 * damping * omega * m

        # Newmark-Beta parameters (average acceleration method)
        gamma = 0.5
        beta = 0.25

        # Initial conditions
        u, v, a = 0.0, 0.0, 0.0
        u_max = 0.0

        # The input acceleration is in g's, convert it to m/s^2 for calculation
        p = -m * (accel * g)

        # Pre-calculate constants for Newmark-Beta
        a1 = (1 / (beta * dt ** 2)) * m + (gamma / (beta * dt)) * c
        a2 = (1 / (beta * dt)) * m + (gamma / beta - 1) * c
        a3 = (1 / (2 * beta) - 1) * m + dt * (gamma / (2 * beta) - 1) * c
        k_hat = k + a1

        for p_i in p:
            p_hat = p_i + a1 * u + a2 * v + a3 * a
            u_new = p_hat / k_hat

            # Calculate velocity and acceleration updates
            delta_u = u_new - u
            v_new = v + (1 - gamma) * dt * a + gamma * dt * (
                        delta_u / (beta * dt ** 2) - v / (beta * dt) - ((1 / (2 * beta)) - 1) * a)
            a_new = (delta_u / (beta * dt ** 2)) - v / (beta * dt) - ((1 / (2 * beta)) - 1) * a

            u, v, a = u_new, v_new, a_new

            if abs(u) > u_max:
                u_max = abs(u)

        # Pseudo-spectral acceleration, converted back to g's
        sa[i] = u_max * omega ** 2 / g

    return sa


def plot_comparison_spectra(frequencies, sa1, sa2, label1, label2):
    """
    Plots two response spectra for comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(frequencies, sa1, label=label1, color='blue')
    ax.semilogx(frequencies, sa2, label=label2, color='green')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Spectral Acceleration (g)')
    ax.set_title('Response Spectra Comparison')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


st.title('Response Spectra Comparison Tool')

st.sidebar.header('Analysis Parameters')
damping = st.sidebar.slider('Damping Ratio (%)', 1.0, 10.0, 5.0, 0.5) / 100.0

st.sidebar.header('Frequency Range')
f_min = st.sidebar.number_input('Minimum Frequency (Hz)', 0.1, 10.0, 0.33)
f_max = st.sidebar.number_input('Maximum Frequency (Hz)', 10.0, 100.0, 33.0)
n_per_octave = st.sidebar.number_input('Points per Octave', 10, 50, 20)

st.sidebar.header('Upload Time Histories')

# --- File 1 Uploader ---
file1 = st.sidebar.file_uploader("Upload First File (.txt or .tim)", type=['txt', 'tim'], key="file1")
rate1 = st.sidebar.number_input('Sampling Rate for File 1 (Hz)', min_value=1, value=512)
channel1 = 0
if file1 and file1.name.lower().endswith('.tim'):
    channel1 = st.sidebar.number_input('Channel Index for File 1', min_value=0, step=1, value=0,
                                       help="0 for first channel, 1 for second, etc.")

st.sidebar.markdown("---")

# --- File 2 Uploader ---
file2 = st.sidebar.file_uploader("Upload Second File (.txt or .tim)", type=['txt', 'tim'], key="file2")
rate2 = st.sidebar.number_input('Sampling Rate for File 2 (Hz)', min_value=1, value=256)
channel2 = 0
if file2 and file2.name.lower().endswith('.tim'):
    channel2 = st.sidebar.number_input('Channel Index for File 2', min_value=0, step=1, value=0,
                                       help="0 for first channel, 1 for second, etc.")

if st.button('Calculate and Plot Spectra'):
    if file1 is not None and file2 is not None:
        try:
            # Save uploaded files to temporary local paths to be read by the loader function
            with open(file1.name, "wb") as f:
                f.write(file1.getbuffer())
            with open(file2.name, "wb") as f:
                f.write(file2.getbuffer())

            # Load data using the selected channel index
            accel1 = load_ground_motion_data(file1.name, channel_index=channel1)
            accel2 = load_ground_motion_data(file2.name, channel_index=channel2)

            # Clean up temporary files
            os.remove(file1.name)
            os.remove(file2.name)

            dt1 = 1.0 / rate1
            dt2 = 1.0 / rate2

            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"PGA (File 1: {file1.name}, Ch: {channel1})", value=f"{np.max(np.abs(accel1)):.4f} g")
                st.write(f"Number of data points: {len(accel1)}")
            with col2:
                st.metric(label=f"PGA (File 2: {file2.name}, Ch: {channel2})", value=f"{np.max(np.abs(accel2)):.4f} g")
                st.write(f"Number of data points: {len(accel2)}")

            # Define frequencies for the spectrum
            frequencies = octave_spacing(f_min, f_max, n_per_octave)

            with st.spinner('Calculating spectra... This may take a moment.'):
                # Calculate response spectra
                sa1 = calculate_response_spectrum(accel1, dt1, frequencies, damping)
                sa2 = calculate_response_spectrum(accel2, dt2, frequencies, damping)

                # Plot the comparison
                fig = plot_comparison_spectra(frequencies, sa1, sa2, f'Spectrum for {file1.name}',
                                              f'Spectrum for {file2.name}')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure the channel index is valid for the uploaded file.")
    else:
        st.warning('Please upload both files before proceeding.')