"""
Microphone Detection, Recording, and Audio Analysis Script
========================================================
This script provides a complete solution for:
  1. Detecting all available microphones (built-in or USB)
  2. Recording audio from a selected microphone
  3. Performing comprehensive audio analysis
  4. Visualizing results with plots and spectrograms

Author: Audio Analysis Tool
Date: January 2026
"""

# ============================================================================
# IMPORTS
# ============================================================================
# sounddevice: Audio recording and device management library
import sounddevice as sd

# numpy: Numerical computing for array operations and calculations
import numpy as np

# scipy.signal: Signal processing functions (unused in current version but useful for future extensions)
from scipy import signal

# scipy.fft: Fast Fourier Transform for frequency domain analysis
from scipy.fft import fft, fftfreq, rfft, rfftfreq

# matplotlib: Plotting library for visualization
import matplotlib.pyplot as plt

# time: Time-related functions (unused in current version but available for timing analysis)
import time

# datetime: For generating timestamped filenames
from datetime import datetime

import scipy.io.wavfile as wavfile


import os, re
class MicrophoneAnalyzer:
    """
    Main class for microphone detection, audio recording, and analysis.
    
    This class encapsulates all functionality needed to work with microphones:
    - Discover available input devices
    - Record audio at specified sample rate and duration
    - Perform time-domain and frequency-domain analysis
    - Generate visualization plots
    
    Attributes:
        sample_rate (int): Audio sample rate in Hz (default 44100 Hz)
        duration (int): Recording duration in seconds (default 5 seconds)
        audio_data (ndarray): Recorded audio samples
        device_id (int): ID of selected microphone device
    """
    
    def __init__(self):
        """Initialize the analyzer with default settings."""
        self.sample_rate = 44100  # Standard CD quality audio (44.1 kHz)
        self.duration = 5  # Record for 5 seconds
        self.audio_data = None  # No audio recorded yet
        self.device_id = None  # Use system default microphone
        
    def list_microphones(self):
        """
        List all available audio input devices (microphones).
        
        Queries the system for all audio devices and filters for input devices
        (those with input channels). Displays detailed information for each.
        
        Returns:
            list: List of tuples (device_id, device_info) for all microphones,
                  or None if no microphones are found.
        """
        print("\n" + "="*60)
        print("AVAILABLE MICROPHONES")
        print("="*60)
        
        # Query the system for all audio devices
        devices = sd.query_devices()
        microphones = []
        
        # Iterate through all devices and filter for input devices
        for i, device in enumerate(devices):
            # Check if device has input channels (microphone capability)
            if device['max_input_channels'] > 0:
                microphones.append((i, device))
                print(f"\n[Device {i}] {device['name']}")
                print(f"  Channels: {device['max_input_channels']}")
                print(f"  Sample Rate: {device['default_samplerate']} Hz")
                print(f"  Latency: {device['default_low_input_latency']} s")
        
        # Handle case where no microphones are found
        if not microphones:
            print("No microphones found!")
            return None
            
        return microphones
    
    def select_microphone(self, microphones):
        """
        Prompt user to select a microphone from available devices.
        
        Allows the user to choose which microphone to use for recording,
        or use the system default if -1 is entered.
        
        Args:
            microphones (list): List of (device_id, device_info) tuples
        """
        print("\n" + "="*60)
        while True:
            try:
                # Get user input for device selection
                choice = int(input("Enter the device number to use (or -1 for default): "))
                
                # Handle default device selection
                if choice == -1:
                    print("Using default microphone")
                    self.device_id = None
                    return
                
                # Validate that selected device exists
                if any(dev[0] == choice for dev in microphones):
                    self.device_id = choice
                    # Find and display the name of selected device
                    for dev_id, dev_info in microphones:
                        if dev_id == choice:
                            print(f"Selected device: {dev_info['name']}")
                    return
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                # Handle non-numeric input
                print("Please enter a valid number.")
                
    def a_weighting(self, n_fft):
        """Calculates A-weighting coefficients for each frequency bin."""
        freqs = rfftfreq(n_fft, 1/self.sample_rate)
        # Avoid log of zero at 0Hz
        f = np.where(freqs == 0, 1e-10, freqs)
        
        # Standard IEC 61672-1:2003 A-weighting formula
        f1 = 20.602
        f2 = 107.652
        f3 = 737.862
        f4 = 12194.217
        a1000 = -2.000 # Normalization factor to ensure 1kHz = 0dB
        
        num = (12194.217**2) * (f**4)
        den = ((f**2 + f1**2) * np.sqrt((f**2 + f2**2) * (f**2 + f3**2)) * (f**2 + f4**2))
        
        # Calculate weight in dB
        ra_f = 20 * np.log10(num / den) - a1000
        return ra_f
    
    def record_audio(self, recording_name=None):
        """
        Record audio from the selected microphone.
        
        Captures audio at the configured sample rate and duration.
        The recording is stored in self.audio_data for later analysis.
        
        Returns:
            bool: True if recording succeeded, False if an error occurred.
        """
        print("\n" + "="*60)
        print("RECORDING AUDIO")
        print("="*60)
        print(f"Recording for {self.duration} seconds...")
        print("(Keep your microphone in the recording position)")
        
        try:
            # Use sounddevice to record audio
            # - int(self.sample_rate * self.duration) = total number of samples to record
            # - channels=1 = mono recording (single channel)
            # - dtype='float32' = 32-bit floating point format for high quality
            self.audio_data = sd.rec(
                int(self.sample_rate * self.duration),
                samplerate=self.sample_rate,
                channels=1,
                device=self.device_id,
                dtype='float32'
            )
            
            # Wait for the recording to complete (blocks until done)
            sd.wait()
            print("✓ Recording complete!")
            if recording_name == None:
                pass
            else:
                wavfile.write(recording_name, self.sample_rate, self.audio_data) # f"extra_audio/{recording_name}.wav", self.sample_rate, self.audio_data)
                print("✓ Recording saved as: " + recording_name)
            return True
        except Exception as e:
            # Catch any errors during recording (e.g., device disconnected)
            print(f"✗ Error during recording: {e}")
            return False
        
    def overall_a_spl(self):
        
        try:
            audio = self.audio_data.flatten()
        except Exception:
            print("No data found!")
        
        
        # Perform FFT (Real-valued FFT for efficiency)
        n = len(audio)
        fft_data = rfft(audio)
        # Normalize magnitude so 1.0 peak in time = 1.0 peak in frequency
        magnitudes = np.abs(fft_data) * (2 / n) 

        # 3. Get A-weighting curve
        weights_db = self.a_weighting(n)

        # 4. Apply weighting
        # Convert magnitudes to dB, add weights, then convert back to linear power
        # Using 1.0 as the digital reference (dBFS)
        mags_db = 20 * np.log10(magnitudes + 1e-12) # +1e-12 to avoid log(0)
        a_weighted_mags_db = mags_db + weights_db

        # 5. Calculate Overall Level
        # Summing power (squared pressure) is the correct way to get 'total' sound level
        total_power = np.sum(10**(a_weighted_mags_db / 10))
        total_db_a = 10 * np.log10(total_power)

        # print(f"Overall Level: {total_db_a:.2f} dBFS(A)")
        return total_db_a
    
    
    def analyze_audio(self):
        """
        Perform comprehensive audio analysis on recorded audio.
        
        Conducts multiple analyses including:
        - Time-domain metrics (RMS, peak, mean, standard deviation)
        - Zero crossing rate (indicates noise/complexity)
        - Frequency-domain analysis using FFT
        - Identification of dominant frequency
        - Energy distribution across frequency bands
        
        Returns:
            dict: Dictionary containing analysis results and intermediate data,
                  or None if no audio data is available.
        """
        if self.audio_data is None:
            print("No audio data to analyze!")
            return
        
        print("\n" + "="*60)
        print("AUDIO ANALYSIS")
        print("="*60)
        
        # Flatten audio to 1D array (removes channel dimension from mono recording)
        audio = self.audio_data.flatten()
        
        # ========== TIME-DOMAIN ANALYSIS ==========
        # RMS (Root Mean Square): Represents overall loudness/intensity
        rms_level = np.sqrt(np.mean(audio**2))
        
        # Peak level: Maximum absolute amplitude reached
        peak_level = np.max(np.abs(audio))
        
        # Mean: Average amplitude (should be close to 0 for zero-centered audio)
        mean_level = np.mean(audio)
        
        # Standard deviation: Spread of amplitude values (high = louder, more variation)
        std_level = np.std(audio)
        
        print(f"\nTime-Domain Analysis:")
        print(f"  RMS Level: {rms_level:.6f}")
        print(f"  Peak Level: {peak_level:.6f}")
        print(f"  Mean Level: {mean_level:.6f}")
        print(f"  Standard Deviation: {std_level:.6f}")
        
        # ========== ZERO CROSSING RATE ==========
        # Counts how many times the signal crosses zero amplitude
        # High ZCR = noisy/complex signal; Low ZCR = simple/pure tone
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        zcr = zero_crossings / len(audio)
        print(f"\nZero Crossing Rate: {zcr:.6f}")
        
        # ========== FREQUENCY-DOMAIN ANALYSIS (FFT) ==========
        # Apply Fast Fourier Transform to convert time-domain to frequency-domain
        fft_values = fft(audio)
        
        # Generate frequency axis (Hz) corresponding to each FFT bin
        fft_freq = fftfreq(len(audio), 1/self.sample_rate)
        
        # Calculate magnitude spectrum (absolute values of complex FFT results)
        fft_magnitude = np.abs(fft_values)
        
        # Get positive frequencies only (negative frequencies are mirror images)
        positive_freq_idx = fft_freq > 0
        positive_freqs = fft_freq[positive_freq_idx]
        positive_magnitude = fft_magnitude[positive_freq_idx]
        
        # Find the frequency with the highest energy (dominant frequency)
        dominant_freq_idx = np.argmax(positive_magnitude)
        dominant_freq = positive_freqs[dominant_freq_idx]
        
        print(f"\nFrequency-Domain Analysis:")
        print(f"  Dominant Frequency: {dominant_freq:.2f} Hz")
        print(f"  Frequency Range: {positive_freqs.min():.2f} - {positive_freqs.max():.2f} Hz")
        
        # ========== FREQUENCY BAND ANALYSIS ==========
        # Analyze energy distribution across different frequency ranges
        # Useful for understanding what types of sounds are present
        bands = {
            'Sub-bass (20-60 Hz)': (20, 60),
            'Bass (60-250 Hz)': (60, 250),
            'Midrange (250-2000 Hz)': (250, 2000),
            'Treble (2000-6000 Hz)': (2000, 6000),
            'High-treble (6000+ Hz)': (6000, self.sample_rate/2)
        }
        
        print(f"\nEnergy by Frequency Band:")
        for band_name, (low_freq, high_freq) in bands.items():
            # Create mask for frequencies within this band
            band_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
            
            # Calculate average energy in this band
            band_energy = np.mean(positive_magnitude[band_mask]) if band_mask.any() else 0
            print(f"  {band_name}: {band_energy:.6f}")
            
        oaspl = self.overall_a_spl()
        
        # calculate psd
        # 1. Calculate the BPF (Blade Passing Frequency)
        # This is the 'signature' frequency of the fan
        # bpf = (rpm * num_blades) / 60
        
        # 2. Compute the Power Spectral Density (PSD) using Welch's Method
        # nperseg defines the frequency resolution (higher = more detail)
        freqs, psd = signal.welch(audio, self.sample_rate, nperseg=self.sample_rate//4) 
        
        # 3. Convert to dB (Reference = 1.0 for dBFS/Hz)
        psd_db = 10 * np.log10(psd + 1e-12)
        
        # Return all analysis results and intermediate data for visualization
        return {
            'rms': rms_level,
            'peak': peak_level,
            'zcr': zcr,
            'dominant_freq': dominant_freq,
            'freqs': positive_freqs,
            'magnitude': positive_magnitude,
            'time': np.linspace(0, self.duration, len(audio)),
            'audio': audio,
            'psd': psd_db,
            'psd_freqs': freqs,
        }
    
    def plot_results(self, analysis_data):
        """
        Create and display visualizations of the audio analysis results.
        
        Generates a 2x2 subplot figure containing:
        1. Waveform: Raw audio signal over time
        2. Frequency Spectrum: Full frequency response (FFT magnitude)
        3. Zoomed Spectrum: Focused view on 0-5 kHz range
        4. Spectrogram: Time-frequency representation showing how frequency content changes over time
        
        The figure is saved as a PNG file with a timestamp in the filename.
        
        Args:
            analysis_data (dict): Dictionary containing analysis results from analyze_audio()
        """
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle('Audio Analysis Results', fontsize=16)
        
        # ========== SUBPLOT 1: WAVEFORM ==========
        # Shows the raw audio amplitude over time
        ax = axes[0, 0]
        ax.plot(analysis_data['time'], analysis_data['audio'], linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform')
        ax.grid(True, alpha=0.3)
        
        # ========== SUBPLOT 2: FULL FREQUENCY SPECTRUM ==========
        # Shows magnitude of each frequency component (logarithmic scale)
        ax = axes[0, 1]
        ax.semilogy(analysis_data['freqs'], analysis_data['magnitude'], linewidth=0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Frequency Spectrum (Dominant: {analysis_data["dominant_freq"]:.1f} Hz)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10000)  # Focus on typical human hearing range
        
        # ========== SUBPLOT 3: ZOOMED FREQUENCY SPECTRUM ==========
        # Provides more detail on lower frequencies (0-5 kHz)
        ax = axes[1, 0]
        zoom_mask = analysis_data['freqs'] <= 1000
        ax.plot(analysis_data['freqs'][zoom_mask], analysis_data['magnitude'][zoom_mask], linewidth=0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('Frequency Spectrum (0-5kHz)')
        ax.grid(True, alpha=0.3)
        
        # ========== SUBPLOT 4: SPECTROGRAM ==========
        # Shows frequency content over time (2D visualization)
        # X-axis = time, Y-axis = frequency, Color = power intensity
        ax = axes[1, 1]
        Pxx, freqs, bins, im = ax.specgram(analysis_data['audio'], 
                                           Fs=self.sample_rate, 
                                           cmap='viridis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')
        ax.set_ylim(0, 5000)  # Focus on 0-5 kHz range
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # ============ SUBPLOT 5; PSD ==============
        bpf = 3000*9/60
        ax = axes[2, 1]
        ax.semilogx(analysis_data['psd_freqs'], analysis_data['psd'],label=f"Signature", color='royalblue')
        ax.axvline(bpf, color='red', linestyle='--', alpha=0.6, label=f'BPF ({bpf:.1f} Hz)')
        ax.text(bpf * 1.1, np.max(analysis_data['psd']), 'BPF', color='red', fontweight='bold')
        # 6. Formatting for a Year 4 Project report
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD [dB/Hz] (ref: 1.0)')
        ax.set_title('PSD')
        ax.grid(True, alpha=0.3)
        
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # ========== SAVE FIGURE ==========
        # Generate timestamped filename to avoid overwriting previous analyses
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = r"audio\audio_analysis_" + f"{timestamp}.png"
        # plt.savefig(filename, dpi=150)
        # print(f"\n✓ Analysis plot saved as: {filename}")
        
        # Display the figure (blocks until window is closed)
        plt.show()
    
    def run(self):
        """
        Execute the complete microphone analysis pipeline.
        
        Orchestrates the entire workflow:
        1. Detects and displays available microphones
        2. Prompts user to select a microphone
        3. Records audio from the selected device
        4. Performs comprehensive audio analysis
        5. Generates and displays visualization plots
        """
        print("\n" + "="*60)
        print("MICROPHONE RECORDER & AUDIO ANALYZER")
        print("="*60)
        
        # Step 1: List all available microphones
        microphones = self.list_microphones()
        if not microphones:
            return
        
        # Step 2: Prompt user to select a microphone
        self.select_microphone(microphones)
        entry    = False
        doe_bool = False
        inverse_design_bool = False
        while not entry:
            doe = input("Is this a DOE test? (y/n) ")
            if doe == "y":
                entry = True
                try:
                    blade_number = int(input("What blade is this? (Enter integer) "))
                    doe_bool = True
                    stl_file = None
                    for file in os.listdir("stl_files"):
                        match = re.search(f"DOE_{blade_number}", file)
                        if match:
                            parts = file.split(sep=".")
                            print(parts)
                            if parts[-1] == "stl":
                                stl_file = file.rstrip(".stl")
                    if stl_file is None:
                        raise ValueError("No file found")
                    # CSV_FILENAME   = f'data/doe_data/{stl_file}.csv'
                    recording_name = f'audio/doe_data_2/{stl_file}.wav'
                    # print(CSV_FILENAME)
                except TypeError:
                    print("invalid entry")
                    entry = False
            elif doe == "n":
                inverse_design = input("Is this inverse design? (y/n) ")
                if inverse_design == "y":
                    entry = True
                    try:
                        inverse_design_bool = True
                        inverse_blade_number = int(input("What blade is this? (Enter integer) "))
                        stl_file = None
                        for file in os.listdir("stl_files"):
                            match = re.search(f"inverse_design_test_{inverse_blade_number}", file)
                            if match:
                                parts = file.split(sep=".")
                                print(parts)
                                if parts[-1] == "stl":
                                    stl_file = file.rstrip(".stl")
                        if stl_file is None:
                            raise ValueError("No file found")
                        # CSV_FILENAME   = f'data/doe_data/{stl_file}.csv'
                        recording_name = f'audio/inverse_design_2/{stl_file}.wav'
                        # print(CSV_FILENAME)
                    except TypeError:
                        print("invalid entry")
                        entry = False
                else:
                    entry = True
            else:
                print("Enter a valid response (y/n)")
        
        
        if not doe_bool and not inverse_design_bool:
            blade = input("Enter blade type: ")
            if blade != "":
                recording_name = f'audio/recording_{blade}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
            
        # Step 3: Record audio from selected microphone
        if not self.record_audio(recording_name):
            return
        
        # Step 4: Analyze the recorded audio
        analysis_data = self.analyze_audio()
        if analysis_data:
            # Step 5: Create and display visualizations
            self.plot_results(analysis_data)
        
        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60 + "\n")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    """
    Program entry point - runs only when script is executed directly,
    not when imported as a module.
    """
    # Create an instance of the analyzer and run the complete pipeline
    analyzer = MicrophoneAnalyzer()
    analyzer.run()
    # print(analyzer.audio_data)
