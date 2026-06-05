"""
SERIAL DATA LOGGER FOR AXIAL FAN RIG EXPERIMENTS

This script handles real-time data acquisition from an instrumented axial fan test rig.
It manages three Arduino serial connections for different measurements and integrates
microphone audio recording. Data is continuously logged to CSV and displayed in real-time plots.

Key workflow:
1. Connect to three Arduino devices (RPM controller, pressure sensors, throttle controller)
2. Initialize microphone recording
3. Send test start command to all devices
4. Continuously receive and process data from each source
5. Synchronize multi-source data using point-based buffering
6. Calculate derived parameters (flow coefficients, pressure rise, efficiency, etc.)
7. Save results to CSV and audio to WAV file
8. Display live updating plots of all parameters
"""

import matplotlib
try:
    matplotlib.use('TkAgg')  # Use TkAgg backend for cross-platform compatibility
except:
    pass

import serial  # Serial communication with Arduino devices
import threading  # Multi-threaded data acquisition
import csv  # CSV file writing
import time  # Timing and delays
import matplotlib.pyplot as plt  # Plotting
from matplotlib.animation import FuncAnimation  # Real-time plot animation
from collections import deque  # Fixed-size buffers for plot data
from datetime import datetime  # Timestamps
import numpy as np  # Numerical operations
import os  # File system operations
import re  # Regular expressions for file matching
import scipy.io.wavfile as wavfile  # Audio file writing

from microphone_recorder import MicrophoneAnalyzer  # Microphone audio recording
from noise import end_alert  # Alert sound when test reaches point 70

# ============================================================================
# ARDUINO & SERIAL COMMUNICATION CONFIGURATION
# ============================================================================
RPM_PORT          = 'COM4'       # Arduino Nano 1: RPM and power data
STATS_PORT        = 'COM5'       # Arduino ESP32: Pressure and temperature data
SUCTION_FAN_PORT  = 'COM6'       # Arduino Nano 2: Throttle position data
BAUDRATE          = 115200       # Serial communication speed (bits/sec)

# Default file paths (may be overridden by user input for DOE/inverse design tests)
CSV_FILENAME      = f'data/test_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
AUDIO_FILENAME    = f'audio/recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'

# ============================================================================
# PHYSICAL CONSTANTS & RIG GEOMETRY
# ============================================================================
# Venturi discharge coefficient and expansion factor
C       = 0.973
epsilon = 1

# Venturi geometry (pressure measurement location)
d       = 0.050   # Venturi throat diameter (m)
D       = 0.080   # Main duct diameter (m)
beta    = d / D   # Diameter ratio
E       = (1 - beta**4) ** (-1.0 / 2.0)  # Expansion factor for venturi equations

# Fan geometry and measurement areas
rmid    = 0.0275  # Fan blade mid-radius (m) - used for velocity calculations
h       = 0.024   # Fan blade height (m)
Afan    = 2 * np.pi * rmid * h  # Fan swept area (m^2) for mass flow calculation
Athroat = np.pi * d**2 / 4      # Venturi throat area (m^2)

# ============================================================================
# ATMOSPHERIC CONDITIONS
# ============================================================================
# NOTE: Adjust these based on current atmospheric/room conditions for accurate density calculations
pressureAtm = 1.01252 * 10**5  # Atmospheric pressure (Pa)
TAtm        = 273.15 + 20      # Room temperature (K) [currently 20°C]

# Derived constants
rho_default    = pressureAtm / 287 / TAtm  # Air density at reference conditions (kg/m^3)
RPM_TO_U_CONV  = rmid * np.pi / 30         # Conversion factor: RPM to blade speed (m/s)

# ============================================================================
# DATA BUFFERING & SYNCHRONIZATION
# ============================================================================
# Point-keyed data buffer strategy:
# - Data arrives asynchronously from 3 Arduino sources and microphone
# - Each measurement point is keyed by index (0, 1, 2, ...)
# - We buffer partial data until ALL sources have reported for that point
# - When complete, the point is processed and removed from buffer
# 
# This ensures synchronized multi-source data for each test point

point_buffer = {}   # { point_index: { 'rpm': {...}, 'pressure': {...}, 'throttle': {...}, 'mic': {...} } }
buffer_lock  = threading.Lock()  # Thread-safe access to point_buffer

# ============================================================================
# MICROPHONE & AUDIO DATA STORAGE
# ============================================================================
# Microphone is recorded at regular intervals during the test
# Data is stored separately then concatenated and saved at end

mic_data             = {'point': None, 'oaspl': None}  # Current microphone reading
mic_analyser         = None  # MicrophoneAnalyzer instance
test_start_time      = None  # Timestamp when test began (used for recording sync)
mic_lock             = threading.Lock()  # Thread-safe access to mic_data

# ============================================================================
# REAL-TIME PLOT DATA BUFFERS
# ============================================================================
# Fixed-size deques for live plotting (auto-discard old data when full)
MAX_PLOT_POINTS = 250  # Keep last 250 points on screen
plot_data = {
    # Time axis
    'time':          deque(maxlen=MAX_PLOT_POINTS),
    
    # Mean values and error bars (std dev)
    'rpm_mean':      deque(maxlen=MAX_PLOT_POINTS),
    'mflow_mean':    deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_mean': deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_mean': deque(maxlen=MAX_PLOT_POINTS),
    'phi_mean':      deque(maxlen=MAX_PLOT_POINTS),  # Flow coefficient
    'pRise_mean':    deque(maxlen=MAX_PLOT_POINTS),  # Pressure rise coefficient
    'rho_mean':      deque(maxlen=MAX_PLOT_POINTS),  # Air density
    'axvel_mean':    deque(maxlen=MAX_PLOT_POINTS),  # Axial velocity
    'efficiency_mean': deque(maxlen=MAX_PLOT_POINTS),
    
    # Error bars (standard deviations)
    'rpm_err':       deque(maxlen=MAX_PLOT_POINTS),
    'mflow_err':     deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_err':deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_err':  deque(maxlen=MAX_PLOT_POINTS),
    'phi_err':       deque(maxlen=MAX_PLOT_POINTS),
    'pRise_err':     deque(maxlen=MAX_PLOT_POINTS),
    'rho_err':       deque(maxlen=MAX_PLOT_POINTS),
    'axvel_err':     deque(maxlen=MAX_PLOT_POINTS),
    'efficiency_err':  deque(maxlen=MAX_PLOT_POINTS),
    
    # Noise
    'oaspl':         deque(maxlen=MAX_PLOT_POINTS),  # Overall A-weighted sound pressure level
}

# CSV file handles
csv_file   = None  # File handle for CSV output
csv_writer = None  # CSV writer object


def init_csv():
    """
    Initialize CSV file and write header row with all column names.
    
    Columns include:
    - Timestamp and point number
    - Primary measurements (RPM, pressure, temperature)
    - Derived parameters (mass flow, flow coefficients, efficiency)
    - Uncertainty/error propagation (std dev for each parameter)
    - Noise measurement (OASPL)
    """
    global csv_file, csv_writer
    csv_file   = open(CSV_FILENAME, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'timestamp', 'point',
        'rpm_mean', 'rpm_stddev',
        'dp_venturi_mean', 'dp_venturi_stddev',
        'dp_stage_mean', 'dp_stage_stddev',
        'rho_mean', 'rho_stddev',
        'mflow_mean', 'mflow_stddev',
        'axvelocity_mean', 'axvelocity_stddev',
        'flow_coefficient_mean', 'flow_coefficient_stddev',
        'pressure_rise_coefficient_mean', 'pressure_rise_coefficient_stddev',
        'throttle_location',
        'oaspl',
        'power_mean', 'power_stddev',
        'efficiency_mean', 'efficiency_stddev'
    ])
    csv_file.flush()


def init_microphone():
    """
    Initialize microphone analyzer for audio recording.
    
    Sets up audio recording with 2-second duration (one recording per test point).
    Returns True if successful, False otherwise.
    """
    global mic_analyser, mic_recordings
    try:
        mic_analyser          = MicrophoneAnalyzer()
        mic_analyser.duration = 2  # Record 2 seconds per point
        mic_analyser.device_id = None  # Use default microphone device
        mic_recordings        = []
        print("✅ Microphone ready")
        return True
    except Exception as e:
        print(f"❌ Microphone init failed: {e}")
        return False


def microphone_recording_thread():
    """
    Continuous microphone recording thread - runs during entire test.
    
    Records audio at intervals synchronized with test points:
    - First recording starts at t=12s (after 10s startup + 2s adjustment)
    - Subsequent recordings every 4s (2s adjust + 2s record window)
    - Each recording is stored and associated with the current point number
    - OASPL (overall A-weighted sound pressure level) is extracted from each recording
    - Data is stored in the point buffer for synchronization with other sensors
    """
    global test_start_time, mic_analyser
    if not mic_analyser:
        return
    print("✅ Connected to microphone analyser")

    # Wait for test to start before beginning recordings
    while test_start_time is None:
        time.sleep(0.1)

    point_count = 0
    while True:
        try:
            elapsed = time.time() - test_start_time
            # Recording window starts at 13s (10s startup + 2s adjust + 1s margin)
            # then every 4s (2s adjust + 2s record)
            recording_interval   = 4
            next_recording_time  = 12 + (point_count * recording_interval)

            if elapsed >= next_recording_time:
                print(f"\n🎤 Recording audio for point {point_count}...")
                if mic_analyser.record_audio():
                    with mic_lock:
                        mic_data['point'] = point_count
                        mic_data['oaspl'] = mic_analyser.overall_a_spl()  # Extract OASPL
                    mic_recordings.append(mic_analyser.audio_data.copy())
                    # Store in point buffer for synchronization
                    _store_in_buffer(point_count, 'mic', {'oaspl': mic_data['oaspl']})
                    print(f"✅ Audio recorded for point {point_count}\n")
                point_count += 1

            time.sleep(0.01)
        except Exception as e:
            print(f"⚠️ Microphone thread error: {e}")
            time.sleep(1)


# ============================================================================
# BUFFER MANAGEMENT & DATA SYNCHRONIZATION
# ============================================================================

def _store_in_buffer(point, source, data):
    """
    Store data from one measurement source into the point buffer.
    
    When all four sources (rpm, pressure, throttle, mic) have reported for a point,
    automatically triggers processing of that complete point.
    
    Args:
        point: Point number (test index)
        source: Source identifier ('rpm', 'pressure', 'throttle', or 'mic')
        data: Dictionary of measurements from this source
    """
    with buffer_lock:
        if point not in point_buffer:
            point_buffer[point] = {}
        point_buffer[point][source] = data

        # Check if all required sources have arrived for this point
        entry = point_buffer[point]
        if 'rpm' in entry and 'pressure' in entry and 'throttle' in entry and 'mic' in entry:
            _process_complete_point(point, entry)
            del point_buffer[point]  # Clean up after processing


def _process_complete_point(point, entry):
    """
    Process a complete data point when all measurement sources have arrived.
    
    This is the main data processing pipeline:
    1. Extract primary measurements from each source
    2. Calculate derived parameters (mass flow, coefficients, efficiency)
    3. Perform error propagation through all calculations
    4. Save results to CSV file
    5. Update real-time plot data
    6. Print summary to console
    
    Key calculations:
    - Blade speed (U) from RPM using hub radius
    - Axial velocity from venturi pressure drop
    - Mass flow from velocity and fan swept area
    - Flow coefficient (Φ = axial_velocity / blade_speed)
    - Pressure rise coefficient (ψ = pressure_rise / (0.5*rho*U^2))
    - Isentropic efficiency = (aero power) / (motor power)
    
    Args:
        point: Point number
        entry: Dictionary with keys 'rpm', 'pressure', 'throttle', 'mic' containing all measurements
    """
    # Extract primary measurements from each source
    rpm      = entry['rpm']      # From Arduino Nano 1
    pressure = entry['pressure']  # From Arduino ESP32
    throttle = entry['throttle']  # From Arduino Nano 2
    mic_oaspl = entry['mic'].get('oaspl', -100)  # From microphone

    # === EXTRACT PRIMARY MEASUREMENTS ===
    rpm_mean  = rpm.get('mean', 0)
    rpm_std   = rpm.get('stddev', 0)
    power_mean = rpm.get('pwr_mean', 0)  # Motor electrical power (mW)
    power_std  = rpm.get('pwr_std', 0)

    throttle_location = throttle.get('location', 0)

    # Pressure differences from venturi and stage measurement sensors
    dp_sensor0_mean = pressure.get('dp_sensor0_mean', 0)  # Venturi throat
    dp_sensor0_std  = pressure.get('dp_sensor0_stddev', 0)
    dp_sensor1_mean = pressure.get('dp_sensor1_mean', 0)  # Stage pressure rise
    dp_sensor1_std  = pressure.get('dp_sensor1_stddev', 0)
    temp_mean       = pressure.get('temp_mean', 15)  # °C
    temp_std        = pressure.get('temp_stddev', 0)

    # === CALCULATE PRIMARY PARAMETERS ===
    # Blade speed at mid-radius
    U      = RPM_TO_U_CONV * rpm_mean  # m/s
    U_sq   = U ** 2
    
    # Convert temperature and calculate air density
    T_abs  = 273.15 + temp_mean  # Absolute temperature (K)
    rho_mean = pressureAtm / (287 * T_abs)  # Air density using ideal gas law

    # Axial velocity from venturi pressure drop (compressible flow formula)
    if dp_sensor0_mean < 0:
        axvel_throat_mean = 0
    else:
        # v = sqrt(2*dp / rho) from Bernoulli equation
        axvel_throat_mean = np.sqrt(dp_sensor0_mean / (0.5 * rho_mean))
    
    # Mass flow rate through fan
    mdot_mean         = rho_mean * Athroat * axvel_throat_mean  # kg/s
    
    # Axial velocity through fan (convert from throat to fan area)
    axvel_mean        = mdot_mean / (Afan * rho_mean)
    
    # Pressure measurements
    dp_stage_mean     = dp_sensor1_mean  # Overall stage pressure rise
    dp_venturi_mean   = dp_sensor0_mean
    dp_venturi_std    = dp_sensor0_std

    # === DIMENSIONLESS COEFFICIENTS ===
    # Flow coefficient: ratio of axial velocity to blade speed
    phi   = axvel_mean / U if abs(U) > 1e-6 else 0.0
    
    # Pressure rise coefficient: normalized by dynamic head
    pRise = dp_stage_mean / (rho_mean * U_sq) if (U_sq > 1e-6 and rho_mean > 0) else 0.0

    # === EFFICIENCY CALCULATION ===
    # Aero power = pressure_rise * volumetric_flow_rate
    aero_power_mean = dp_stage_mean * mdot_mean / rho_mean  # W (convert mass flow to volume flow)
    
    # Isentropic efficiency = aero power / motor power
    efficiency_mean = aero_power_mean / (power_mean / 1000) if power_mean > 0 else 0.0  # power_mean in mW

    # === ERROR PROPAGATION ===
    # All parameters calculated above have uncertainty due to measurement noise
    # Standard error propagation: for f(x,y), σ_f = sqrt((∂f/∂x)^2 * σ_x^2 + (∂f/∂y)^2 * σ_y^2)
    
    # Density uncertainty from temperature variation
    rho_std = rho_mean * (temp_std / T_abs) if T_abs != 0 else 0

    # Axial velocity at venturi throat
    rel_var_dp0        = (dp_sensor0_std / dp_sensor0_mean) ** 2 if dp_sensor0_mean != 0 else 0
    rel_var_rho_local  = (rho_std / rho_mean) ** 2 if rho_mean != 0 else 0
    axvel_throat_std   = axvel_throat_mean * np.sqrt(0.25 * rel_var_dp0 + 0.25 * rel_var_rho_local) if axvel_throat_mean != 0 else 0

    # Mass flow uncertainty
    rel_var_axvel = (axvel_throat_std / axvel_throat_mean) ** 2 if axvel_throat_mean != 0 else 0
    mdot_std      = mdot_mean * np.sqrt(rel_var_rho_local + rel_var_axvel) if mdot_mean != 0 else 0
    
    # Axial velocity uncertainty at fan
    axvel_std     = axvel_mean * np.sqrt(rel_var_rho_local + (mdot_std / mdot_mean) ** 2) if axvel_mean != 0 else 0

    # Blade speed uncertainty
    rel_var_rpm = (rpm_std / rpm_mean) ** 2 if rpm_mean != 0 else 0
    rel_var_U   = rel_var_rpm
    rel_var_rho = (rho_std / rho_mean) ** 2 if rho_mean != 0 else 0

    # Pressure rise uncertainty (from sensor and velocity uncertainties)
    term1_sq   = dp_sensor1_std ** 2
    term2_sq   = (0.5 * axvel_mean ** 2 * rho_std) ** 2
    term3_sq   = (rho_mean * axvel_mean * axvel_std) ** 2
    dp_stage_std = np.sqrt(term1_sq + term2_sq + term3_sq)

    # Flow coefficient uncertainty
    rel_var_vx = (axvel_std / axvel_mean) ** 2 if axvel_mean != 0 else 0
    phi_std    = abs(phi) * np.sqrt(rel_var_vx + rel_var_U)

    # Pressure rise coefficient uncertainty
    rel_var_dp = (dp_stage_std / dp_stage_mean) ** 2 if dp_stage_mean != 0 else 0
    pRise_std  = abs(pRise) * np.sqrt(rel_var_dp + rel_var_rho + 4 * rel_var_U)

    # Efficiency uncertainty
    rel_std_dp       = dp_stage_std / dp_stage_mean if dp_stage_mean != 0 else 0
    rel_std_mdot     = mdot_std / mdot_mean if mdot_mean != 0 else 0
    rel_std_rho      = rho_std / rho_mean if rho_mean != 0 else 0
    rel_std_power    = power_std / power_mean if power_mean != 0 else 0
    rel_std_eff      = np.sqrt(rel_std_dp**2 + rel_std_mdot**2 + rel_std_rho**2 + rel_std_power**2)
    efficiency_std   = efficiency_mean * rel_std_eff

    # === SAVE TO CSV ===
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    csv_writer.writerow([
        timestamp, point,
        rpm_mean, rpm_std,
        dp_venturi_mean, dp_venturi_std,
        dp_stage_mean, dp_stage_std,
        rho_mean, rho_std,
        mdot_mean, mdot_std,
        axvel_mean, axvel_std,
        phi, phi_std,
        pRise, pRise_std,
        throttle_location,
        mic_oaspl,
        power_mean, power_std,
        efficiency_mean, efficiency_std,
    ])
    csv_file.flush()

    # === UPDATE PLOT DATA ===
    # Store values and error bars for real-time visualization
    plot_data['time'].append(time.time())
    plot_data['rpm_mean'].append(rpm_mean)
    plot_data['rpm_err'].append(rpm_std)
    plot_data['mflow_mean'].append(mdot_mean)
    plot_data['mflow_err'].append(mdot_std)
    plot_data['dp_venturi_mean'].append(dp_venturi_mean)
    plot_data['dp_venturi_err'].append(dp_venturi_std)
    plot_data['dp_stage_mean'].append(dp_stage_mean)
    plot_data['dp_stage_err'].append(dp_stage_std)
    plot_data['rho_mean'].append(rho_mean)
    plot_data['rho_err'].append(rho_std)
    plot_data['phi_mean'].append(phi)
    plot_data['phi_err'].append(phi_std)
    plot_data['pRise_mean'].append(pRise)
    plot_data['pRise_err'].append(pRise_std)
    plot_data['axvel_mean'].append(axvel_mean)
    plot_data['axvel_err'].append(axvel_std)
    plot_data['efficiency_mean'].append(efficiency_mean)
    plot_data['efficiency_err'].append(efficiency_std)
    plot_data['oaspl'].append(mic_oaspl)

    # === CONSOLE OUTPUT ===
    print(f"✅ Pt {point}: Φ={phi:.3f}±{phi_std:.3f}, pRise={pRise:.3f}±{pRise_std:.3f}")

# ============================================================================
# SERIAL COMMUNICATION THREADS
# ============================================================================
# Three independent threads read from three Arduino devices in parallel.
# Each parses its own data format and stores in the point buffer when ready.

serial_ports = {'rpm': None, 'stats': None, 'suction_fan': None}  # Serial port handles


def read_rpm_serial():
    """
    Read RPM and motor power data from Arduino Nano 1 (COM4).
    
    Expected message format:
        DATA_PKT,<point>,<rpm>,<rpm_std>,<power_mW>,<power_std>
    
    Extracts:
    - Shaft RPM (measured from Hall effect sensor)
    - RPM standard deviation (from multiple revolutions)
    - Motor electrical power (mW)
    - Power standard deviation
    
    Stores in point buffer under 'rpm' key.
    """
    try:
        ser = serial.Serial(RPM_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to RPM port: {RPM_PORT}")
        serial_ports['rpm'] = ser

        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if not line:
                        continue
                    if line.startswith("DATA_PKT"):
                        # Parse comma-separated data packet
                        parts = line.split(",")
                        if len(parts) >= 6:
                            pt = int(parts[1])
                            data = {
                                'mean':     float(parts[2]),   # RPM mean
                                'stddev':   float(parts[3]),   # RPM std dev
                                'pwr_mean': float(parts[4]),   # Power in mW
                                'pwr_std':  float(parts[5]),   # Power std dev
                            }
                            print(f"RPM Pt {pt}: {data['mean']:.0f} RPM | {data['pwr_mean']:.1f} mW")
                            _store_in_buffer(pt, 'rpm', data)
                    else:
                        # Log any other messages (debug/status info)
                        print(f"RPM Log: {line}")
            except (ValueError, IndexError, UnicodeDecodeError):
                pass
            except serial.SerialException:
                break
            time.sleep(0.005)

    except serial.SerialException as e:
        print(f"❌ RPM serial error: {e}")


def read_stats_serial():
    """
    Read pressure and temperature data from Arduino ESP32 (COM5).
    
    Expected message format (multiline):
        Point: <point_number>
        <sensor_name>,<mean>,<stddev>
        <sensor_name>,<mean>,<stddev>
        ---------------------------
    
    Extracts:
    - Venturi pressure drop (dp_sensor0)
    - Stage pressure rise (dp_sensor1)
    - Temperature (°C)
    - All with standard deviations
    
    Stores in point buffer under 'pressure' key.
    Triggers end_alert() when point ≥ 70 (end of test).
    """
    try:
        ser = serial.Serial(STATS_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to Stats port: {STATS_PORT}")
        serial_ports['stats'] = ser

        temp_stats   = {}  # Buffer for multi-line message
        current_point = None

        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    
                    # Parse point number header
                    if line.startswith("Point: "):
                        current_point = int(line.split(": ")[1])
                        temp_stats    = {'point': current_point}
                        # Alert when test reaches final point
                        if current_point >= 70:
                            end_alert()
                    
                    # Parse sensor data lines (format: "name,mean,stddev")
                    elif ',' in line and not line.startswith("Variable"):
                        parts = line.split(',')
                        if len(parts) == 3:
                            var_name = parts[0].strip()
                            temp_stats[f"{var_name}_mean"]   = float(parts[1])
                            temp_stats[f"{var_name}_stddev"] = float(parts[2])
                    
                    # End of message marker: process accumulated stats
                    elif line.startswith("---------------------------"):
                        if temp_stats and 'point' in temp_stats:
                            pt = temp_stats.pop('point')
                            print(f"Pressure Pt {pt}: dp0={temp_stats.get('dp_sensor0_mean',0):.2f} Pa")
                            _store_in_buffer(pt, 'pressure', temp_stats)
                            temp_stats = {}
                            
            except (ValueError, IndexError, UnicodeDecodeError):
                print("⚠️ Malformed Stats line, skipping...")
            except serial.SerialException:
                break
            time.sleep(0.005)

    except serial.SerialException as e:
        print(f"❌ Stats serial error: {e}")


def read_suction_fan_serial():
    """
    Read throttle position data from Arduino Nano 2 (COM6).
    
    Expected message format:
        Point: <point_number>, Location: <position_value>
    
    Extracts:
    - Throttle/suction fan position (0-100 or 0-255 depending on controller)
    
    Stores in point buffer under 'throttle' key.
    """
    try:
        ser = serial.Serial(SUCTION_FAN_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to Suction Fan port: {SUCTION_FAN_PORT}")
        serial_ports['suction_fan'] = ser

        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line.startswith("Point: "):
                        # Parse comma-separated point and location
                        parts = line.split(", ")
                        pt       = int(parts[0].replace("Point: ", ""))
                        location = float(parts[1].replace("Location: ", ""))
                        data     = {'location': location}
                        print(f"Throttle Pt {pt}: location={location}")
                        _store_in_buffer(pt, 'throttle', data)
            except (ValueError, IndexError):
                print("⚠️ Malformed Throttle line, skipping...")
            time.sleep(0.005)

    except serial.SerialException as e:
        print(f"❌ Suction Fan serial error: {e}")


def start_test():
    """
    Send start command to all connected Arduino devices.
    
    Broadcasts "run_test\n" to RPM, pressure, and throttle controllers.
    Records timestamp when test actually starts (used for microphone sync).
    """
    global test_start_time
    command = "run_test\n"
    print(f"\nSending '{command.strip()}' to all devices...")
    for key, ser in serial_ports.items():
        if ser is not None and ser.is_open:
            try:
                ser.write(command.encode('utf-8'))
                print(f"-> Sent to {key}")
            except Exception as e:
                print(f"Error sending to {key}: {e}")
    test_start_time = time.time()
    print("-" * 30)


# ============================================================================
# REAL-TIME PLOTTING
# ============================================================================

def update_plot(frame):
    """
    Update all six subplots with latest data from the test.
    
    Called ~4 times per second (250ms interval) by FuncAnimation.
    Displays:
    
    [0,0] RPM vs time - Shows shaft speed with measurement uncertainty
    [0,1] Mass flow and axial velocity vs time - Flow measurements
    [1,0] Performance map - Pressure rise coefficient vs flow coefficient (Φ vs ψ curve)
    [1,1] Pressures and density vs time - Raw sensor measurements
    [2,0] Efficiency map - Efficiency vs flow coefficient
    [2,1] Noise map - OASPL vs flow coefficient
    
    All plots show error bars representing measurement uncertainty.
    """
    with buffer_lock:
        if len(plot_data['time']) < 2:
            return  # Wait for at least 2 points before plotting

        # Convert timestamps to relative time (elapsed seconds from start)
        times     = np.array(plot_data['time'])
        rel_times = times - times[0]

        # Clear all subplots for fresh redraw
        for ax in axs.flat:
            ax.clear()

        # Common error bar formatting
        err_style = {'fmt': 'x', 'capsize': 3, 'markersize': 4, 'alpha': 0.3}

        # ===== [0,0] RPM vs Time =====
        axs[0, 0].errorbar(rel_times, plot_data['rpm_mean'], yerr=plot_data['rpm_err'],
                           color='b', label='RPM', **err_style)
        axs[0, 0].set_ylabel('RPM')
        axs[0, 0].legend(loc='upper left')
        axs[0, 0].grid(True, alpha=0.7)

        # ===== [0,1] Flow Measurements vs Time =====
        # Mass flow rate (scaled by 1e3 for visibility)
        axs[0, 1].errorbar(rel_times, np.array(plot_data['mflow_mean']) * 1e3,
                           yerr=np.array(plot_data['mflow_err']) * 1e3,
                           color='r', label='Mass Flow*1e3', **err_style)
        # Axial velocity on same axes
        axs[0, 1].errorbar(rel_times, plot_data['axvel_mean'], yerr=plot_data['axvel_err'],
                           color='g', label='Axial Velocity', **err_style)
        axs[0, 1].legend(loc='upper left')
        axs[0, 1].set_ylabel('Mass Flow*1e3, Axial Vel')
        axs[0, 1].grid(True, alpha=0.7)

        # ===== [1,0] Performance Map (Φ vs ψ) =====
        # This is the key fan performance curve: flow coefficient vs pressure rise coefficient
        axs[1, 0].errorbar(plot_data['phi_mean'], plot_data['pRise_mean'],
                           xerr=plot_data['phi_err'], yerr=plot_data['pRise_err'],
                           color='k', label='Perf. Curve', **err_style)
        axs[1, 0].set_ylabel('Pressure Rise Coeff')
        axs[1, 0].set_xlabel('Flow Coeff ($\\Phi$)')
        axs[1, 0].grid(True, alpha=0.7)

        # ===== [1,1] Pressures and Density vs Time =====
        # Venturi pressure drop (for flow measurement)
        axs[1, 1].errorbar(rel_times, plot_data['dp_venturi_mean'], yerr=plot_data['dp_venturi_err'],
                           color='orange', label='$\\Delta P$ Venturi', **err_style)
        # Fan stage pressure rise
        axs[1, 1].errorbar(rel_times, plot_data['dp_stage_mean'], yerr=plot_data['dp_stage_err'],
                           color='purple', label='$\\Delta P$ Stage', **err_style)
        # Air density (scaled by 10 for visibility)
        axs[1, 1].errorbar(rel_times, np.array(plot_data['rho_mean']) * 10,
                           yerr=np.array(plot_data['rho_err']) * 10,
                           color='brown', label='rho*10', **err_style)
        axs[1, 1].set_ylabel('Pressure (Pa), Density*10')
        axs[1, 1].legend(loc='upper left')
        axs[1, 1].grid(True, alpha=0.7)

        # ===== [2,0] Efficiency Map (Φ vs η) =====
        # How efficiently the fan converts motor power to aerodynamic power
        axs[2, 0].errorbar(plot_data['phi_mean'], plot_data['efficiency_mean'],
                           xerr=plot_data['phi_err'], yerr=plot_data['efficiency_err'],
                           color='b', label='Efficiency', **err_style)
        axs[2, 0].set_ylabel('Efficiency')
        axs[2, 0].set_xlabel('Flow Coeff ($\\Phi$)')
        axs[2, 0].legend(loc='upper left')
        axs[2, 0].grid(True, alpha=0.7)
        
        # ===== [2,1] Noise Map (Φ vs OASPL) =====
        # Fan acoustic performance - Overall A-weighted sound pressure level
        axs[2, 1].scatter(plot_data['phi_mean'], plot_data['oaspl'],
                           color='r', label='OASPL', marker = 'x')
        axs[2, 1].set_ylabel('OASPL')
        axs[2, 1].set_xlabel('Flow Coeff ($\\Phi$)')
        axs[2, 1].legend(loc='upper left')
        axs[2, 1].grid(True, alpha=0.7)


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for the data acquisition system.
    
    Workflow:
    1. Prompt user for test type (DOE, inverse design, or standard)
    2. Set output file paths accordingly
    3. Initialize CSV logging and microphone
    4. Start all communication threads (RPM, pressure, throttle, microphone)
    5. Send start command to all Arduino devices
    6. Display live updating plots
    7. On exit: close files, save audio, close serial connections
    """

    print("Starting Scientific Data Logger...")

    # ===== USER INPUT: Test Type Classification =====
    # Determines output directory and file naming
    entry    = False
    doe_bool = False
    inverse_design_bool = False
    
    while not entry:
        doe = input("Is this a DOE test? (y/n) ")
        if doe == "y":
            entry = True
            try:
                # Design of Experiments test - use STL file naming convention
                blade_number = int(input("What blade is this? (Enter integer) "))
                doe_bool = True
                stl_file = None
                
                # Search for matching STL file in stl_files/ directory
                for file in os.listdir("stl_files"):
                    match = re.search(f"DOE_{blade_number}", file)
                    if match:
                        parts = file.split(sep=".")
                        if parts[-1] == "stl":
                            stl_file = file.rstrip(".stl")
                
                if stl_file is None:
                    raise ValueError("No file found")
                
                # Set output paths to DOE data directories
                CSV_FILENAME   = f'data/doe_data/{stl_file}.csv'
                AUDIO_FILENAME = f'audio/doe_data/{stl_file}.wav'
                print(CSV_FILENAME)
                
            except TypeError:
                print("invalid entry")
                entry = False
                
        elif doe == "n":
            # Check for inverse design test
            inverse_design = input("Is this inverse design? (y/n) ")
            if inverse_design == "y":
                entry = True
                try:
                    # Inverse design test
                    inverse_design_bool = True
                    inverse_blade_number = int(input("What blade is this? (Enter integer) "))
                    stl_file = None
                    
                    # Search for matching inverse design STL file
                    for file in os.listdir("stl_files"):
                        match = re.search(f"inverse_design_test_{inverse_blade_number}", file)
                        if match:
                            parts = file.split(sep=".")
                            if parts[-1] == "stl":
                                stl_file = file.rstrip(".stl")
                    
                    if stl_file is None:
                        raise ValueError("No file found")
                    
                    # Set output paths to inverse design directories
                    CSV_FILENAME   = f'data/inverse_design_2/{stl_file}.csv'
                    AUDIO_FILENAME = f'audio/inverse_design_2/{stl_file}.wav'
                    print(CSV_FILENAME)
                    print(AUDIO_FILENAME)
                    
                except TypeError:
                    print("invalid entry")
                    entry = False
            else:
                # Standard test (not DOE, not inverse design)
                entry = True
        else:
            print("Enter a valid response (y/n)")

    # ===== SET OUTPUT FILENAMES =====
    # For standard tests, allow user to tag the filename with blade type
    if not doe_bool and not inverse_design_bool:
        blade = input("Enter blade type: ")
        if blade != "":
            CSV_FILENAME   = f'data/test_data_{blade}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            AUDIO_FILENAME = f'audio/recording_{blade}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'

    print(f"[SAVE FILE] set to {CSV_FILENAME}")
    
    # ===== INITIALIZATION =====
    init_csv()
    init_microphone()

    # ===== START COMMUNICATION THREADS =====
    # All three Arduino connections run in parallel background threads
    t1    = threading.Thread(target=read_rpm_serial,       daemon=True)
    t2    = threading.Thread(target=read_stats_serial,     daemon=True)
    t3    = threading.Thread(target=read_suction_fan_serial, daemon=True)
    t_mic = threading.Thread(target=microphone_recording_thread, daemon=True)

    t1.start()
    t2.start()
    t3.start()
    t_mic.start()

    # Wait for serial connections to establish
    print("Waiting for connections...")
    time.sleep(3)
    
    # Send test start command to all devices
    start_test()

    # ===== SETUP PLOTTING =====
    # Create 3x2 subplot layout for real-time visualization
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(3, 2, figsize=(14, 8))
    
    # FuncAnimation redraws plots every 250ms as data arrives
    ani = FuncAnimation(fig, update_plot, interval=250, cache_frame_data=False)

    try:
        plt.tight_layout()
        print("Opening Plot Window...")
        plt.show()  # Blocking call - continues until user closes plot window
        
    except KeyboardInterrupt:
        print("\nStopping...")
        
    finally:
        # ===== CLEANUP =====
        # Close CSV file
        csv_file.close()
        
        # Save concatenated audio recordings to WAV file
        if mic_recordings:
            full_recording = np.concatenate(mic_recordings, axis=0)
            wavfile.write(AUDIO_FILENAME, mic_analyser.sample_rate, full_recording)
            print(f"Saved {len(mic_recordings)} sections to {AUDIO_FILENAME}")
        
        # Close all serial connections and send stop command
        for s in serial_ports.values():
            if s and s.is_open:
                s.write(b"stop\n")  # Tell Arduino to stop test
                s.close()
        
        print(f"Data saved to {CSV_FILENAME}")