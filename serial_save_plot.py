import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import serial
import threading
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from datetime import datetime
import numpy as np
import os
import re
import scipy.io.wavfile as wavfile

from microphone_recorder import MicrophoneAnalyzer
from unused_files.live_plotter import run_live_plot
from noise import end_alert

# Configuration
RPM_PORT          = 'COM4'
STATS_PORT        = 'COM5'
SUCTION_FAN_PORT  = 'COM6   '
BAUDRATE          = 115200
CSV_FILENAME      = f'data/test_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
AUDIO_FILENAME    = f'audio/recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'

# Physical constants
C       = 0.973
epsilon = 1
d       = 0.050   # m (Venturi throat diameter)
D       = 0.080   # m (Duct diameter)
beta    = d / D
E       = (1 - beta**4) ** (-1.0 / 2.0)
rmid    = 0.0275  # m (Fan mid-radius)
h       = 0.024   # m (Fan blade height)
Afan    = 2 * np.pi * rmid * h
Athroat = np.pi * d**2 / 4

## CHANGE THIS BASED ON ATM CONDITIONS
pressureAtm = 1.01252 * 10**5
TAtm        = 273.15 + 20
#######
rho_default    = pressureAtm / 287 / TAtm
RPM_TO_U_CONV  = rmid * np.pi / 30

# ---------------------------------------------------------------------------
# Point-keyed data buffer
# Each entry is a dict that accumulates data from all three sources.
# Once all three keys are present the point is processed and removed.
# ---------------------------------------------------------------------------
point_buffer = {}   # { point_index: { 'rpm': {...}, 'pressure': {...}, 'throttle': {...}, 'mic': {...} } }
buffer_lock  = threading.Lock()

# Mic
mic_data             = {'point': None, 'oaspl': None}
mic_analyser         = None
test_start_time      = None
mic_lock             = threading.Lock()

# Plotting
MAX_PLOT_POINTS = 250
plot_data = {
    'time':          deque(maxlen=MAX_PLOT_POINTS),
    'rpm_mean':      deque(maxlen=MAX_PLOT_POINTS),
    'mflow_mean':    deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_mean': deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_mean': deque(maxlen=MAX_PLOT_POINTS),
    'phi_mean':      deque(maxlen=MAX_PLOT_POINTS),
    'pRise_mean':    deque(maxlen=MAX_PLOT_POINTS),
    'rho_mean':      deque(maxlen=MAX_PLOT_POINTS),
    'axvel_mean':    deque(maxlen=MAX_PLOT_POINTS),
    'rpm_err':       deque(maxlen=MAX_PLOT_POINTS),
    'mflow_err':     deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_err':deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_err':  deque(maxlen=MAX_PLOT_POINTS),
    'phi_err':       deque(maxlen=MAX_PLOT_POINTS),
    'pRise_err':     deque(maxlen=MAX_PLOT_POINTS),
    'rho_err':       deque(maxlen=MAX_PLOT_POINTS),
    'axvel_err':     deque(maxlen=MAX_PLOT_POINTS),
    'efficiency_mean': deque(maxlen=MAX_PLOT_POINTS),
    'efficiency_err':  deque(maxlen=MAX_PLOT_POINTS),
    'oaspl':         deque(maxlen=MAX_PLOT_POINTS),
}

csv_file   = None
csv_writer = None


def init_csv():
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
    global mic_analyser, mic_recordings
    try:
        mic_analyser          = MicrophoneAnalyzer()
        mic_analyser.duration = 2
        mic_analyser.device_id = None
        mic_recordings        = []
        print("✅ Microphone ready")
        return True
    except Exception as e:
        print(f"❌ Microphone init failed: {e}")
        return False


def microphone_recording_thread():
    global test_start_time, mic_analyser
    if not mic_analyser:
        return
    print("✅ Connected to microphone analyser")

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
                        mic_data['oaspl'] = mic_analyser.overall_a_spl()
                    mic_recordings.append(mic_analyser.audio_data.copy())
                    # Store in point buffer
                    _store_in_buffer(point_count, 'mic', {'oaspl': mic_data['oaspl']})
                    print(f"✅ Audio recorded for point {point_count}\n")
                point_count += 1

            time.sleep(0.01)
        except Exception as e:
            print(f"⚠️ Microphone thread error: {e}")
            time.sleep(1)


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def _store_in_buffer(point, source, data):
    """Store data from one source into the point buffer and check if complete."""
    with buffer_lock:
        if point not in point_buffer:
            point_buffer[point] = {}
        point_buffer[point][source] = data

        # Check if all three required sources have arrived
        entry = point_buffer[point]
        if 'rpm' in entry and 'pressure' in entry and 'throttle' in entry and 'mic' in entry:
            _process_complete_point(point, entry)
            del point_buffer[point]


def _process_complete_point(point, entry):
    """Called (inside buffer_lock) when all sources for a point have arrived."""
    rpm      = entry['rpm']
    pressure = entry['pressure']
    throttle = entry['throttle']
    mic_oaspl = entry['mic'].get('oaspl', -100)

    rpm_mean  = rpm.get('mean', 0)
    rpm_std   = rpm.get('stddev', 0)
    power_mean = rpm.get('pwr_mean', 0)
    power_std  = rpm.get('pwr_std', 0)

    throttle_location = throttle.get('location', 0)

    dp_sensor0_mean = pressure.get('dp_sensor0_mean', 0)
    dp_sensor0_std  = pressure.get('dp_sensor0_stddev', 0)
    dp_sensor1_mean = pressure.get('dp_sensor1_mean', 0)
    dp_sensor1_std  = pressure.get('dp_sensor1_stddev', 0)
    temp_mean       = pressure.get('temp_mean', 15)
    temp_std        = pressure.get('temp_stddev', 0)

    # Derived values
    U      = RPM_TO_U_CONV * rpm_mean
    U_sq   = U ** 2
    T_abs  = 273.15 + temp_mean
    rho_mean = pressureAtm / (287 * T_abs)

    if dp_sensor0_mean < 0:
        axvel_throat_mean = 0
    else:
        axvel_throat_mean = np.sqrt(dp_sensor0_mean / (0.5 * rho_mean))
    mdot_mean         = rho_mean * Athroat * axvel_throat_mean
    axvel_mean        = mdot_mean / (Afan * rho_mean)
    dp_stage_mean     = dp_sensor1_mean
    dp_venturi_mean   = dp_sensor0_mean
    dp_venturi_std    = dp_sensor0_std

    phi   = axvel_mean / U if abs(U) > 1e-6 else 0.0
    pRise = dp_stage_mean / (rho_mean * U_sq) if (U_sq > 1e-6 and rho_mean > 0) else 0.0

    aero_power_mean = dp_stage_mean * mdot_mean / rho_mean
    efficiency_mean = aero_power_mean / (power_mean / 1000) if power_mean > 0 else 0.0

    # Error propagation
    rho_std = rho_mean * (temp_std / T_abs) if T_abs != 0 else 0

    rel_var_dp0        = (dp_sensor0_std / dp_sensor0_mean) ** 2 if dp_sensor0_mean != 0 else 0
    rel_var_rho_local  = (rho_std / rho_mean) ** 2 if rho_mean != 0 else 0
    axvel_throat_std   = axvel_throat_mean * np.sqrt(0.25 * rel_var_dp0 + 0.25 * rel_var_rho_local) if axvel_throat_mean != 0 else 0

    rel_var_axvel = (axvel_throat_std / axvel_throat_mean) ** 2 if axvel_throat_mean != 0 else 0
    mdot_std      = mdot_mean * np.sqrt(rel_var_rho_local + rel_var_axvel) if mdot_mean != 0 else 0
    axvel_std     = axvel_mean * np.sqrt(rel_var_rho_local + (mdot_std / mdot_mean) ** 2) if axvel_mean != 0 else 0

    rel_var_rpm = (rpm_std / rpm_mean) ** 2 if rpm_mean != 0 else 0
    rel_var_U   = rel_var_rpm
    rel_var_rho = (rho_std / rho_mean) ** 2 if rho_mean != 0 else 0

    term1_sq   = dp_sensor1_std ** 2
    term2_sq   = (0.5 * axvel_mean ** 2 * rho_std) ** 2
    term3_sq   = (rho_mean * axvel_mean * axvel_std) ** 2
    dp_stage_std = np.sqrt(term1_sq + term2_sq + term3_sq)

    rel_var_vx = (axvel_std / axvel_mean) ** 2 if axvel_mean != 0 else 0
    phi_std    = abs(phi) * np.sqrt(rel_var_vx + rel_var_U)

    rel_var_dp = (dp_stage_std / dp_stage_mean) ** 2 if dp_stage_mean != 0 else 0
    pRise_std  = abs(pRise) * np.sqrt(rel_var_dp + rel_var_rho + 4 * rel_var_U)

    rel_std_dp       = dp_stage_std / dp_stage_mean if dp_stage_mean != 0 else 0
    rel_std_mdot     = mdot_std / mdot_mean if mdot_mean != 0 else 0
    rel_std_rho      = rho_std / rho_mean if rho_mean != 0 else 0
    rel_std_power    = power_std / power_mean if power_mean != 0 else 0
    rel_std_eff      = np.sqrt(rel_std_dp**2 + rel_std_mdot**2 + rel_std_rho**2 + rel_std_power**2)
    efficiency_std   = efficiency_mean * rel_std_eff

    # Save to CSV
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

    # Update plot
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

    print(f"✅ Pt {point}: Φ={phi:.3f}±{phi_std:.3f}, pRise={pRise:.3f}±{pRise_std:.3f}")


# ---------------------------------------------------------------------------
# Serial reading threads
# ---------------------------------------------------------------------------

serial_ports = {'rpm': None, 'stats': None, 'suction_fan': None}


def read_rpm_serial():
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
                        parts = line.split(",")
                        if len(parts) >= 6:
                            pt = int(parts[1])
                            data = {
                                'mean':     float(parts[2]),
                                'stddev':   float(parts[3]),
                                'pwr_mean': float(parts[4]),
                                'pwr_std':  float(parts[5]),
                            }
                            print(f"RPM Pt {pt}: {data['mean']:.0f} RPM | {data['pwr_mean']:.1f} mW")
                            _store_in_buffer(pt, 'rpm', data)
                    else:
                        print(f"RPM Log: {line}")
            except (ValueError, IndexError, UnicodeDecodeError):
                pass
            except serial.SerialException:
                break
            time.sleep(0.005)

    except serial.SerialException as e:
        print(f"❌ RPM serial error: {e}")


def read_stats_serial():
    try:
        ser = serial.Serial(STATS_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to Stats port: {STATS_PORT}")
        serial_ports['stats'] = ser

        temp_stats   = {}
        current_point = None

        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line.startswith("Point: "):
                        current_point = int(line.split(": ")[1])
                        temp_stats    = {'point': current_point}
                        if current_point >= 70:
                            end_alert()
                    elif ',' in line and not line.startswith("Variable"):
                        parts = line.split(',')
                        if len(parts) == 3:
                            var_name = parts[0].strip()
                            temp_stats[f"{var_name}_mean"]   = float(parts[1])
                            temp_stats[f"{var_name}_stddev"] = float(parts[2])
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
    try:
        ser = serial.Serial(SUCTION_FAN_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to Suction Fan port: {SUCTION_FAN_PORT}")
        serial_ports['suction_fan'] = ser

        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line.startswith("Point: "):
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def update_plot(frame):
    with buffer_lock:
        if len(plot_data['time']) < 2:
            return

        times     = np.array(plot_data['time'])
        rel_times = times - times[0]

        for ax in axs.flat:
            ax.clear()

        err_style = {'fmt': 'x', 'capsize': 3, 'markersize': 4, 'alpha': 0.3}

        axs[0, 0].errorbar(rel_times, plot_data['rpm_mean'], yerr=plot_data['rpm_err'],
                           color='b', label='RPM', **err_style)
        axs[0, 0].set_ylabel('RPM')
        axs[0, 0].legend(loc='upper left')
        axs[0, 0].grid(True, alpha=0.7)

        axs[0, 1].errorbar(rel_times, np.array(plot_data['mflow_mean']) * 1e3,
                           yerr=np.array(plot_data['mflow_err']) * 1e3,
                           color='r', label='Mass Flow*1e3', **err_style)
        axs[0, 1].errorbar(rel_times, plot_data['axvel_mean'], yerr=plot_data['axvel_err'],
                           color='g', label='Axial Velocity', **err_style)
        axs[0, 1].legend(loc='upper left')
        axs[0, 1].set_ylabel('Mass Flow*1e3, Axial Vel')
        axs[0, 1].grid(True, alpha=0.7)

        axs[1, 0].errorbar(plot_data['phi_mean'], plot_data['pRise_mean'],
                           xerr=plot_data['phi_err'], yerr=plot_data['pRise_err'],
                           color='k', label='Perf. Curve', **err_style)
        axs[1, 0].set_ylabel('Pressure Rise Coeff')
        axs[1, 0].set_xlabel('Flow Coeff ($\\Phi$)')
        axs[1, 0].grid(True, alpha=0.7)

        axs[1, 1].errorbar(rel_times, plot_data['dp_venturi_mean'], yerr=plot_data['dp_venturi_err'],
                           color='orange', label='$\\Delta P$ Venturi', **err_style)
        axs[1, 1].errorbar(rel_times, plot_data['dp_stage_mean'], yerr=plot_data['dp_stage_err'],
                           color='purple', label='$\\Delta P$ Stage', **err_style)
        axs[1, 1].errorbar(rel_times, np.array(plot_data['rho_mean']) * 10,
                           yerr=np.array(plot_data['rho_err']) * 10,
                           color='brown', label='rho*10', **err_style)
        axs[1, 1].set_ylabel('Pressure (Pa), Density*10')
        axs[1, 1].legend(loc='upper left')
        axs[1, 1].grid(True, alpha=0.7)

        axs[2, 0].errorbar(plot_data['phi_mean'], plot_data['efficiency_mean'],
                           xerr=plot_data['phi_err'], yerr=plot_data['efficiency_err'],
                           color='b', label='Efficiency', **err_style)
        axs[2, 0].set_ylabel('Efficiency')
        axs[2, 0].set_xlabel('Flow Coeff ($\\Phi$)')
        axs[2, 0].legend(loc='upper left')
        axs[2, 0].grid(True, alpha=0.7)
        
        axs[2, 1].scatter(plot_data['phi_mean'], plot_data['oaspl'],
                           color='r', label='OASPL', marker = 'x')
        axs[2, 1].set_ylabel('OASPL')
        axs[2, 1].set_xlabel('Flow Coeff ($\\Phi$)')
        axs[2, 1].legend(loc='upper left')
        axs[2, 1].grid(True, alpha=0.7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("Starting Scientific Data Logger...")

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
                CSV_FILENAME   = f'data/doe_data/{stl_file}.csv'
                AUDIO_FILENAME = f'audio/doe_data/{stl_file}.wav'
                print(CSV_FILENAME)
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
                            # print(parts)
                            if parts[-1] == "stl":
                                stl_file = file.rstrip(".stl")
                    if stl_file is None:
                        raise ValueError("No file found")
                    CSV_FILENAME   = f'data/inverse_design_2/{stl_file}.csv'
                    AUDIO_FILENAME = f'audio/inverse_design_2/{stl_file}.wav'
                    print(CSV_FILENAME)
                    print(AUDIO_FILENAME)
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
            CSV_FILENAME   = f'data/test_data_{blade}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            AUDIO_FILENAME = f'audio/recording_{blade}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'

    print(f"[SAVE FILE] set to {CSV_FILENAME}")
    init_csv()
    init_microphone()

    # Start threads
    t1    = threading.Thread(target=read_rpm_serial,       daemon=True)
    t2    = threading.Thread(target=read_stats_serial,     daemon=True)
    t3    = threading.Thread(target=read_suction_fan_serial, daemon=True)
    t_mic = threading.Thread(target=microphone_recording_thread, daemon=True)

    t1.start()
    t2.start()
    t3.start()
    t_mic.start()

    print("Waiting for connections...")
    time.sleep(3)
    start_test()

    # Setup Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(3, 2, figsize=(14, 8))
    ani = FuncAnimation(fig, update_plot, interval=250, cache_frame_data=False)

    try:
        plt.tight_layout()
        print("Opening Plot Window...")
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        csv_file.close()
        # fig.savefig(f"figs/dataplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        if mic_recordings:
            full_recording = np.concatenate(mic_recordings, axis=0)
            wavfile.write(AUDIO_FILENAME, mic_analyser.sample_rate, full_recording)
            print(f"Saved {len(mic_recordings)} sections to {AUDIO_FILENAME}")
        for s in serial_ports.values():
            if s and s.is_open:
                s.write(b"stop\n")
                s.close()
        print(f"Data saved to {CSV_FILENAME}")