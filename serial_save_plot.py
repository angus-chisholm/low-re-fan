import matplotlib
# Force a backend that supports interactive windows
try:
    matplotlib.use('TkAgg') 
except:
    pass # Fallback to default if TkAgg isn't available
import serial
import threading
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from datetime import datetime
import numpy as np

# Configuration
RPM_PORT = 'COM6'
STATS_PORT = 'COM4'
SUCTION_FAN_PORT = 'COM5'
BAUDRATE = 115200
CSV_FILENAME = f'test_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Constants
R_MID_M = 0.0275
RPM_TO_U_CONV = R_MID_M * np.pi / 30 

# Shared data
rpm_data = {'point': None, 'mean': None, 'stddev': None}
stats_buffer = {}
data_lock = threading.Lock()

# Data for plotting
MAX_PLOT_POINTS = 100
plot_data = {
    'time': deque(maxlen=MAX_PLOT_POINTS),
    # Means
    'rpm_mean': deque(maxlen=MAX_PLOT_POINTS),
    'mflow_mean': deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_mean': deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_mean': deque(maxlen=MAX_PLOT_POINTS),
    'phi_mean': deque(maxlen=MAX_PLOT_POINTS), 
    'psi_mean': deque(maxlen=MAX_PLOT_POINTS), 
    # Standard Deviations (Errors)
    'rpm_err': deque(maxlen=MAX_PLOT_POINTS),
    'mflow_err': deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_err': deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_err': deque(maxlen=MAX_PLOT_POINTS),
    'phi_err': deque(maxlen=MAX_PLOT_POINTS),
    'psi_err': deque(maxlen=MAX_PLOT_POINTS),
}

csv_file = None
csv_writer = None

def init_csv():
    global csv_file, csv_writer
    csv_file = open(CSV_FILENAME, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'timestamp', 'point', 'rpm_mean', 'rpm_stddev',
        'mflow_mean', 'mflow_stddev',
        'axvelocity_mean', 'axvelocity_stddev',
        'dp_venturi_mean', 'dp_venturi_stddev',
        'dp_stage_mean', 'dp_stage_stddev',
        'flow_coefficient', 'pressure_rise_coefficient'
    ])
    csv_file.flush()

# --- SERIAL READING FUNCTIONS ---
def read_rpm_serial():
    try:
        ser = serial.Serial(RPM_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to RPM port: {RPM_PORT}")
        global serial_ports
        serial_ports['rpm'] = ser
        current_point = None
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith("Point ") and " - Mean: " in line:
                    try:
                        point_part = line.split(" - ")[0]
                        current_point = int(point_part.replace("Point ", ""))
                        data_part = line.split(" - Mean: ")[1]
                        mean_part = data_part.split(" RPM, StdDev: ")[0]
                        stddev_part = data_part.split(" RPM, StdDev: ")[1].replace(" RPM", "")
                        
                        with data_lock:
                            rpm_data['point'] = current_point
                            rpm_data['mean'] = float(mean_part)
                            rpm_data['stddev'] = float(stddev_part)
                    except (ValueError, IndexError):
                        pass
    except serial.SerialException:
        pass

def read_stats_serial():
    try:
        ser = serial.Serial(STATS_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to Stats port: {STATS_PORT}")
        global serial_ports
        serial_ports['stats'] = ser
        current_point = None
        temp_stats = {}
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith("Point: "):
                    current_point = int(line.split(": ")[1])
                    temp_stats = {'point': current_point}
                elif ',' in line and not line.startswith("Variable"):
                    parts = line.split(',')
                    if len(parts) == 3:
                        var_name = parts[0].strip()
                        temp_stats[f"{var_name}_mean"] = float(parts[1])
                        temp_stats[f"{var_name}_stddev"] = float(parts[2])
                elif line.startswith("---------------------------"):
                    if temp_stats and 'point' in temp_stats:
                        process_and_save_data(temp_stats)
                        temp_stats = {}
    except serial.SerialException:
        pass

def read_suction_fan_serial():
    try:
        ser = serial.Serial(SUCTION_FAN_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to Suction Fan port: {SUCTION_FAN_PORT}")
        global serial_ports
        serial_ports['suction_fan'] = ser
        while True:
            if ser.in_waiting > 0:
                # Just drain the buffer or print debug info
                ser.readline()
    except serial.SerialException:
        pass

def start_test():
    command = "run_test\n" 
    print(f"\nSending command '{command.strip()}' to all connected devices...")
    global serial_ports
    for key, ser in serial_ports.items():
        if ser is not None and ser.is_open:
            try:
                ser.write(command.encode('utf-8'))
                print(f"-> Sent to {key}")
            except Exception as e:
                print(f"Error {key}: {e}")
    print("-" * 30)

# --- DATA PROCESSING & ERROR PROPAGATION ---
def process_and_save_data(stats):
    with data_lock:
        # 1. Retrieve Raw Means and StdDevs
        rpm_mean = rpm_data.get('mean', 0)
        rpm_std = rpm_data.get('stddev', 0)
        
        rho_mean = stats.get('rho_mean', 1.225)
        rho_std = stats.get('rho_stddev', 0)
        
        mflow_mean = stats.get('mflow_mean', 0)
        mflow_std = stats.get('mflow_stddev', 0)
        
        axvel_mean = stats.get('axvelocity_mean', 0)
        axvel_std = stats.get('axvelocity_stddev', 0)
        
        dp_venturi_mean = stats.get('dp_venturi(P2-P3)_mean', 0)
        dp_venturi_std = stats.get('dp_venturi(P2-P3)_stddev', 0)
        
        dp_stage_mean = stats.get('dp_stage(P2-P1)_mean', 0)
        dp_stage_std = stats.get('dp_stage(P2-P1)_stddev', 0)
        
        # 2. Calculate Derived Values
        U = RPM_TO_U_CONV * rpm_mean
        U_sq = U**2
        
        # Avoid division by zero
        phi = axvel_mean / U if abs(U) > 1e-6 else 0.0
        psi = dp_stage_mean / (rho_mean * U_sq) if (U_sq > 1e-6 and rho_mean > 0) else 0.0
        
        # 3. Calculate Error Propagation (Standard Deviations)
        
        # Relative errors (squared)
        # If mean is 0, assume relative error contribution is 0 to prevent crash
        rel_var_rpm = (rpm_std / rpm_mean)**2 if rpm_mean != 0 else 0
        rel_var_U = rel_var_rpm # Since U is linear with RPM
        rel_var_rho = (rho_std / rho_mean)**2 if rho_mean != 0 else 0
        
        # Error in Phi (Φ = Vx / U)
        # sigma_phi = phi * sqrt( (sigma_vx/vx)^2 + (sigma_U/U)^2 )
        rel_var_vx = (axvel_std / axvel_mean)**2 if axvel_mean != 0 else 0
        phi_std = abs(phi) * np.sqrt(rel_var_vx + rel_var_U)
        
        # Error in Psi (Ψ = dP / (rho * U^2))
        # sigma_psi = psi * sqrt( (sigma_dp/dp)^2 + (sigma_rho/rho)^2 + (2*sigma_U/U)^2 )
        rel_var_dp = (dp_stage_std / dp_stage_mean)**2 if dp_stage_mean != 0 else 0
        psi_std = abs(psi) * np.sqrt(rel_var_dp + rel_var_rho + (4 * rel_var_U)) # 4 comes from (2*sigma_U/U)^2

        # 4. Save to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([
            timestamp, stats.get('point', 0), rpm_mean, rpm_std,
            mflow_mean, mflow_std, axvel_mean, axvel_std,
            dp_venturi_mean, dp_venturi_std, dp_stage_mean, dp_stage_std,
            phi, psi
        ])
        csv_file.flush()
        
        # 5. Update Plot Arrays
        plot_data['time'].append(time.time())
        plot_data['rpm_mean'].append(rpm_mean)
        plot_data['rpm_err'].append(rpm_std)
        
        plot_data['mflow_mean'].append(mflow_mean)
        plot_data['mflow_err'].append(mflow_std)
        
        plot_data['dp_venturi_mean'].append(dp_venturi_mean)
        plot_data['dp_venturi_err'].append(dp_venturi_std)
        plot_data['dp_stage_mean'].append(dp_stage_mean)
        plot_data['dp_stage_err'].append(dp_stage_std)
        
        plot_data['phi_mean'].append(phi)
        plot_data['phi_err'].append(phi_std)
        plot_data['psi_mean'].append(psi)
        plot_data['psi_err'].append(psi_std)

        print(f"Pt {stats.get('point',0)}: Φ={phi:.3f}±{phi_std:.3f}, Ψ={psi:.3f}±{psi_std:.3f}")

def update_plot(frame):
    with data_lock:
        if len(plot_data['time']) < 2:
            return
        
        times = np.array(plot_data['time'])
        rel_times = times - times[0]
        
        for ax in axs.flat:
            ax.clear()
        
        # Style for error bars
        err_style = {'fmt': 'o-', 'capsize': 3, 'markersize': 4, 'alpha': 0.8}
        
        # Plot 1: RPM
        axs[0, 0].errorbar(rel_times, plot_data['rpm_mean'], yerr=plot_data['rpm_err'], 
                           color='b', label='RPM', **err_style)
        axs[0, 0].set_ylabel('RPM')
        axs[0, 0].legend(loc='upper left')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Mass Flow
        axs[0, 1].errorbar(rel_times, plot_data['mflow_mean'], yerr=plot_data['mflow_err'], 
                           color='r', label='Mass Flow', **err_style)
        axs[0, 1].set_ylabel('Mass Flow')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance Curve (XY Error Bars)
        axs[1, 0].errorbar(plot_data['phi_mean'], plot_data['psi_mean'], 
                           xerr=plot_data['phi_err'], yerr=plot_data['psi_err'],
                           color='k', label='Perf. Curve', **err_style)
        axs[1, 0].set_ylabel('Pressure Rise Coeff ($\Psi$)')
        axs[1, 0].set_xlabel('Flow Coeff ($\Phi$)')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Pressures
        axs[1, 1].errorbar(rel_times, plot_data['dp_venturi_mean'], yerr=plot_data['dp_venturi_err'], 
                           color='orange', label='$\Delta P$ Venturi', **err_style)
        axs[1, 1].errorbar(rel_times, plot_data['dp_stage_mean'], yerr=plot_data['dp_stage_err'], 
                           color='purple', label='$\Delta P$ Stage', **err_style)
        axs[1, 1].set_ylabel('Pressure (Pa)')
        axs[1, 1].legend(loc='upper left')
        axs[1, 1].grid(True, alpha=0.3)

if __name__ == "__main__":
    global serial_ports
    serial_ports = {'rpm': None, 'stats': None, 'suction_fan': None}
    
    print("Starting Scientific Data Logger with Error Propagation...")
    init_csv()
    
    # Start threads
    threading.Thread(target=read_rpm_serial, daemon=True).start()
    threading.Thread(target=read_stats_serial, daemon=True).start()
    threading.Thread(target=read_suction_fan_serial, daemon=True).start()
    
    time.sleep(5)
    start_test()
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    ani = FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)
    
    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        csv_file.close()
        fig.savefig(f"dataplot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png")
        for s in serial_ports.values():
            if s and s.is_open: s.close()
        print(f"Data saved to {CSV_FILENAME}")