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
RPM_PORT = 'COM3'  # Change to your RPM Arduino port
STATS_PORT = 'COM4'  # Change to your stats Arduino port
BAUDRATE = 9600
CSV_FILENAME = f'test_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Shared data structures
rpm_data = {'point': None, 'mean': None, 'stddev': None}
stats_buffer = {}
data_lock = threading.Lock()

# Data for plotting (store last N points)
MAX_PLOT_POINTS = 100
plot_data = {
    'time': deque(maxlen=MAX_PLOT_POINTS),
    'rpm_mean': deque(maxlen=MAX_PLOT_POINTS),
    'rpm_stddev': deque(maxlen=MAX_PLOT_POINTS),
    'mflow_mean': deque(maxlen=MAX_PLOT_POINTS),
    'mflow_stddev': deque(maxlen=MAX_PLOT_POINTS),
    'axvelocity_mean': deque(maxlen=MAX_PLOT_POINTS),
    'axvelocity_stddev': deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_mean': deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_stddev': deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_mean': deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_stddev': deque(maxlen=MAX_PLOT_POINTS),
    'rho_mean': deque(maxlen=MAX_PLOT_POINTS),
    'rho_stddev': deque(maxlen=MAX_PLOT_POINTS),
    'flow_coefficient_mean': deque(maxlen=MAX_PLOT_POINTS),
    'flow_coefficient_stddev': deque(maxlen=MAX_PLOT_POINTS),
    'pressure_rise_coefficient_mean': deque(maxlen=MAX_PLOT_POINTS),
    'pressure_rise_coefficient_stddev': deque(maxlen=MAX_PLOT_POINTS),
}

# CSV file setup
csv_file = None
csv_writer = None

def init_csv():
    global csv_file, csv_writer
    csv_file = open(CSV_FILENAME, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Write header
    csv_writer.writerow([
        'timestamp', 'point', 'target_rpm', 'smoothed_rpm', 'pwm',
        'mflow_mean', 'mflow_stddev',
        'axvelocity_mean', 'axvelocity_stddev',
        'dp_venturi_mean', 'dp_venturi_stddev',
        'dp_stage_mean', 'dp_stage_stddev',
        'flow_coefficient', 'pressure_ratio', 'efficiency'  # Processed data
    ])
    csv_file.flush()

def read_rpm_serial():
    """Read RPM data from first Arduino"""
    try:
        ser = serial.Serial(RPM_PORT, BAUDRATE, timeout=1)
        print(f"Connected to RPM port: {RPM_PORT}")
        
        current_point = None
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                
                # Parse: "Point X - Mean: Y.YY RPM, StdDev: Z.ZZ RPM"
                if line.startswith("Point ") and " - Mean: " in line:
                    try:
                        # Extract point number
                        point_part = line.split(" - ")[0]
                        current_point = int(point_part.replace("Point ", ""))
                        
                        # Extract mean and stddev
                        data_part = line.split(" - Mean: ")[1]
                        mean_part = data_part.split(" RPM, StdDev: ")[0]
                        stddev_part = data_part.split(" RPM, StdDev: ")[1].replace(" RPM", "")
                        
                        rpm_mean = float(mean_part)
                        rpm_stddev = float(stddev_part)
                        
                        with data_lock:
                            rpm_data['point'] = current_point
                            rpm_data['mean'] = rpm_mean
                            rpm_data['stddev'] = rpm_stddev
                        
                        print(f"RPM Point {current_point}: {rpm_mean:.2f} ± {rpm_stddev:.2f} RPM")
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing RPM line: {line} - {e}")
                        
    except serial.SerialException as e:
        print(f"Error with RPM serial port: {e}")

def read_stats_serial():
    """Read stats data from second Arduino"""
    try:
        ser = serial.Serial(STATS_PORT, BAUDRATE, timeout=1)
        print(f"Connected to Stats port: {STATS_PORT}")
        
        current_point = None
        temp_stats = {}
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                
                # Check for point number
                if line.startswith("Point: "):
                    current_point = int(line.split(": ")[1])
                    temp_stats = {'point': current_point}
                
                # Parse CSV data lines
                elif ',' in line and not line.startswith("Variable"):
                    parts = line.split(',')
                    if len(parts) == 3:
                        var_name = parts[0].strip()
                        mean_val = float(parts[1])
                        stddev_val = float(parts[2])
                        temp_stats[f"{var_name}_mean"] = mean_val
                        temp_stats[f"{var_name}_stddev"] = stddev_val
                
                # Check if we've completed a data point
                elif line.startswith("---------------------------"):
                    if temp_stats and 'point' in temp_stats:
                        process_and_save_data(temp_stats)
                        temp_stats = {}
                        
    except serial.SerialException as e:
        print(f"Error with Stats serial port: {e}")

def process_and_save_data(stats):
    """Process the data and save to CSV"""
    with data_lock:
        # Get current RPM data
        target_rpm = rpm_data.get('target', 0)
        smoothed_rpm = rpm_data.get('smoothed', 0)
        pwm = rpm_data.get('pwm', 0)
        
        # Extract stats
        mflow_mean = stats.get('mflow_mean', 0)
        mflow_std = stats.get('mflow_stddev', 0)
        axvel_mean = stats.get('axvelocity_mean', 0)
        axvel_std = stats.get('axvelocity_stddev', 0)
        dp_venturi_mean = stats.get('dp_venturi(P2-P3)_mean', 0)
        dp_venturi_std = stats.get('dp_venturi(P2-P3)_stddev', 0)
        dp_stage_mean = stats.get('dp_stage(P2-P1)_mean', 0)
        dp_stage_std = stats.get('dp_stage(P2-P1)_stddev', 0)
        
        # === PROCESSING CALCULATIONS ===
        # Example calculations - modify based on your actual needs
        
        # Flow coefficient (dimensionless flow parameter)
        # Typically: Φ = Q / (N * D^3) where Q is flow, N is RPM, D is diameter
        # Using mflow as proxy for flow rate
        flow_coefficient = mflow_mean / (smoothed_rpm + 1) if smoothed_rpm > 0 else 0
        
        # Pressure ratio across stage
        pressure_ratio = dp_stage_mean / (dp_venturi_mean + 1) if dp_venturi_mean != 0 else 0
        
        # Simplified efficiency estimate (modify based on your system)
        # Typically: η = (pressure rise * flow) / (torque * speed)
        # This is a placeholder - adjust to your actual efficiency calculation
        efficiency = (dp_stage_mean * mflow_mean) / (smoothed_rpm * pwm + 1) if (smoothed_rpm * pwm) > 0 else 0
        efficiency = min(efficiency * 100, 100)  # Convert to percentage, cap at 100%
        
        # Write to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = [
            timestamp, stats.get('point', 0),
            target_rpm, smoothed_rpm, pwm,
            mflow_mean, mflow_std,
            axvel_mean, axvel_std,
            dp_venturi_mean, dp_venturi_std,
            dp_stage_mean, dp_stage_std,
            flow_coefficient, pressure_ratio, efficiency
        ]
        
        csv_writer.writerow(row)
        csv_file.flush()
        
        # Update plot data
        plot_data['time'].append(time.time())
        plot_data['target_rpm'].append(target_rpm)
        plot_data['smoothed_rpm'].append(smoothed_rpm)
        plot_data['mflow_mean'].append(mflow_mean)
        plot_data['axvelocity_mean'].append(axvel_mean)
        plot_data['dp_venturi_mean'].append(dp_venturi_mean)
        plot_data['dp_stage_mean'].append(dp_stage_mean)
        
        print(f"Point {stats.get('point', 0)}: RPM={smoothed_rpm:.0f}, mflow={mflow_mean:.4f}, η={efficiency:.2f}%")

def update_plot(frame):
    """Update the live plot"""
    with data_lock:
        if len(plot_data['time']) < 2:
            return
        
        # Convert time to relative seconds
        times = np.array(plot_data['time'])
        rel_times = times - times[0]
        
        # Clear all subplots
        for ax in axs.flat:
            ax.clear()
        
        # Plot 1: RPM
        axs[0, 0].plot(rel_times, plot_data['target_rpm'], 'b--', label='Target RPM', alpha=0.7)
        axs[0, 0].plot(rel_times, plot_data['smoothed_rpm'], 'b-', label='Actual RPM', linewidth=2)
        axs[0, 0].set_ylabel('RPM')
        axs[0, 0].legend(loc='upper left')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Mass Flow
        axs[0, 1].plot(rel_times, plot_data['mflow_mean'], 'r-', linewidth=2)
        axs[0, 1].set_ylabel('Mass Flow')
        axs[0, 1].set_title('Mass Flow Rate')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Axial Velocity
        axs[1, 0].plot(rel_times, plot_data['axvelocity_mean'], 'g-', linewidth=2)
        axs[1, 0].set_ylabel('Axial Velocity')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Pressure Differentials
        axs[1, 1].plot(rel_times, plot_data['dp_venturi_mean'], 'orange', label='ΔP Venturi', linewidth=2)
        axs[1, 1].plot(rel_times, plot_data['dp_stage_mean'], 'purple', label='ΔP Stage', linewidth=2)
        axs[1, 1].set_ylabel('Pressure (Pa)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].legend(loc='upper left')
        axs[1, 1].grid(True, alpha=0.3)

# Main execution
if __name__ == "__main__":
    print("Starting dual serial data logger...")
    print(f"Data will be saved to: {CSV_FILENAME}")
    
    # Initialize CSV
    init_csv()
    
    # Start serial reading threads
    rpm_thread = threading.Thread(target=read_rpm_serial, daemon=True)
    stats_thread = threading.Thread(target=read_stats_serial, daemon=True)
    
    rpm_thread.start()
    stats_thread.start()
    
    print("Waiting for serial connections...")
    time.sleep(2)
    
    # Setup live plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Live Data Monitoring', fontsize=16)
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)
    
    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        csv_file.close()
        print(f"Data saved to {CSV_FILENAME}")