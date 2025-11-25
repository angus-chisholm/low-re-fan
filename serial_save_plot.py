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
SUCTION_FAN_PORT = 'COM7'
BAUDRATE = 115200
CSV_FILENAME = f'data/test_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Constants
C = 0.973
epsilon = 1
d = 0.050  # m (Venturi throat diameter)
D = 0.080  # m (Duct diameter)
beta = d / D
E = (1 - beta**4) ** (-1.0 / 2.0)
rmid = 0.0275  # m (Fan mid-radius)
h = 0.024  # m (Fan blade height)
Afan = 2 * np.pi * rmid * h  # m^2 (Fan flow area)
Athroat = np.pi * d**2 / 4

## CHANGE THIS BASED ON ATM CONDITIONS
pressureAtm = 1.01167 * 10**5 
TAtm = 273.15 + 11.6
#######
rho_default = pressureAtm / 287 / TAtm

RPM_TO_U_CONV = rmid * np.pi / 30 

# Shared data
rpm_data = {'point': None, 'mean': None, 'stddev': None}
stats_buffer = {}
data_lock = threading.Lock()

# Data for plotting
MAX_PLOT_POINTS = 250
plot_data = {
    'time': deque(maxlen=MAX_PLOT_POINTS),
    # Means
    'rpm_mean': deque(maxlen=MAX_PLOT_POINTS),
    'mflow_mean': deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_mean': deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_mean': deque(maxlen=MAX_PLOT_POINTS),
    'phi_mean': deque(maxlen=MAX_PLOT_POINTS), 
    'pRise_mean': deque(maxlen=MAX_PLOT_POINTS), 
    'rho_mean': deque(maxlen=MAX_PLOT_POINTS),
    'axvel_mean': deque(maxlen=MAX_PLOT_POINTS),
    # Standard Deviations (Errors)
    'rpm_err': deque(maxlen=MAX_PLOT_POINTS),
    'mflow_err': deque(maxlen=MAX_PLOT_POINTS),
    'dp_venturi_err': deque(maxlen=MAX_PLOT_POINTS),
    'dp_stage_err': deque(maxlen=MAX_PLOT_POINTS),
    'phi_err': deque(maxlen=MAX_PLOT_POINTS),
    'pRise_err': deque(maxlen=MAX_PLOT_POINTS),
    'rho_err': deque(maxlen=MAX_PLOT_POINTS),
    'axvel_err': deque(maxlen=MAX_PLOT_POINTS),
}

csv_file = None
csv_writer = None

def init_csv():
    global csv_file, csv_writer
    csv_file = open(CSV_FILENAME, 'w', newline='')
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
    ])
    csv_file.flush()

# --- SERIAL READING FUNCTIONS (FIXED CPU HOGGING) ---
def read_rpm_serial():
    try:
        ser = serial.Serial(RPM_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to RPM port: {RPM_PORT}")
        global serial_ports
        serial_ports['rpm'] = ser
        
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line.startswith("Point ") and " - Mean: " in line:
                        # ... (Parsing logic remains the same) ...
                        point_part = line.split(" - ")[0]
                        current_point = int(point_part.replace("Point ", ""))
                        data_part = line.split(" - Mean: ")[1]
                        mean_part = data_part.split(" RPM, StdDev: ")[0]
                        stddev_part = data_part.split(" RPM, StdDev: ")[1].replace(" RPM", "")
                        
                        with data_lock:
                            rpm_data['point'] = current_point
                            rpm_data['mean'] = float(mean_part)
                            rpm_data['stddev'] = float(stddev_part)
            except (ValueError, IndexError, UnicodeDecodeError):
                pass
            except serial.SerialException:
                break
            
            # CRITICAL: Yield CPU so the plot can update
            time.sleep(0.005) 
            
    except serial.SerialException as e:
        print(f"❌ Error with RPM serial port: {e}")

def read_stats_serial():
    try:
        ser = serial.Serial(STATS_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to Stats port: {STATS_PORT}")
        global serial_ports
        serial_ports['stats'] = ser
        
        temp_stats = {}
        
        while True:
            try:
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
            except (ValueError, IndexError, UnicodeDecodeError):
                print("⚠️ Malformed Stats line, skipping...")
                pass
            except serial.SerialException as e:
                print(f"⚠️ Stats serial error: {e}")
                break

            # CRITICAL: Yield CPU so the plot can update
            time.sleep(0.005)

    except serial.SerialException as e:
        print(f"❌ Error with Stats serial port: {e}")

def read_suction_fan_serial():
    try:
        ser = serial.Serial(SUCTION_FAN_PORT, BAUDRATE, timeout=1)
        print(f"✅ Connected to Suction Fan port: {SUCTION_FAN_PORT}")
        global serial_ports
        serial_ports['suction_fan'] = ser
        while True:
            if ser.in_waiting > 0:
                ser.readline() # clear buffer
            
            # CRITICAL: Yield CPU so the plot can update
            time.sleep(0.01) 
            
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
        
        # Skip processing if RPM data not yet available
        if rpm_mean is None or rpm_std is None:
            print("⚠️ Waiting for RPM data... skipping this measurement")
            return
        
        # Sensor0 is usually Venturi total - static (from pitot and tapping) (positive)
        dp_sensor0_mean= stats.get('dp_sensor0_mean', 0)
        dp_sensor0_std = stats.get('dp_sensor0_stddev', 0)
        
        # Sensor 1 is usually Atm total - static tapping after stage (negative)
        dp_sensor1_mean = stats.get('dp_sensor1_mean', 0)
        dp_sensor1_std = stats.get('dp_sensor1_stddev', 0)
        
        temp_mean = stats.get('temp_mean', 15)
        temp_std = stats.get('temp_stddev', 0)       
        
        # 2. Calculate Derived Values
        U = RPM_TO_U_CONV * rpm_mean
        U_sq = U**2
        
        T_abs_mean = 273.15 + temp_mean
        rho_mean = pressureAtm / (287 * T_abs_mean)
        
        # Mass flow and axial velocity using venturi pitot and tapping (METHOD 1: Pitot-based)
        axvel_throat_mean = np.sqrt(np.abs(dp_sensor0_mean/(0.5 * rho_mean)))
        mdot_mean = rho_mean * Athroat * axvel_throat_mean
        axvel_mean = mdot_mean / (Afan * rho_mean)
        
        # ALTERNATIVE METHOD 2: Venturi static tappings only
        # mdot_mean = C * E * epsilon * np.pi / 4.0 * pow(d, 2) * np.sqrt(2.0 * abs(dp_sensor0_mean) * rho_mean)
        # axvel_mean = mdot_mean / (Afan * rho_mean)
        
        dp_stage_mean = -(dp_sensor1_mean-0.5*rho_mean*axvel_mean**2) # Correction to inlet static (static-static stage dp)
        
        
        # Venturi pressure (same as sensor0 differential)
        dp_venturi_mean = dp_sensor0_mean
        dp_venturi_std = dp_sensor0_std
        
        phi = axvel_mean / U if abs(U) > 1e-6 else 0.0
        pRise = dp_stage_mean / (rho_mean * U_sq) if (U_sq > 1e-6 and rho_mean > 0) else 0.0
        
        # 3. Calculate Error Propagation (Standard Deviations)
        
        # Error in rho (ρ = P / (R * T))
        # sigma_rho = rho * sqrt( (sigma_T/T)^2 )
        # (pressure is constant, R is constant)
        
        rho_std = rho_mean * (temp_std / T_abs_mean) if T_abs_mean != 0 else 0
        
        # Error in axvel (V = sqrt(2*dP/rho))
        # sigma_axvel = axvel * sqrt( (sigma_dP/dP)^2 / 4 + (sigma_rho/rho)^2 / 4 )
        rel_var_dp0 = (dp_sensor0_std / dp_sensor0_mean)**2 if dp_sensor0_mean != 0 else 0
        rel_var_rho_local = (rho_std / rho_mean)**2 if rho_mean != 0 else 0
        
        axvel_throat_std = axvel_throat_mean * np.sqrt(0.25 * rel_var_dp0 + 0.25 * rel_var_rho_local) if axvel_throat_mean != 0 else 0
        
        # Error in mdot (mdot = rho * A * V)
        # sigma_mdot = mdot * sqrt( (sigma_rho/rho)^2 + (sigma_V/V)^2 )
        rel_var_axvel = (axvel_throat_std / axvel_throat_mean)**2 if axvel_throat_mean != 0 else 0
        mdot_std = mdot_mean * np.sqrt(rel_var_rho_local + rel_var_axvel) if mdot_mean != 0 else 0
        # Error in axvel (from mdot)
        axvel_std = axvel_mean * np.sqrt(rel_var_rho_local + (mdot_std / mdot_mean)**2) if axvel_mean != 0 else 0
        
        # ALTERNATIVE METHOD 2: Venturi static tappings only
        # Error in axvel: V = mdot / (A * rho), sigma_V = V * sqrt( (sigma_mdot/mdot)^2 + (sigma_rho/rho)^2 )
        # mdot formula has: sqrt(2*dP*rho), so error is mdot_alt * sqrt( (sigma_dP/dP)^2 / 4 + (sigma_rho/rho)^2 / 4 )
        # axvel_std = axvel_mean * np.sqrt(rel_var_mdot + rel_var_rho_local) if axvel_mean != 0 else 0
        
        # Relative errors (squared)
        # If mean is 0, assume relative error contribution is 0 to prevent crash
        rel_var_rpm = (rpm_std / rpm_mean)**2 if rpm_mean != 0 else 0
        rel_var_U = rel_var_rpm # Since U is linear with RPM
        rel_var_rho = (rho_std / rho_mean)**2 if rho_mean != 0 else 0
        
        # Error in dp_stage (dP_stage = dP_sensor1 - 0.5*rho*V^2)
        # sigma_dp_stage = sqrt( sigma_dp_sensor1^2 + (0.5*V^2*sigma_rho)^2 + (rho*V*sigma_V)^2 )
        term1_sq = dp_sensor1_std**2
        term2_sq = (0.5 * axvel_mean**2 * rho_std)**2
        term3_sq = (rho_mean * axvel_mean * axvel_std)**2
        dp_stage_std = np.sqrt(term1_sq + term2_sq + term3_sq)
        
        # Error in Phi (Φ = Vx / U)
        # sigma_phi = phi * sqrt( (sigma_vx/vx)^2 + (sigma_U/U)^2 )
        rel_var_vx = (axvel_std / axvel_mean)**2 if axvel_mean != 0 else 0
        phi_std = abs(phi) * np.sqrt(rel_var_vx + rel_var_U)
        
        # Error in pRise (pRise = dP / (rho * U^2))
        # sigma_pRise = pRise * sqrt( (sigma_dp/dp)^2 + (sigma_rho/rho)^2 + (2*sigma_U/U)^2 )
        rel_var_dp = (dp_stage_std / dp_stage_mean)**2 if dp_stage_mean != 0 else 0
        pRise_std = abs(pRise) * np.sqrt(rel_var_dp + rel_var_rho + (4 * rel_var_U)) # 4 comes from (2*sigma_U/U)^2

        # 4. Save to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([
            timestamp, stats.get('point', 0), 
            rpm_mean, rpm_std,
            dp_venturi_mean, dp_venturi_std,
            dp_stage_mean, dp_stage_std,
            rho_mean, rho_std,
            mdot_mean, mdot_std,
            axvel_mean, axvel_std,
            phi, phi_std,
            pRise, pRise_std
        ])
        csv_file.flush()
        
        # 5. Update Plot Arrays
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

        print(f"Pt {stats.get('point',0)}: Φ={phi:.3f}±{phi_std:.3f}, pRise={pRise:.3f}±{pRise_std:.3f}")

def update_plot(frame):
    with data_lock:
        if len(plot_data['time']) < 2:
            return
        
        times = np.array(plot_data['time'])
        rel_times = times - times[0]
        
        for ax in axs.flat:
            ax.clear()
        
        # Style for error bars
        err_style = {'fmt': 'x', 'capsize': 3, 'markersize': 4, 'alpha': 0.3}
        
        # Plot 1: RPM
        axs[0, 0].errorbar(rel_times, plot_data['rpm_mean'], yerr=plot_data['rpm_err'], 
                           color='b', label='RPM', **err_style)
        axs[0, 0].set_ylabel('RPM')
        axs[0, 0].legend(loc='upper left')
        axs[0, 0].grid(True, alpha=0.7)
        
        # Plot 2: Mass Flow
        axs[0, 1].errorbar(rel_times, np.array(plot_data['mflow_mean'])*1e3, yerr=np.array(plot_data['mflow_err'])*1e3, 
                           color='r', label='Mass Flow*1e3', **err_style)
        axs[0, 1].errorbar(rel_times, plot_data['axvel_mean'], yerr=plot_data['axvel_err'], 
                           color='g', label='Axial Velocity', **err_style)
        axs[0, 1].legend(loc='upper left')
        axs[0, 1].set_ylabel('Mass Flow*1e3, Axial Vel')
        axs[0, 1].grid(True, alpha=0.7)
        
        # Plot 3: Performance Curve (XY Error Bars)
        axs[1, 0].errorbar(plot_data['phi_mean'], plot_data['pRise_mean'], 
                           xerr=plot_data['phi_err'], yerr=plot_data['pRise_err'],
                           color='k', label='Perf. Curve', **err_style)
        axs[1, 0].set_ylabel('Pressure Rise Coeff')
        axs[1, 0].set_xlabel('Flow Coeff ($\\Phi$)')
        # axs[1, 0].set_ylim(0,1.5)
        # axs[1, 0].set_xlim(0,1)
        axs[1, 0].grid(True, alpha=0.7)
        
        # Plot 4: Pressures
        axs[1, 1].errorbar(rel_times, plot_data['dp_venturi_mean'], yerr=plot_data['dp_venturi_err'], 
                           color='orange', label='$\\Delta P$ Venturi', **err_style)
        axs[1, 1].errorbar(rel_times, plot_data['dp_stage_mean'], yerr=plot_data['dp_stage_err'], 
                           color='purple', label='$\\Delta P$ Stage', **err_style)
        axs[1, 1].errorbar(rel_times, np.array(plot_data['rho_mean'])*10, 
                           yerr=np.array(plot_data['rho_err'])*10, 
                           color='brown', label='rho*10', **err_style)
        axs[1, 1].set_ylabel('Pressure (Pa), Density*10')
        axs[1, 1].legend(loc='upper left')
        axs[1, 1].grid(True, alpha=0.7)

if __name__ == "__main__":
    global serial_ports
    serial_ports = {'rpm': None, 'stats': None, 'suction_fan': None}
    
    print("Starting Scientific Data Logger...")
    init_csv()
    
    # Start threads
    t1 = threading.Thread(target=read_rpm_serial, daemon=True)
    t2 = threading.Thread(target=read_stats_serial, daemon=True)
    t3 = threading.Thread(target=read_suction_fan_serial, daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    
    print("Waiting for connections...")
    time.sleep(3)
    start_test()
    
    # Setup Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    
    ani = FuncAnimation(fig, update_plot, interval=250, cache_frame_data=False)
    
    try:
        plt.tight_layout()
        print("Opening Plot Window...")
        plt.show() # This blocks here. Window should open and update.
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        csv_file.close()
        fig.savefig(f"figs/dataplot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png")
        for s in serial_ports.values():
            if s and s.is_open: 
                command = "stop\n"
                s.write(command.encode('utf-8'))
                s.close()
        print(f"Data saved to {CSV_FILENAME}")