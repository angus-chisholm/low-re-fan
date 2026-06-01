import serial
import csv
from datetime import datetime
import msvcrt

# Configure your serial port
ser = serial.Serial('COM4', 9600, timeout=1)  # Adjust COM port and baud rate
csv_file = open('data/measurements.csv', 'a', newline='')
writer = csv.writer(csv_file)

# Optional: Write header row if file is new
# writer.writerow(['Timestamp', 'fan_type', 'p2-p3', 'mdot', 'Vx', 
#                  'p1-patm', 'p2-patm', 'p1-p2', 'rho'])

print("Displaying live data. Press ENTER to record measurement, 'q' to quit\n")

print("Live data streaming... Press SPACE to record, 'q' to quit\n")
print("-" * 50)

recent_lines = []

try:
    while True:
        # Read and display incoming serial data
        if ser.in_waiting:
            line = ser.readline().decode().strip()
            if line:
                print(line)
                recent_lines.append(line)
                # Keep only last 2 lines in buffer
                if len(recent_lines) > 2:
                    recent_lines.pop(0)
        
        # Check for key press (non-blocking)
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').lower()
            
            if key == 'q':
                print("\nQuitting...")
                break
            
            elif key == ' ':  # Spacebar
                if len(recent_lines) >= 2:
                    line1 = recent_lines[-2]
                    line2 = recent_lines[-1]
                    
                    values1 = line1.split(',')[1:]
                    values2 = line2.split(',')[1:]
                    
                    timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
                    row = [timestamp] + values1 + values2
                    
                    writer.writerow(row)
                    csv_file.flush()
                    
                    print("\n" + "=" * 50)
                    print(f"✓ RECORDED at {timestamp}")
                    print(f"  Line 1: {line1}")
                    print(f"  Line 2: {line2}")
                    print("=" * 50 + "\n")
                else:
                    print("\n⚠ Not enough data yet, wait for 2 lines\n")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    csv_file.close()
    ser.close()
    print("\nData saved to data/measurements.csv")
    
    
    
    
# import serial
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from collections import deque
# import time # Optional, for timing/delay

# # --- Configuration ---
# SERIAL_PORT = 'COM3'  # Change this to your port
# BAUD_RATE = 9600
# MAX_POINTS = 100      # Number of data points to show on the graph
# INTERVAL_MS = 50      # Update interval in milliseconds

# # --- 1. Initialize Serial Connection ---
# try:
#     ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
#     time.sleep(2) # Give time for the connection to establish
# except Exception as e:
#     print(f"Error opening serial port: {e}")
#     exit()

# # --- 2. Initialize Data Structure ---
# # Create a deque for the y-values (and optionally one for x-values if needed)
# data_y = deque([0] * MAX_POINTS, maxlen=MAX_POINTS)
# # Example of an x-axis showing index/time
# data_x = deque(range(MAX_POINTS), maxlen=MAX_POINTS)

# # --- 3. Define the Plot ---
# fig, ax = plt.subplots()
# line, = ax.plot(data_x, data_y, label='Live Signal')

# ax.set_ylim(0, 1023) # Set appropriate limits for your data (e.g., 0-1023 for 10-bit ADC)
# ax.set_title("Live Serial Data Plotter")
# ax.set_xlabel("Time/Sample")
# ax.set_ylabel("Sensor Value")
# ax.legend()
# ax.grid(True)

# # --- 4. Create the Update Function ---
# def update_plot(frame):
#     """Reads serial data, processes it, and updates the plot."""
#     try:
#         # Check if any data is in the buffer
#         if ser.in_waiting > 0:
#             # Read a full line (until newline character '\n')
#             serial_data = ser.readline().decode('utf-8').strip()

#             if serial_data:
#                 # --- Processing Step ---
#                 # Assuming your device sends a single integer/float per line, e.g., "456"
#                 try:
#                     # Convert the string to a number
#                     new_value = float(serial_data)

#                     # --- Update Data ---
#                     data_y.append(new_value)
                    
#                     # Update X-axis labels to reflect new samples if needed
#                     # data_x.append(data_x[-1] + 1)

#                     # --- Update Plot ---
#                     line.set_ydata(data_y)
                    
#                     # Auto-scale the X-axis for the new set of data
#                     ax.set_xlim(min(data_x), max(data_x))
                    
#                     # Optional: Auto-scale the Y-axis if data range changes
#                     # min_y = min(data_y)
#                     # max_y = max(data_y)
#                     # ax.set_ylim(min_y - 1, max_y + 1)
                    
#                 except ValueError:
#                     # Handles cases where the line read is not a valid number
#                     print(f"Skipping malformed data: {serial_data}")

#     except serial.SerialException as e:
#         print(f"Serial Error: {e}")
    
#     # Return the updated artist (line)
#     return line,

# # --- 5. Start the Animation ---
# # FuncAnimation calls the update_plot function repeatedly
# ani = FuncAnimation(fig, update_plot, interval=INTERVAL_MS, blit=True)

# plt.show()

# # Clean up
# ser.close()