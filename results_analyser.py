import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime as dt
import re

file_names = [#r"data\5blade_2511_test_data_20251125_131042.csv",
              r"data\7blade_2511_test_data_20251125_130254.csv",
              r"data\7blade_2511_test_data_20251125_145210.csv",
              r"data\7blade_2511_test_data_20251125_145922.csv",]

# fan_results = []
flowcoeffs = []
prisecoeffs = []
avgSpeeds = []

for i, f in enumerate(file_names):
    fan_result = (pd.read_csv(f))
    flowcoeffs.append(fan_result['flow_coefficient_mean'])
    prisecoeffs.append(fan_result['pressure_rise_coefficient_mean'])
    avgSpeeds.append(np.median(fan_result['rpm_mean']))
    

def datetime_conversion(timestamp):
    timestamp = timestamp[:-4]
    return dt.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S')
def moving_average(data, window_size=5):
    """Calculate moving average of data"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

re_expression = r"([0-9]+blade)"
labels = [re.search(re_expression,f).group(0)+f" (Avg RPM: {avgSpeeds[i]:.1f})" for i,f in enumerate(file_names)]

polyfit_lines = [np.poly1d(np.polyfit(flowcoeffs[i],prisecoeffs[i],3)) for i in range(len(file_names))]

fig,ax = plt.subplots()
for x in range(len(flowcoeffs)):
    ax.scatter(flowcoeffs[x],prisecoeffs[x],marker='x',label=labels[x])
    ax.plot(np.linspace(0.25,0.4,100),polyfit_lines[x](np.linspace(0.25,0.4,100)),label=f"{labels[x]} Fit", linestyle='--', alpha=0.5)
ax.set_xlabel(r"Flow Coeff")
ax.set_ylabel(r"Prise coeff")
# ax.set_title(f"Axial Velocity")
ax.set_ylim(0.15,0.3)
ax.set_xlim(0.25,0.4)
ax.legend()
ax.grid()
# fig.savefig(f"figs/axial_velocity_errors_plot.png")
plt.show()

# fig, ax = plt.subplots()
# for x in range(len(flowcoeffs)):
#     # Sort both arrays by pressure rise coefficient
#     sorted_indices = np.argsort(flowcoeffs[x].values)
#     sorted_flowcoeff = flowcoeffs[x].values[sorted_indices]
#     sorted_prisecoeff = prisecoeffs[x].values[sorted_indices]
    
#     # Calculate moving average
#     window_size = 15
#     ma_prisecoeff = moving_average(sorted_prisecoeff, window_size=window_size)
#     # Trim flowcoeff to match moving average length
#     ma_flowcoeff = sorted_flowcoeff[:(len(sorted_flowcoeff) - window_size + 1)]
    
#     ax.scatter(flowcoeffs[x], prisecoeffs[x], marker='x', label=labels[x], alpha=0.3)
#     ax.plot(ma_flowcoeff, ma_prisecoeff, label=f"{labels[x]} Moving Avg", linestyle='--', alpha=1, linewidth=2)

# ax.set_xlabel(r"Flow Coeff")
# ax.set_ylabel(r"Prise coeff")
# ax.set_ylim(0.15, 0.3)
# ax.set_xlim(0.25, 0.4)
# ax.legend()
# ax.grid()
# plt.show()
