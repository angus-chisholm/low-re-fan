import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime as dt
import re

file_names = [r"data\test_data_20251125_175221.csv"]

fan_results = []
flowcoeffs = []
flowcoeffs_errors = []
prisecoeffs = []
prisecoeffs_errors = []
avgSpeeds = []
dpventuris = []
dpventuris_errors = []
dpstages = []
dpstages_errors = []


def datetime_conversion(timestamp):
    timestamp = timestamp[:-4]
    return dt.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S')

times = pd.read_csv(file_names[0])['timestamp'].apply(datetime_conversion)

for i, f in enumerate(file_names):
    fan_results.append(pd.read_csv(f))
    avgSpeeds.append(np.mean(fan_results[i]['rpm_mean']))
    flowcoeffs.append(fan_results[i]['flow_coefficient_mean'])
    flowcoeffs_errors.append(fan_results[i]['flow_coefficient_stddev'])
    prisecoeffs.append(fan_results[i]['pressure_rise_coefficient_mean'])
    prisecoeffs_errors.append(fan_results[i]['pressure_rise_coefficient_stddev'])
    dpventuris.append(fan_results[i]['dp_venturi_mean'])
    dpventuris_errors.append(fan_results[i]['dp_venturi_stddev'])
    dpstages.append(fan_results[i]['dp_stage_mean'])
    dpstages_errors.append(fan_results[i]['dp_stage_stddev'])
    

def moving_average(data, window_size=5):
    """Calculate moving average of data"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

re_expression = r"([0-9]+blade)"
re_expression2 = r"blade_([a-z0-9]+)[_,.]"
labels = ['''re.search(re_expression,f).group(0)+''' f" (Avg RPM: {avgSpeeds[i]:.1f})" for i,f in enumerate(file_names)]

#polyfit_lines = [np.poly1d(np.polyfit(flowcoeffs[i],prisecoeffs[i],3)) for i in range(len(file_names))]

fig,ax = plt.subplots()
for x in range(len(flowcoeffs)):
    #ax.scatter(flowcoeffs[x],prisecoeffs[x],marker='x',label=labels[x])
    #ax.plot(np.linspace(0.25,0.4,100),polyfit_lines[x](np.linspace(0.25,0.4,100)),label=f"{labels[x]} Fit", linestyle='--', alpha=0.5)
    ax.errorbar(times,dpventuris[x],yerr=dpventuris_errors[x],fmt='x',capsize=3, label=f"Venturi DP", alpha=0.5)
    ax.errorbar(times,dpstages[x],yerr=dpstages_errors[x],fmt='x',capsize=3,label=f"Stage DP", alpha=0.5)
# ax.set_xlabel(r"Flow Coeff")
# ax.set_ylabel(r"Prise coeff")
# ax.set_title(f"Axial Velocity")
# ax.set_ylim(0,0.5)
# ax.set_xlim(0,0.5)
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
