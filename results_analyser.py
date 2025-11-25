import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime as dt
import re

file_names = [r"data\5blade_test_data_20251121_151802.csv",
              r"data\7blade_test_data_20251121_150811.csv",
              r"data\9blade_test_data_20251121_123704.csv",
              r"data\5blade_redone_test_data_20251121_160151.csv"]
# fan_results = []
flowcoeffs = []
prisecoeffs = []

for i, f in enumerate(file_names):
    fan_result = (pd.read_csv(f))
    flowcoeffs.append(fan_result['flow_coefficient'])
    prisecoeffs.append(fan_result['pressure_rise_coefficient'])
    

def datetime_conversion(timestamp):
    timestamp = timestamp[:-4]
    return dt.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S')

re_expression = r"([0-9]+blade)"
labels = [re.search(re_expression,f).group(0) for f in file_names]

fig,ax = plt.subplots()
for x in range(len(flowcoeffs)):
    ax.scatter(flowcoeffs[x],prisecoeffs[x],marker='x',label=labels[x])
ax.set_xlabel(r"Flow Coeff")
ax.set_ylabel(r"Prise coeff")
# ax.set_title(f"Axial Velocity")
ax.set_ylim(0,1.2)
ax.set_xlim(0.2,1)
ax.legend()
ax.grid()
# fig.savefig(f"figs/axial_velocity_errors_plot.png")
plt.show()
