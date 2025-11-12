import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime as dt

file_name = "data/measurements.csv"
results_df = pd.read_csv(file_name)

#print(results_df)

rmid = 0.0275
u = 3200 /60*2*np.pi*rmid # u_mid in rad/s
plotting_fan_type = '9_blade'
testing_time = dt.datetime(2025,11,11,16,13)
plotting_fan_type2 = '7_blade_flat'
testing_time2 = dt.datetime(2025,11,11,16,21)
margin = dt.timedelta(0,120)

def datetime_conversion(timestamp):
    return dt.datetime.strptime(timestamp,'%d/%m/%Y %H:%M')

results_df["flow_coefficient"] = results_df["Vx"].div(u)
results_df["p_rise_coefficent"] = results_df["p1-p2"].div(results_df["rho"]).div(u**2)
results_df["Timestamp"] = results_df["Timestamp"].apply(datetime_conversion)

x = results_df[(results_df['fan_type']== plotting_fan_type) & (results_df['Timestamp'] < testing_time+margin) & (results_df['Timestamp'] > testing_time-margin)]["flow_coefficient"]
y = results_df[(results_df['fan_type']== plotting_fan_type) & (results_df['Timestamp'] < testing_time+margin) & (results_df['Timestamp'] > testing_time-margin)]["p_rise_coefficent"]
x2 = results_df[(results_df['fan_type']== plotting_fan_type2) & (results_df['Timestamp'] < testing_time2+margin) & (results_df['Timestamp'] > testing_time2-margin)]["flow_coefficient"]
y2 = results_df[(results_df['fan_type']== plotting_fan_type2) & (results_df['Timestamp'] < testing_time2+margin) & (results_df['Timestamp'] > testing_time2-margin)]["p_rise_coefficent"]


print(len(y))

fig,ax = plt.subplots()
ax.scatter(x,y,marker='x',label=plotting_fan_type)
ax.scatter(x2,y2,marker='x',label=plotting_fan_type2)
ax.set_xlabel(r"$\frac{Vx}{U}$")
ax.set_ylabel(r"$\frac{\Delta p}{\rho U^2}$")
ax.set_title(f"Pressure rise against flow coefficient")
ax.set_ylim(0,0.6)
ax.set_xlim(0.4,0.8)
ax.legend()
ax.grid()
fig.savefig(f"figs/{plotting_fan_type}_{plotting_fan_type2}_{testing_time.strftime('%d_%m_%Y_%H_%M')}_plot")
plt.show()
