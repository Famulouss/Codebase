import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -------------------
# Butterworth Tiefpass-Filter
# -------------------
def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# -------------------
# CSV laden
# -------------------
Tk().withdraw()
filename = askopenfilename(
    title="Bitte CSV-Datei auswählen",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

df = pd.read_csv(filename, skiprows=3)

# relative Zeit (s)
t0 = df["timestamp"].iloc[0]
df["time_rel"] = (df["timestamp"] - t0) / 1000.0

# Spalten bereinigen & umbenennen
df = df.drop(columns=["lsm6dso_accelerometer.z","lsm6dso_gyroscope.x", "lsm6dso_gyroscope.y"])
df = df.rename(columns={
    "lsm6dso_accelerometer.x": "acc_x",
    "lsm6dso_accelerometer.y": "acc_y",
    "lsm6dso_gyroscope.z": "yaw"
})

# -------------------
# Filter anwenden
# -------------------
fs = 10.0      # Abtastrate [Hz]
cutoff = 2.0   # Grenzfrequenz [Hz]

df["acc_x_filt"] = lowpass_filter(df["acc_x"], cutoff, fs)
df["acc_y_filt"] = lowpass_filter(df["acc_y"], cutoff, fs)

# -------------------
# Plot: Beschleunigung roh vs. gefiltert
# -------------------
plt.figure(figsize=(10, 5))
plt.plot(df['time_rel'], df['acc_x'], label='Acc X (raw)', alpha=0.5)
plt.plot(df['time_rel'], df['acc_x_filt'], label='Acc X (filtered)', linewidth=2)
plt.plot(df['time_rel'], df['acc_y'], label='Acc Y (raw)', alpha=0.5)
plt.plot(df['time_rel'], df['acc_y_filt'], label='Acc Y (filtered)', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s²]')
plt.title('IMU Linear Acceleration (with Lowpass Filter)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# -------------------
# Winkelbeschleunigung berechnen (Ableitung Winkelgeschwindigkeit)
# -------------------
a = df['yaw'].values
t = df['time_rel'].values
yaw_acc = np.gradient(a, t)
df['yaw_acc'] = yaw_acc

yaw_acc_cut = 1.0   # Hz
df['yaw_acc_filt'] = lowpass_filter(df['yaw_acc'].values, yaw_acc_cut, fs, order=2)

# -------------------
# Winkelgeschwindigkeit und -beschleunigung plotten (unverändert)
# -------------------
plt.figure(figsize=(10, 5))
plt.plot(df['time_rel'], df['yaw'], label='Yaw-Rate')
plt.plot(df['time_rel'], df['yaw_acc_filt'], label='Yaw-Acceleration (filtered)')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.title('IMU Angular Velocity')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# -------------------
# Jerk berechnen aus den gefilterten Daten
# -------------------
for axis in ['x', 'y']:
    a = df[f'acc_{axis}_filt'].values
    t = df['time_rel'].values
    jerk = np.gradient(a, t)
    df[f'jerk.{axis}_raw'] = jerk

jerk_cut = 1.0  # Hz
df["jerk.x"] = lowpass_filter(df["jerk.x_raw"].values, jerk_cut, fs, order=2)
df["jerk.y"] = lowpass_filter(df["jerk.y_raw"].values, jerk_cut, fs, order=2)

# Jerk plotten
plt.figure(figsize=(10, 6))
plt.plot(df['time_rel'], df['jerk.x'], label='Jerk X (filtered)')
plt.plot(df['time_rel'], df['jerk.y'], label='Jerk Y (filtered)')
plt.xlabel('Time [s]')
plt.ylabel('Jerk [m/s³]')
plt.title('Jerk (from filtered Acceleration)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
