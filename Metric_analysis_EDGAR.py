import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, lfilter, freqz
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -------------------
# Hilfsfunktionen
# -------------------
def butter_lowpass(cutoff, fs, order=2):
    """Butterworth Tiefpass-Filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=2):
    """Tiefpass-Filter für Butterworth-Filter"""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def numerical_derivative(values, times):
    """Eigene Gradienten Berechnung (keinen 0 Teiler) TODO: Braucht es das noch? (Nachdem das Problem mit den Zeitstempeln gelöst wurde)"""
    times = np.asarray(times)
    values = np.asarray(values)

    dt = np.diff(times)
    dv = np.diff(values)

    # Division durch 0 vermeiden
    dt[dt == 0] = np.nan

    derivative = dv / dt

    # Länge anpassen: vorne 0 oder NaN einsetzen
    derivative = np.insert(derivative, 0, 0.0)

    # NaN und Inf ersetzen
    return np.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0)

def integral_abs(times, signal):
    """Integral der absoluten Beschleunigung: ∫ |a(t)| dt (Trapezregel)."""
    return np.trapezoid(np.abs(signal), times)

def highpass_norm(omega1):
    """
    Aufteilung in Numerator und Denominator des Hochpassfilters aus der Norm 2631: 
    Hh(p) = 1/(1+sqrt(2)*omega1/p+(omega1/p)^2)
    """
    num = [1, 0, 0] # Numerator (Zählerkoeffizienten N(s))
    den = [1, np.sqrt(2)*omega1, omega1**2] # Denominator (Nennerkoeffizienten (D(s))
    return num, den

def lowpass_norm(omega2):
    """
    Aufteilung in Numerator und Denominator des Tiefpassfilters aus der Norm 2631:
    Hl(p) = 1/(1+sqrt(2)*p/omega2+(p/omega2)^2)
    """
    num = [1]
    den = [1/(omega2**2), np.sqrt(2)/omega2, 1]
    return num, den

def acc_vel_transition_filter(omega3, omega4, Q4):
    """
    Aufteilung in Numerator und Denominator des "acceleration-velocity transition"-Filters aus der Norm 2631:
    Hl(p) = (1+p/omega3)/(1+p/(Q4*omega4)+(p/omega4)^2)
    """
    num = [1/omega3, 1]
    den = [1/(omega4**2), 1/(Q4*omega4), 1]
    return num, den

def design_wd_filter(fs):
    """
    Entwirft digitale Filterkoeffizienten für Wd-Gewichtung gemäß ISO 2631 (Annäherung).
    fs: Abtastrate in Hz
    returns: b, a (arrays)
    """
    T = 1.0 / fs
    omega1 = 2 * np.pi * 0.4    # f1 = 0,4 Hz
    omega2 = 2 * np.pi * 100.0  # f2 = 100 Hz
    omega3 = 2 * np.pi * 2.0    # f3 = 2 Hz
    omega4 = 2 * np.pi * 2.0    # f4 = 2 Hz
    q4 = 0.63

    # Konstruktion der Teilfilter (analog)
    num_h, den_h = highpass_norm(omega1)    # Hochpassfilter
    num_l, den_l = lowpass_filter(omega2)   # Tiefpassfilter
    num_t, den_t = acc_vel_transition_filter(omega3, omega4, q4)    # Acceleration-velocity transition Filter
    
    # Gesamtnumerator/-denominator durch Polynom-Multiplikation (Faltung der Koeffizienten)
    num_temp = np.convolve(num_h, num_l)
    num_total = np.convolve(num_temp, num_t)
    den_temp = np.convolve(den_h, den_l)
    den_total = np.convolve(den_temp, den_t)

    # Diskretisierung/Umwandlung in digitales Filter (bilineare Transformation)
    b, a = signal.bilinear(num_total, den_total, fs)
    
    return b, a

# -------------------
# CSV laden
# -------------------
Tk().withdraw()
filename = askopenfilename(
    title="Bitte CSV-Datei auswählen",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

df = pd.read_csv(filename)

# relative Zeit (s)
t0 = df["timestamp"].iloc[0]
df["time_rel"] = (df["timestamp"] - t0)

# Spalten bereinigen & umbenennen
#df = df.drop(columns=["lsm6dso_accelerometer.z","lsm6dso_gyroscope.x", "lsm6dso_gyroscope.y"])
columns_to_keep = ["timestamp", "time_rel", "linear_acceleration_x", "linear_acceleration_y", "angular_velocity_z"]
df = df[columns_to_keep]
df = df.rename(columns={
    "linear_acceleration_x": "acc_x",
    "linear_acceleration_y": "acc_y",
    "angular_velocity_z": "yaw"
})

print(f'Column names: {df.columns}')

# -------------------
# Filter anwenden
# -------------------
fs = 10.0      # Abtastrate [Hz]
cutoff = 2.0   # Grenzfrequenz [Hz]

df["acc_x_filt"] = lowpass_filter(df["acc_x"], cutoff, fs)
df["acc_y_filt"] = lowpass_filter(df["acc_y"], cutoff, fs)

# -------------------
# Winkelbeschleunigung berechnen
# -------------------
a = df['yaw'].values
t = df['time_rel'].values
yaw_acc = numerical_derivative(a, t)
df['yaw_acc'] = yaw_acc

yaw_acc_cut = 1.0   # Hz
df['yaw_acc_filt'] = lowpass_filter(df['yaw_acc'].values, yaw_acc_cut, fs, order=2)

# -------------------
# Jerk berechnen
# -------------------
for axis in ['x', 'y']:
    a = df[f'acc_{axis}_filt'].values
    t = df['time_rel'].values
    jerk = numerical_derivative(a, t)
    df[f'jerk.{axis}_raw'] = jerk

jerk_cut = 1.0  # Hz
df["jerk.x"] = lowpass_filter(df["jerk.x_raw"].values, jerk_cut, fs, order=2)
df["jerk.y"] = lowpass_filter(df["jerk.y_raw"].values, jerk_cut, fs, order=2)

# -------------------
# Alle Plots in EINEM Fenster
# -------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 1) Beschleunigung
axes[0].plot(df['time_rel'], df['acc_x'], label='Acc X (raw)', alpha=0.5)
axes[0].plot(df['time_rel'], df['acc_x_filt'], label='Acc X (filtered)', linewidth=2)
axes[0].plot(df['time_rel'], df['acc_y'], label='Acc Y (raw)', alpha=0.5)
axes[0].plot(df['time_rel'], df['acc_y_filt'], label='Acc Y (filtered)', linewidth=2)
axes[0].set_ylabel('Acceleration [m/s²]')
axes[0].set_title('IMU Linear Acceleration')
axes[0].grid()
axes[0].legend()

# 2) Yaw und Yaw-Beschleunigung
axes[1].plot(df['time_rel'], df['yaw'], label='Yaw-Rate')
axes[1].plot(df['time_rel'], df['yaw_acc'], label='Yaw-Acceleration (unfiltered)')
axes[1].plot(df['time_rel'], df['yaw_acc_filt'], label='Yaw-Acceleration (filtered)')
axes[1].set_ylabel('Angular Velocity [rad/s]')
axes[1].set_title('IMU Angular Velocity')
axes[1].grid()
axes[1].legend()

# 3) Jerk
axes[2].plot(df['time_rel'], df['jerk.x'], label='Jerk X (filtered)')
axes[2].plot(df['time_rel'], df['jerk.y'], label='Jerk Y (filtered)')
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Jerk [m/s³]')
axes[2].set_title('Jerk (from filtered Acceleration)')
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()

# Integral/Aufsummierung der Beschleunigungskurve/-werte (Absolutwerte)
sum_x = integral_abs(df['time_rel'], df['acc_x'])
sum_y = integral_abs(df['time_rel'], df['acc_y'])
print(f"Integral x-Beschleunigung: {sum_x}\nIntegral y-Beschleunigung: {sum_y}")


# ISO 2631 Metrik (RMS der frequenzgewichteten Beschleunigung)