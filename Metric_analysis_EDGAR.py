import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy.signal as signal
import tkinter as tk
from scipy.signal import butter, filtfilt, lfilter, freqz
from tkinter import Tk, ttk
from tkinter.filedialog import askopenfilename
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

# -------------------
# Globale Variablen
# -------------------
# m1: Methode 1 -> Einfache Threshold-Detektion (Threshold aus Paper)
# m2: Methode 2 -> Thresholds für Integral über unterschiedlich große Intervalle (1s, 3s, 5s) (Thresholds müssen noch bestimmt werden)
# m3: Methode 3 -> Thresholds für RMS nach Norm 2631
norm_thresh = [0.315, 0.5, 0.63, 0.8, 1.25, 1.6, 2, 2.5]
thresholds_x = {'m1': 1.23, 'm2': [10.0, 10.0, 10.0], 'm3':norm_thresh}
thresholds_y = {'m1': 0.98, 'm2': [10.0, 10.0, 10.0], 'm3':norm_thresh}
thresholds_yaw = {'m1': 0.97, 'm2': [10.0, 10.0, 10.0], 'm3':norm_thresh}
thresholds_dict = {'acc_x': thresholds_x, 'acc_y': thresholds_y, 'acc_yaw': thresholds_yaw}


thresh_x = 1.23     # m/s^2
thresh_y = 0.98     # m/s^2
thresh_yaw = 0.97   # m/s^2
thresh_x_integral = 2.0
thresh_y_integral = 2.0
thresh_yaw_integral = 2.0


# -------------------
# Hilfsfunktionen
# -------------------
def confirm_selection():    # Für die Abtastratenauswahl
    value = float(combo.get())
    global fs_glob
    fs_glob = 1000/value
    print(f"Gewählte Abtastrate: {value} ms, {fs_glob} Hz")
    root.destroy()

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
    omega1 = 2 * np.pi * 0.4    # f1 = 0,4 Hz
    omega2 = 2 * np.pi * 100.0  # f2 = 100 Hz
    omega3 = 2 * np.pi * 2.0    # f3 = 2 Hz
    omega4 = 2 * np.pi * 2.0    # f4 = 2 Hz
    q4 = 0.63

    # Konstruktion der Teilfilter (analog)
    num_h, den_h = highpass_norm(omega1)    # Hochpassfilter
    num_l, den_l = lowpass_norm(omega2)   # Tiefpassfilter
    num_t, den_t = acc_vel_transition_filter(omega3, omega4, q4)    # Acceleration-velocity transition Filter
    
    # Gesamtnumerator/-denominator durch Polynom-Multiplikation (Faltung der Koeffizienten)
    num_temp = np.convolve(num_h, num_l)
    num_total = np.convolve(num_temp, num_t)
    den_temp = np.convolve(den_h, den_l)
    den_total = np.convolve(den_temp, den_t)

    # Diskretisierung/Umwandlung in digitales Filter (bilineare Transformation)
    b, a = signal.bilinear(num_total, den_total, fs)
    
    return b, a

def design_we_filter(fs):
    """
    Entwirft digitale Filterkoeffizienten für We-Gewichtung gemäß ISO 2631 (Annäherung).
    fs: Abtastrate in Hz
    returns: b, a (arrays)
    """
    omega1 = 2 * np.pi * 0.4    # f1 = 0,4 Hz
    omega2 = 2 * np.pi * 100.0  # f2 = 100 Hz
    omega3 = 2 * np.pi * 1.0    # f3 = 1 Hz
    omega4 = 2 * np.pi * 1.0    # f4 = 1 Hz
    q4 = 0.63

    # Konstruktion der Teilfilter (analog)
    num_h, den_h = highpass_norm(omega1)    # Hochpassfilter
    num_l, den_l = lowpass_norm(omega2)   # Tiefpassfilter
    num_t, den_t = acc_vel_transition_filter(omega3, omega4, q4)    # Acceleration-velocity transition Filter
    
    # Gesamtnumerator/-denominator durch Polynom-Multiplikation (Faltung der Koeffizienten)
    num_temp = np.convolve(num_h, num_l)
    num_total = np.convolve(num_temp, num_t)
    den_temp = np.convolve(den_h, den_l)
    den_total = np.convolve(den_temp, den_t)

    # Diskretisierung/Umwandlung in digitales Filter (bilineare Transformation)
    b, a = signal.bilinear(num_total, den_total, fs)

    return b, a

def sliding_metrics(signal, times, window_s=1.0, overlap=0.5, weighted=False):
    """
    Berechnet die gleitenden RMS-Intervalle übers Signal hinweg. Achtung: times muss in Sekunden sein (nicht ms)!
    Weighted=True: Der frequenzgewichtete Mittelwert (RMS) wird pro Intervall berechnet.
    Weighted=False: Es wird das Integral über die absoluten Beschleunigungswerte pro Intervall berechnet.
    Return: rms_list, rms_times
    """
    times = np.asarray(times)
    signal = np.asarray(signal)
    
    step = window_s * (1.0-overlap)
    if step <= 0:
        raise ValueError("Overlap ist zu groß (<1)")
    # Ermittle die Fenstermittelpunkte
    t_start = times[0]
    t_end = times[-1]
    centers = np.arange(t_start + window_s/2.0, t_end - window_s/2.0 + 1e-9, step)  # Von 1. Argument werden im Abstand vom 3. Argument die Werte ermittelt bis zum ausschließlich 2. Argument

    value_list = []

    # Ermittle den RMS Wert für jedes Intervall
    for tc in centers:
        t0 = tc - window_s/2.0
        t1 = tc + window_s/2.0
        mask = (times >= t0) & (times <= t1)
        seg_t = times[mask]
        seg = signal[mask]

        # RMS Berechnung
        if weighted:
            # Integral für den frequenzgewichteten RMS (∫(aw^2)dt)
            integral = np.trapezoid(seg**2, seg_t)
            # weighted r.m.s (sqrt(1/T*I))
            rms = np.sqrt(integral/window_s)
            value_list.append(rms)
        # Integralberechnung
        else:
            integral = integral_abs(seg_t, seg)
            value_list.append(integral)

    return value_list, centers

def threshold_detection_single(signal, times, threshold):
    """
    Berechnet die Anzahl und die Gesamtdauer der Threshold-Überschreitungen mit einem threshold.
    """
    times = np.asarray(times)
    signal = np.asarray(signal)

    # Maske: True, wenn Signal über Schwellwert
    above = signal > threshold

    exceed_durations = []
    in_exceed = False
    start_time = None

    for i in range(len(signal)):
        if above[i] and not in_exceed:
            # Beginn einer Überschreitung
            in_exceed = True
            start_time = times[i]
        elif not above[i] and in_exceed:
            # Ende einer Überschreitung
            end_time = times[i]
            exceed_durations.append(end_time - start_time)
            in_exceed = False
    # Falls das Signal am Ende immer noch über Threshold ist
    if in_exceed and start_time is not None:
        exceed_durations.append(times[-1] - start_time)

    exceed_count = len(exceed_durations)
    total_duration = np.sum(exceed_durations) if exceed_durations else 0.0

    return exceed_count, exceed_durations, total_duration

def get_max_per_interval(time, signal, interval_size=1.0):
    """
    Berechnet den Maximalwert des Signals in jedem Intervall der gegebenen Größe.
    Gibt ein Dictionary mit Intervallzentren als Keys und Maximalwerten als Values zurück.
    """
    time = np.asarray(time)
    signal = np.asarray(signal)

    max_values = {}
    start_time = np.floor(time[0])
    end_time = np.ceil(time[-1])
    
    intervals = np.arange(start_time, end_time, interval_size)
    
    for start in intervals:
        end = start + interval_size
        mask = (time >= start) & (time < end)
        if np.any(mask):
            max_val = np.max(signal[mask])
            center = start + interval_size / 2
            max_values[center] = max_val
        else:
            # Falls kein Wert im Intervall liegt
            max_values[start + interval_size / 2] = np.nan

    return max_values

# -------------------
# CSV laden. WITCHTIG: In der csv-Datei müssen die Spalten richtig benannt sein: timestamp (in Sekunden!), linear_acceleration_x, linear_acceleration_y, angular_velocity_z. Die Reihenfolge und ob noch andere Spalten sind spielt keinen Rolle.
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
# Kontrolle ob s oder ms
if df['time_rel'][1] > 1:
    df['time_rel'] /= 1000

# Spalten bereinigen & umbenennen.
columns_to_keep = ["timestamp", "time_rel", "linear_acceleration_x", "linear_acceleration_y", "angular_velocity_z"]
df = df[columns_to_keep]
df = df.rename(columns={
    "linear_acceleration_x": "acc_x",
    "linear_acceleration_y": "acc_y",
    "angular_velocity_z": "yaw"
})

# -------------------
# Abtastrate wählen (EDGAR: 25ms, HyperIMU: 100ms)
# -------------------
# Hauptfenster
root = tk.Tk()
root.title("Abtastrate wählen")

# Label
label = tk.Label(root, text="Bitte Abtastrate (in ms) wählen.\nEDGAR: 25ms\nHyperIMU: 100ms")
label.pack(padx=10, pady=10)

# Dropdown-Menü (Combobox)
options = [25.0, 100.0]
combo = ttk.Combobox(root, values=options, state="readonly")
combo.current(0)  # Standardwert
combo.pack(padx=10, pady=5)

# OK-Button
ok_button = tk.Button(root, text="OK", command=confirm_selection)
ok_button.pack(padx=10, pady=10)

# GUI starten
root.wait_window()

# -------------------
# Filter anwenden
# -------------------
cutoff = 2.0   # Grenzfrequenz [Hz]
df["acc_x_filt"] = lowpass_filter(df["acc_x"], cutoff, fs_glob)
df["acc_y_filt"] = lowpass_filter(df["acc_y"], cutoff, fs_glob)
df['yaw_filt'] = lowpass_filter(df["yaw"], cutoff, fs_glob)

# -------------------
# Winkelbeschleunigung berechnen
# -------------------
a = df['yaw_filt'].values
t = df['time_rel'].values
yaw_acc = np.gradient(a, t)
df['yaw_acc'] = yaw_acc

yaw_acc_cut = 1.0   # Hz
df['yaw_acc_filt'] = lowpass_filter(df['yaw_acc'].values, yaw_acc_cut, fs_glob, order=2)

# -------------------
# Jerk berechnen
# -------------------
for axis in ['x', 'y']:
    a = df[f'acc_{axis}_filt'].values
    t = df['time_rel'].values
    jerk = np.gradient(a, t)
    df[f'jerk.{axis}_raw'] = jerk

jerk_cut = 1.0  # Hz
df["jerk.x"] = lowpass_filter(df["jerk.x_raw"].values, jerk_cut, fs_glob, order=2)
df["jerk.y"] = lowpass_filter(df["jerk.y_raw"].values, jerk_cut, fs_glob, order=2)

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
sum_x = integral_abs(df['time_rel'], df['acc_x_filt'])
sum_y = integral_abs(df['time_rel'], df['acc_y_filt'])
print(f"Integral x-Beschleunigung: {sum_x}\nIntegral y-Beschleunigung: {sum_y}")
# Intervallweise berechnen (1s, 3s, 5s)
t_total = df['time_rel'].iloc[-1]
intervals = [1.0, 3.0, 5.0, t_total]
sum_x, sum_x_interval_centers = {}, {}
sum_y, sum_y_interval_centers = {}, {}
sum_yaw, sum_yaw_interval_centers = {}, {}
for i in intervals:
    temp1, temp2 = sliding_metrics(df['acc_x_filt'], df['time_rel'], i)
    sum_x[i] = temp1
    sum_x_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['acc_y_filt'], df['time_rel'], i)
    sum_y[i] = temp1
    sum_y_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['yaw_acc_filt'], df['time_rel'], i)
    sum_yaw[i] = temp1
    sum_yaw_interval_centers[i] = temp2


# Beschleunigungsgraphen mit intervallweisem Integral
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(df['time_rel'], df['acc_x_filt'], label='Acc X (filtered)', linewidth=2)
for i in intervals[:-1]:
    axes[0].step(sum_x_interval_centers[i], sum_x[i], where="mid", label=f"Sliding Integral {i}s Intervall", linewidth=2)
axes[0].plot()
axes[0].set_ylabel('Acceleration [m/s²]')
axes[0].set_title('Filtered x-acceleration')
axes[0].grid()
axes[0].legend()

axes[1].plot(df['time_rel'], df['acc_y_filt'], label='Raw', alpha=0.7)
for i in intervals[:-1]:
    axes[1].step(sum_y_interval_centers[i], sum_y[i], where="mid", label=f"Sliding Integral {i}s Intervall", linewidth=2)
axes[1].set_ylabel('Acceleration [m/s²]')
axes[1].set_title('Filtered y-acceleration')
axes[1].grid()
axes[1].legend()

axes[2].plot(df['time_rel'], df['yaw_acc_filt'], label='Raw', alpha=0.7)
for i in intervals[:-1]:
    axes[2].step(sum_yaw_interval_centers[i], sum_yaw[i], where="mid", label=f"Sliding Integral {i}s Intervall", linewidth=2)
axes[2].set_ylabel('Acceleration [rad/s²]')
axes[2].set_xlabel('Time [s]')
axes[2].set_title('Filtered yaw-acceleration')
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()

# -------------------
# ISO 2631 Metrik (RMS der frequenzgewichteten Beschleunigung)
# -------------------
b, a = design_wd_filter(fs_glob)     # Gewichtung ist laut Norm für die x- und y-Beschleunigung diesselbe
df['acc_x_normfilt'] = filtfilt(b, a, df['acc_x'])
df['acc_y_normfilt'] = filtfilt(b, a, df['acc_y'])
b, a = design_we_filter(fs_glob)
df['yaw_acc_normfilt'] = filtfilt(b, a, df['yaw_acc'])
df['yaw_acc_normfilt'] *= 0.2   # k-Faktor nach Norm [m/rad]

# Berechne die rms Werte in unterschiedlich großen gleitenden Intervallen
# Pro Beschleunigungsrichtung gibt es ein dict welches als Schlüssel die jeweiligen Intervalle hat {1.0: [0.674, 0.121, ...], 3.0: [...], ...}
acc_x_rms_dict, x_interval_centers = {}, {}
acc_y_rms_dict, y_interval_centers = {}, {}
yaw_acc_rms_dict, yaw_interval_centers = {}, {}
for i in intervals:
    temp1, temp2 = sliding_metrics(df['acc_x_normfilt'], df['time_rel'], i, weighted=True)
    acc_x_rms_dict[i] = temp1
    x_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['acc_y_normfilt'], df['time_rel'], i, weighted=True)
    acc_y_rms_dict[i] = temp1
    y_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['yaw_acc_normfilt'], df['time_rel'], i, weighted=True)
    yaw_acc_rms_dict[i] = temp1
    yaw_interval_centers[i] = temp2

# Alternative rms Berechnungen laut Norm 2631
# MTVV (max transient vibration value) aus sliding rms (Tau=1s)
MTVV_x = np.max(acc_x_rms_dict[1.0])
MTVV_y = np.max(acc_y_rms_dict[1.0])
MTVV_yaw = np.max(yaw_acc_rms_dict[1.0])

# VDV (∫(aw^4)dt)^(1/4)
integral = np.trapezoid(df['acc_x_normfilt']**4, df['time_rel'])
vdv_x = integral**(1/4)
integral = np.trapezoid(df['acc_y_normfilt']**4, df['time_rel'])
vdv_y = integral**(1/4)
integral = np.trapezoid(df['yaw_acc_normfilt']**4, df['time_rel'])
vdv_yaw = integral**(1/4)

# Berechnen ob laut Norm die Alternativen rms Methoden besser sind
# MTVV Verhältnis berechnen (Wenn Schwellwert überschritten, macht es mehr Sinn intervallweise (1s) auszuwerten)
ratio_mtvv = MTVV_x/acc_x_rms_dict[t_total]
print(f"MTVV-Verhältnis für x-Beschleunigung: {ratio_mtvv}. (Schwellwert: 1,5)")
ratio_mtvv = MTVV_y/acc_y_rms_dict[t_total]
print(f"MTVV-Verhältnis für y-Beschleunigung: {ratio_mtvv}. (Schwellwert: 1,5)")
ratio_mtvv = MTVV_yaw/yaw_acc_rms_dict[t_total]
print(f"MTVV-Verhältnis für yaw-Beschleunigung: {ratio_mtvv}. (Schwellwert: 1,5)")

print(acc_x_rms_dict[t_total])
ratio_vdv = vdv_x/(acc_x_rms_dict[t_total][0]*(t_total**(1/4)))
print(f"VDV-Verhältnis für x-Beschleunigung: {ratio_vdv}. (Schwellwert: 1,75)")
ratio_vdv = vdv_y/(acc_y_rms_dict[t_total][0]*(t_total**(1/4)))
print(f"VDV-Verhältnis für y-Beschleunigung: {ratio_vdv}. (Schwellwert: 1,75)")
ratio_vdv = vdv_yaw/(yaw_acc_rms_dict[t_total][0]*(t_total**(1/4)))
print(f"VDV-Verhältnis für yaw-Beschleunigung: {ratio_vdv}. (Schwellwert: 1,75)")


# Gefilterte Beschleunigungsgrafiken
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(df['time_rel'], df['acc_x'], label='Raw', alpha=0.7)
axes[0].plot(df['time_rel'], df['acc_x_normfilt'], label='Filtered', alpha = 0.7)
for i in intervals[:-1]:
    axes[0].step(x_interval_centers[i], acc_x_rms_dict[i], where="mid", label=f"Sliding RMS (line) {i}s Intervall", linewidth=2)
axes[0].plot()
axes[0].set_ylabel('Acceleration [m/s²]')
axes[0].set_title('Norm-filtered x-acceleration')
axes[0].grid()
axes[0].legend()

axes[1].plot(df['time_rel'], df['acc_y'], label='Raw', alpha=0.7)
axes[1].plot(df['time_rel'], df['acc_y_normfilt'], label='Filtered', alpha = 0.7)
for i in intervals[:-1]:
    axes[1].step(y_interval_centers[i], acc_y_rms_dict[i], where="mid", label=f"Sliding RMS (line) {i}s Intervall", linewidth=2)
axes[1].set_ylabel('Acceleration [m/s²]')
axes[1].set_title('Norm-filtered y-acceleration')
axes[1].grid()
axes[1].legend()

axes[2].plot(df['time_rel'], df['yaw_acc'], label='Raw', alpha=0.7)
axes[2].plot(df['time_rel'], df['yaw_acc_normfilt'], label='Filtered', alpha = 0.7)
for i in intervals[:-1]:
    axes[2].step(yaw_interval_centers[i], yaw_acc_rms_dict[i], where="mid", label=f"Sliding RMS (line) 2achse{i}s Intervall", linewidth=2)
axes[2].set_ylabel('Acceleration [rad/s²]')
axes[2].set_xlabel('Time [s]')
axes[2].set_title('Norm-filtered yaw-acceleration')
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()

import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment


max_x = np.asarray(list(get_max_per_interval(df['time_rel'], df['acc_x_filt']).values()))
max_y = np.asarray(list(get_max_per_interval(df['time_rel'], df['acc_y_filt']).values()))
max_yaw = np.asarray(list(get_max_per_interval(df['time_rel'], df['yaw_filt']).values()))
t_total_int = len(max_x)

# Signale und ihre Methode1 Werte
signals_m1 = {
    "acc_x": max_x,
    "acc_y": max_y,
    "acc_yaw": max_yaw
}

# Methode 2 Werte (Dicts)
signals_m2 = {
    "acc_x": sum_x,
    "acc_y": sum_y,
    "acc_yaw": sum_yaw
}

# Methode 3 Werte (Dicts)
signals_m3 = {
    "acc_x": acc_x_rms_dict,
    "acc_y": acc_y_rms_dict,
    "acc_yaw": yaw_acc_rms_dict
}

# === Farben definieren ===
red_fill = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")  # hellrot
white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid") # weiß

# Neues Workbook erstellen
wb = openpyxl.Workbook()
first = True

for signal_name in ["acc_x", "acc_y", "acc_yaw"]:
    if first:
        ws = wb.active
        ws.title = signal_name
        first = False
    else:
        ws = wb.create_sheet(title=signal_name)
    
    # ---- Spalte 1: Methoden ----
    ws.cell(row=1, column=1, value="Zeitleiste in 1s Intervallen")
    ws.cell(row=2, column=1, value="Methode 1 (Maxwerte pro Intervall)")
    ws.cell(row=3, column=1, value="Methode 2 (1s Intervall)")
    ws.cell(row=4, column=1, value="Methode 2 (1s Intervall)")
    ws.cell(row=5, column=1, value="Methode 2 (3s Intervall)")
    ws.cell(row=6, column=1, value="Methode 2 (3s Intervall)")
    ws.cell(row=7, column=1, value="Methode 2 (5s Intervall)")
    ws.cell(row=8, column=1, value="Methode 2 (5s Intervall)")
    ws.cell(row=9, column=1, value="Methode 2 (t_total)")
    ws.cell(row=10, column=1, value="Methode 3 (1s Intervall)")
    ws.cell(row=11, column=1, value="Methode 3 (1s Intervall)")
    ws.cell(row=12, column=1, value="Methode 3 (2s Intervall)")
    ws.cell(row=13, column=1, value="Methode 3 (2s Intervall)")
    ws.cell(row=14, column=1, value="Methode 3 (5s Intervall)")
    ws.cell(row=15, column=1, value="Methode 3 (5s Intervall)")
    ws.cell(row=16, column=1, value="Methode 3 (t_total)")


    # ---- Zeile 1: 1s Intervalle ----
    for i in range(1, t_total_int, 2):
        ws.merge_cells(start_row=1, start_column=i+1, end_row=1, end_column=min(i+2, t_total_int*2))
        ws.cell(row=1, column=i+1, value=f"{int(i/2)}-{int(i/2)+1}s")
        ws.column_dimensions[get_column_letter(i+1)].width = 10
        ws.column_dimensions[get_column_letter(i+2)].width = 10
        ws.cell(row=1, column=i+1).alignment = Alignment(horizontal='center')
    ws.row_dimensions[1].height = 20
    
    # ---- Zeile 2: Methode 1 ----
    m1_values = signals_m1[signal_name]
    for i in range(1, t_total_int, 2):
        ws.merge_cells(start_row=2, start_column=i+1, end_row=2, end_column=min(i+2, t_total_int*2))
        ws.cell(row=2, column=i+1, value=round(m1_values[int(i/2)], 3))
        cell = ws.cell(row=2, column=i+1)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m1']:
            cell.fill = red_fill
    
    # ---- Methode 2 ----
    m2_dict = signals_m2[signal_name]
    # 1s Intervalle
    col = 2
    for i in range(1, len(m2_dict[1.0])-1, 2):
        ws.merge_cells(start_row=3, start_column=col, end_row=3, end_column=min(col+1, t_total_int*2))
        ws.cell(row=3, column=col, value=round(m2_dict[1.0][i], 3))
        cell = ws.cell(row=3, column=col)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m2'][0]:
            cell.fill = red_fill
        ws.merge_cells(start_row=4, start_column=col+1, end_row=4, end_column=min(col+2, t_total_int*2))
        ws.cell(row=4, column=col+1, value=round(m2_dict[1.0][i+1], 3))
        cell = ws.cell(row=4, column=col+1)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m2'][0]:
            cell.fill = red_fill
        col += 2
    # 3s Intervalle
    col = 2
    for i in range(1, len(m2_dict[3.0])-1, 2):
        ws.merge_cells(start_row=5, start_column=col, end_row=5, end_column=min(col+5, t_total_int*2))
        ws.cell(row=5, column=col, value=round(m2_dict[3.0][i], 3))
        cell = ws.cell(row=5, column=col)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m2'][1]:
            cell.fill = red_fill
        ws.merge_cells(start_row=6, start_column=col+3, end_row=6, end_column=min(col+8, t_total_int*2))
        ws.cell(row=6, column=col+3, value=round(m2_dict[3.0][i+1], 3))
        cell = ws.cell(row=6, column=col+3)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m2'][1]:
            cell.fill = red_fill
        col += 6
    # 5s Intervalle
    col = 2
    for i in range(1, len(m2_dict[5.0])-1, 2):
        ws.merge_cells(start_row=7, start_column=col, end_row=7, end_column=min(col+9, t_total_int*2))
        ws.cell(row=7, column=col, value=round(m2_dict[5.0][i], 3))
        cell = ws.cell(row=7, column=col)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m2'][2]:
            cell.fill = red_fill
        ws.merge_cells(start_row=8, start_column=col+5, end_row=8, end_column=min(col+14, t_total_int*2))
        ws.cell(row=8, column=col+5, value=round(m2_dict[5.0][i+1], 3))
        cell = ws.cell(row=8, column=col+5)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m2'][2]:
            cell.fill = red_fill
        col += 10
    # t_total
    ws.merge_cells(start_row=9, start_column=2, end_row=9, end_column=t_total_int*2)
    ws.cell(row=9, column=2, value=round(m2_dict[t_total][0], 3))
    cell = ws.cell(row=9, column=1)
    cell.alignment = Alignment(horizontal='center')
    
    # ---- Methode 3 ----
    m3_dict = signals_m3[signal_name]
    # 1s Intervalle
    col = 2
    for i in range(1, len(m3_dict[1.0])-1, 2):
        ws.merge_cells(start_row=10, start_column=col, end_row=10, end_column=min(col+1, t_total_int*2))
        ws.cell(row=10, column=col, value=round(m3_dict[1.0][i], 3))
        cell = ws.cell(row=10, column=col)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m3'][3]:
            cell.fill = red_fill
        ws.merge_cells(start_row=11, start_column=col+1, end_row=11, end_column=min(col+2, t_total_int*2))
        ws.cell(row=11, column=col+1, value=round(m3_dict[1.0][i+1], 3))
        cell = ws.cell(row=11, column=col+1)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m3'][3]:
            cell.fill = red_fill
        col += 2
    # 3s Intervalle
    col = 2
    for i in range(1, len(m3_dict[3.0])-1, 2):
        ws.merge_cells(start_row=12, start_column=col, end_row=12, end_column=min(col+5, t_total_int*2))
        ws.cell(row=12, column=col, value=round(m3_dict[3.0][i], 3))
        cell = ws.cell(row=12, column=col)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m3'][3]:
            cell.fill = red_fill
        ws.merge_cells(start_row=13, start_column=col+3, end_row=13, end_column=min(col+8, t_total_int*2))
        ws.cell(row=13, column=col+3, value=round(m3_dict[3.0][i+1], 3))
        cell = ws.cell(row=13, column=col+3)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m3'][3]:
            cell.fill = red_fill
        col += 6
    # 5s Intervalle
    col = 2
    for i in range(1, len(m3_dict[5.0])-1, 2):
        ws.merge_cells(start_row=14, start_column=col, end_row=14, end_column=min(col+9, t_total_int*2))
        ws.cell(row=14, column=col, value=round(m3_dict[5.0][i], 3))
        cell = ws.cell(row=14, column=col)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m3'][3]:
            cell.fill = red_fill
        ws.merge_cells(start_row=15, start_column=col+5, end_row=15, end_column=min(col+14, t_total_int*2))
        ws.cell(row=15, column=col+5, value=round(m3_dict[5.0][i+1], 3))
        cell = ws.cell(row=15, column=col+5)
        cell.alignment = Alignment(horizontal='center')
        if cell.value > thresholds_dict[signal_name]['m3'][3]:
            cell.fill = red_fill
        col += 10
    # t_total
    ws.merge_cells(start_row=16, start_column=2, end_row=16, end_column=t_total_int*2)
    ws.cell(row=16, column=2, value=round(m3_dict[t_total][0], 3))
    cell = ws.cell(row=16, column=2)
    cell.alignment = Alignment(horizontal='center')
    if cell.value > thresholds_dict[signal_name]['m3'][3]:
            cell.fill = red_fill

# Excel speichern
wb.save(r"C:\Users\kompa\Documents\Uni\TUM\Bachelorarbeit\Codebase\IMU_Methoden_Vergleich_Signale.xlsx")
print("Excel-Datei 'IMU_Methoden_Vergleich_Signale.xlsx' erstellt!")

