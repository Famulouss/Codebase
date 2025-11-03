import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from scipy.signal import butter, filtfilt
import matplotlib.colors as mcolors
import scipy.signal as signal
import tkinter as tk
from scipy.signal import butter, filtfilt, lfilter, freqz
from tkinter import ttk, filedialog
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
import os
import seaborn as sns

target_tpr = 0.9

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

def is_uncomfortable(time_value):
    """Prüft, ob Zeitwert innerhalb eines unkomfortablen Intervalls liegt."""
    # Hole den Label für den Zeitwert (nächster Zeitindex)
    idx = (df['time_rel'] - time_value).abs().idxmin()
    return df.loc[idx, 'discomfort'] == 1

def berechne_grenzwert(signal, label, p=90):
    if label.sum() == 0:
        return np.nan
    return np.percentile(signal[label == 1], p)

def plot_roc_for_dicts(signal_dict, centers_dict, signal_name):
    """ Plottet die ROC Kurven für die Signal-Dictionaries von der Summations- und RMS-Methode.
        Format der Dictionaries: {Intervallgröße: Werte, ... }"""
    plt.figure(figsize=(10, 6))

    for interval, values in signal_dict.items():
        centers = centers_dict[interval]

        # Labels erzeugen
        y_true = [1 if is_uncomfortable(t) else 0 for t in centers]

        # ROC berechnen
        fpr, tpr, thresholds = roc_curve(y_true, values)
        roc_auc = auc(fpr, tpr)

        # Optimalen Grenzwert bestimmen (Youden-Index)
        optimal_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_index]

        # Finde Schwelle mit TPR >= target_tpr, die der Youden-Schwelle am nächsten ist
        idx_sens = np.where(tpr >= target_tpr)[0]
        if len(idx_sens) > 0:
            best_idx = idx_sens[0]  # erste Schwelle, die gewünschte Sensitivität erreicht
            best_threshold = thresholds[best_idx]

        # Plot
        plt.plot(fpr, tpr, label=f'{interval} (AUC = {roc_auc:.2f}, Opt. Th Youden= {optimal_threshold:.2f}, Opt. Th (TPR > 0,9) = {best_threshold:.2f})')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal (Youden): {optimal_threshold:.2f}')
        plt.scatter(fpr[best_idx], tpr[best_idx], color='blue', label=f'Optimal (TPR > 0,9): {best_threshold:.2f}')

    plt.plot([0, 1], [0, 1], 'k--', label='Zufall')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-Kurven für {signal_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =====================================================
# 1️⃣  UNKOMFORT-ZEITEN EINTRAGEN (manuell anpassbar)
# =====================================================
# Liste mit Start- und Endzeiten (Sekunden)
unangenehme_zeiten = [
    (48, 50),
    (60, 62),
    (73.5, 75.5),
    (95, 97),
    (101.4, 103.4)
]

# =====================================================
# 2️⃣  DATEN EINLESEN
# =====================================================
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
acc_yaw = np.gradient(a, t)
df['acc_yaw'] = acc_yaw

acc_yaw_cut = 1.0   # Hz
df['acc_yaw_filt'] = lowpass_filter(df['acc_yaw'].values, acc_yaw_cut, fs_glob, order=2)

# -------------------
# Jerk berechnen
# -------------------
for axis in ['x', 'y']:
    a = df[f'acc_{axis}_filt'].values
    t = df['time_rel'].values
    jerk = np.gradient(a, t)
    df[f'jerk_{axis}_raw'] = jerk

jerk_cut = 1.0  # Hz
df["jerk_x_filt"] = lowpass_filter(df["jerk_x_raw"].values, jerk_cut, fs_glob, order=2)
df["jerk_y_filt"] = lowpass_filter(df["jerk_y_raw"].values, jerk_cut, fs_glob, order=2)


# =====================================================
# 3️⃣  LABEL ERSTELLEN
# =====================================================
df['discomfort'] = 0  # Default = komfortabel

for start, end in unangenehme_zeiten:
    df.loc[(df['time_rel'] >= start) & (df['time_rel'] <= end), 'discomfort'] = 1

print(f"✅ {df['discomfort'].sum()} Zeitpunkte als 'unkomfortabel' markiert.")

# =====================================================
# 4️⃣  PERZENTIL-GRENZWERTE
# =====================================================
grenzwerte = {
    'acc_x_filt': berechne_grenzwert(df['acc_x_filt'], df['discomfort']),
    'acc_y_filt': berechne_grenzwert(df['acc_y_filt'], df['discomfort']),
    'acc_yaw_filt': berechne_grenzwert(df['acc_yaw_filt'], df['discomfort']),
    'jerk_x_filt': berechne_grenzwert(df['jerk_x_filt'], df['discomfort']),
    'jerk_y_filt': berechne_grenzwert(df['jerk_y_filt'], df['discomfort']),
}

print("\n📊 Empirische 90%-Grenzwerte (aus unangenehmen Phasen):")
for key, val in grenzwerte.items():
    print(f"{key:10s}: {val:.3f} m/s²")

# =====================================================
# 6️⃣  Summen-Berechnung
# =====================================================
# Intervallweise berechnen (1s, 3s, 5s)
t_total = int(df['time_rel'].iloc[-1])
intervals = [1.0, 3.0, 5.0, t_total]
sum_x, sum_x_interval_centers = {}, {}
sum_y, sum_y_interval_centers = {}, {}
sum_acc_yaw, sum_acc_yaw_interval_centers = {}, {}
sum_jerk_x, sum_jerk_x_interval_centers = {}, {}
sum_jerk_y, sum_jerk_y_interval_centers = {}, {}
for i in intervals:
    temp1, temp2 = sliding_metrics(df['acc_x_filt'], df['time_rel'], i)
    sum_x[i] = temp1
    sum_x_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['acc_y_filt'], df['time_rel'], i)
    sum_y[i] = temp1
    sum_y_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['acc_yaw_filt'], df['time_rel'], i)
    sum_acc_yaw[i] = temp1
    sum_acc_yaw_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['jerk_x_filt'], df['time_rel'], i)
    sum_jerk_x[i] = temp1
    sum_jerk_x_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['jerk_y_filt'], df['time_rel'], i)
    sum_jerk_y[i] = temp1
    sum_jerk_y_interval_centers[i] = temp2    
sum_all = {'acc_x': sum_x, 'acc_y': sum_y, 'acc_yaw': sum_acc_yaw, 'jerk_x': sum_jerk_x, 'jerk_y': sum_jerk_y}
sum_centers_all = {'acc_x': sum_x_interval_centers, 
                   'acc_y': sum_y_interval_centers, 
                   'acc_yaw': sum_acc_yaw_interval_centers, 
                   'jerk_x': sum_jerk_x_interval_centers, 
                   'jerk_y': sum_jerk_y_interval_centers}
# =====================================================
# 6️⃣  Summen-Darstellung
# =====================================================
# === PARAMETER ===
signale = ['acc_x', 'acc_y', 'acc_yaw', 'jerk_x', 'jerk_y']
intervals = [1.0, 3.0, 5.0]

# === HAUPTSCHLEIFE ===
for sig in signale:
    for interval in intervals:
        centers = sum_x_interval_centers[interval]
        sums = sum_all[sig][interval]

        # komfortabel / unkomfortabel trennen
        comfortable_vals = []
        uncomfortable_vals = []

        for c, val in zip(centers, sums):
            if is_uncomfortable(c):
                uncomfortable_vals.append(val)
            else:
                comfortable_vals.append(val)

        # === GRENZWERT berechnen (z. B. 90. Perzentil der unkomfortablen Intervalle) ===
        if len(uncomfortable_vals) > 0:
            grenzwert = np.percentile(uncomfortable_vals, 90)
        else:
            grenzwert = np.nan

        # === PLOT ===
        plt.figure(figsize=(8, 5))
        bins = 30  # Anzahl der Bins anpassbar
        plt.hist(comfortable_vals, bins=bins, alpha=0.6, color='skyblue', label='Komfortabel')
        plt.hist(uncomfortable_vals, bins=bins, alpha=0.6, color='salmon', label='Unkomfortabel')
        plt.axvline(grenzwert, color='red', linestyle='--', label=f'Grenzwert {grenzwert:.2f}')
        plt.title(f"Histogramm Summationswerte ({sig.upper()}-Achse, {interval}s)")
        plt.xlabel("Summationswert [m/s² * s]")
        plt.ylabel("Auftretungshäufigkeit")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Anzeigen und speichern
        plt.tight_layout()
        plt.show()
# =====================================================
# 6️⃣  RMS-Berechnung
# =====================================================
b, a = design_wd_filter(fs_glob)     # Gewichtung ist laut Norm für die x- und y-Beschleunigung diesselbe
df['acc_x_normfilt'] = filtfilt(b, a, df['acc_x'])
df['acc_y_normfilt'] = filtfilt(b, a, df['acc_y'])
b, a = design_we_filter(fs_glob)
df['acc_yaw_normfilt'] = filtfilt(b, a, df['acc_yaw'])
df['acc_yaw_normfilt'] *= 0.2   # k-Faktor nach Norm [m/rad]

# Berechne die rms Werte in unterschiedlich großen gleitenden Intervallen
# Pro Beschleunigungsrichtung gibt es ein dict welches als Schlüssel die jeweiligen Intervalle hat {1.0: [0.674, 0.121, ...], 3.0: [...], ...}
acc_x_rms_dict, x_interval_centers = {}, {}
acc_y_rms_dict, y_interval_centers = {}, {}
acc_yaw_rms_dict, yaw_interval_centers = {}, {}
for i in intervals:
    temp1, temp2 = sliding_metrics(df['acc_x_normfilt'], df['time_rel'], i, weighted=True)
    acc_x_rms_dict[i] = temp1
    x_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['acc_y_normfilt'], df['time_rel'], i, weighted=True)
    acc_y_rms_dict[i] = temp1
    y_interval_centers[i] = temp2
    temp1, temp2 = sliding_metrics(df['acc_yaw_normfilt'], df['time_rel'], i, weighted=True)
    acc_yaw_rms_dict[i] = temp1
    yaw_interval_centers[i] = temp2
rms_all = {'acc_x': acc_x_rms_dict, 'acc_y': acc_y_rms_dict, 'acc_yaw': acc_yaw_rms_dict}

# =====================================================
# 6️⃣  RMS-DARSTELLUNG
# =====================================================
for sig in ['acc_x', 'acc_y', 'acc_yaw']:
    plt.figure(figsize=(10, 6))
    plt.plot(df['time_rel'], df[f'{sig}_filt'], alpha=0.5, label='Signal')
    for win in intervals:
        plt.plot(x_interval_centers[win], rms_all[sig][win],
                 label=f'RMS {win:.0f}s')
    plt.title(f"RMS über Zeitfenster – {sig}")
    plt.xlabel("Zeit [s]")
    plt.ylabel(f"{sig} RMS [m/s²]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =====================================================
# 5️⃣  HISTOGRAMM-VERGLEICH
# =====================================================
signale = ['acc_x_filt', 'acc_y_filt', 'acc_yaw_filt', 'jerk_x_filt', 'jerk_y_filt']

for s in signale:
    plt.figure(figsize=(6, 3))
    plt.hist(df.loc[df['discomfort'] == 0, s], bins=40, alpha=0.6, label='Komfortabel')
    plt.hist(df.loc[df['discomfort'] == 1, s], bins=40, alpha=0.6, label='Unkomfortabel')
    plt.axvline(grenzwerte[s], color='red', linestyle='--', label=f'Grenzwert {grenzwerte[s]:.2f}')
    plt.title(f"Histogramm – {s}")
    plt.xlabel(s + " [m/s²]")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =====================================================
# 6️⃣  ROC-ANALYSE
# =====================================================
optimal_thresholds = {}
for s in signale:
    fpr, tpr, thresholds = roc_curve(df['discomfort'], df[s])
    roc_auc = auc(fpr, tpr)

    # 👉 Youden-Index berechnen
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_thresholds[s] = optimal_threshold
    
    # Finde Schwelle mit TPR >= target_tpr, die der Youden-Schwelle am nächsten ist
    idx_sens = np.where(tpr >= target_tpr)[0]
    if len(idx_sens) > 0:
        best_idx = idx_sens[0]  # erste Schwelle, die gewünschte Sensitivität erreicht
        best_threshold = thresholds[best_idx]

    # ROC-Plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal (Youden): {optimal_threshold:.2f}')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='blue', label=f'Optimal (TPR > 0,9): {best_threshold:.2f}')
    plt.title(f"ROC-Kurve – {s}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n🎯 Optimale Grenzwerte (Youden-Index):")
for s, val in optimal_thresholds.items():
    print(f"{s:10s}: {val:.3f} m/s²")

signale = ['acc_x', 'acc_y', 'acc_yaw', 'jerk_x', 'jerk_y']

for sig in signale:
    plot_roc_for_dicts(sum_all[sig], sum_centers_all[sig], f"Summationswerte - {sig}")

for sig in signale[:-2]:
    plot_roc_for_dicts(rms_all[sig], sum_centers_all[sig], f"RMS-Werte - {sig}")

# =====================================================
# 7️⃣  Precision-Recall Kurve
# =====================================================
# Zeitbasierte Signale
y_scores = {}
# Wahre Labels (y_true)
y_true = df['discomfort'].values 

for sig in signale:
    # Signalwerte (y_scores)
    y_scores[sig] = df[f'{sig}_filt'].values 
    
    # Plot
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores[sig])
    ap = average_precision_score(y_true, y_scores[sig])

    plt.plot(recall, precision, label=f'{sig} (AP={ap:.2f})')
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(f'Precision-Recall Kurve {sig}_Signal')
    plt.grid(True)

# Summationswerte
for sig in sum_all.keys():
    for interval in sig.keys(): 
        y_scores= np.array(sum_all[sig][interval])  # Summationswerte
        y_true = np.array([
        1 if is_uncomfortable(t) else 0
        for t in sum_centers_all[sig][interval]
        ])
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, label=f'{signal_name} (AP={ap:.2f})')
        
    plt.plot(recall, precision, label=f'{sig} (AP={ap:.2f})')
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision-Recall Kurve {sig}-Signal')
    plt.grid(True)
    

# RMS-WErte

# =====================================================
# 7️⃣  VERGLEICH MIT LITERATUR
# =====================================================
literatur = {
    'acc_x': 1.23,
    'acc_y': 0.98,
}

vergleich = pd.DataFrame({
    'Parameter': grenzwerte.keys(),
    'Grenzwert_eigeneMessung': grenzwerte.values(),
    'Grenzwert_Literatur': [literatur.get(k, np.nan) for k in grenzwerte.keys()]
})

vergleich['Abweichung_%'] = 100 * (vergleich['Grenzwert_eigeneMessung'] - vergleich['Grenzwert_Literatur']) / vergleich['Grenzwert_Literatur']
print("\n📋 Vergleich mit Literatur:")
print(vergleich.round(3))
