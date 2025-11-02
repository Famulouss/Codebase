import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import matplotlib.colors as mcolors
import scipy.signal as signal
import tkinter as tk
from scipy.signal import butter, filtfilt, lfilter, freqz
from tkinter import Tk, ttk, filedialog
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
import os
import seaborn as sns


fs_glob = 10

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
def berechne_grenzwert(signal, label, name, p=90):
    if label.sum() == 0:
        return np.nan
    return np.percentile(signal[label == 1], p)

grenzwerte = {
    'acc_x_filt': berechne_grenzwert(df['acc_x_filt'], df['discomfort'], 'acc_x'),
    'acc_y_filt': berechne_grenzwert(df['acc_y_filt'], df['discomfort'], 'acc_y'),
    'yaw_acc_filt': berechne_grenzwert(df['yaw_acc_filt'], df['discomfort'], 'yaw_acc'),
    'jerk_x_filt': berechne_grenzwert(df['jerk_x_filt'], df['discomfort'], 'jerk_x'),
    'jerk_y_filt': berechne_grenzwert(df['jerk_y_filt'], df['discomfort'], 'jerk_y'),
}

print("\n📊 Empirische 90%-Grenzwerte (aus unangenehmen Phasen):")
for key, val in grenzwerte.items():
    print(f"{key:10s}: {val:.3f} m/s²")

# =====================================================
# 5️⃣  HISTOGRAMM-VERGLEICH
# =====================================================
signale = ['acc_x_filt', 'acc_y_filt', 'yaw_acc_filt', 'jerk_x_filt', 'jerk_y_filt']

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
for s in signale:
    fpr, tpr, thresholds = roc_curve(df['discomfort'], df[s])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.title(f"ROC-Kurve – {s}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
