import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from scipy.signal import butter, filtfilt
import scipy.signal as signal
import tkinter as tk
from scipy.signal import butter, filtfilt
from tkinter import ttk
import os
import json

person_id = 'VP1'
fahrt_id = 'Szenario 3'
target_tpr = 0.9
grenzwerte_gesamt = {}
delta_time = 0.5

# =====================================================
# UNKOMFORTABLE-ZEITEN EINTRAGEN (manuell anpassbar)
# =====================================================
# Liste mit Start- und Endzeiten (Sekunden)
unangenehme_zeiten = [18, 20, 22, 58, 69]
unangenehme_intervalle = []
for i, time in enumerate(unangenehme_zeiten):
    unangenehme_intervalle.append((time-delta_time, time+delta_time))

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

def timestamp_is_uncomfortable(time_value):
    """Prüft, ob Zeitwert innerhalb eines unkomfortablen Intervalls liegt."""
    # Hole das Label für den Zeitwert (nächster Zeitindex)
    idx = (df['time_rel'] - time_value).abs().idxmin()
    return df.loc[idx, 'discomfort'] == 1

def interval_is_uncomfortable(time_value, interval_length=1.0):
    """
    Prüft, ob innerhalb eines gegebenen Zeitintervalls ein unkomfortabler Zeitstempel liegt.

    Parameter:
        time_value (float): Zentrum des Intervalls (in Sekunden)
        interval_length (float): Länge des Intervalls (z. B. 1.0 für 1 s)
    
    Rückgabe:
        bool: True, wenn im Intervall ein unkomfortabler Wert liegt, sonst False
    """

    # Intervallgrenzen berechnen
    half = interval_length / 2
    start = time_value - half
    end = time_value + half

    # Daten innerhalb des Intervalls auswählen
    mask = (df['time_rel'] >= start) & (df['time_rel'] <= end)
    interval_data = df.loc[mask, 'discomfort']

    # Prüfen, ob irgendein Wert in diesem Intervall unkomfortabel ist
    return (interval_data == 1).any()

def berechne_grenzwert(signal, label, p=90):
    if label.sum() == 0:
        return np.nan
    return np.percentile(signal[label == 1], p)

def plot_roc_pr_for_dicts(signal_dict, centers_dict, signal_name, sig_n, methode):
    """Plottet ROC- & PR-Kurven pro Intervall (1s, 3s, 5s) für Summations- oder RMS-Daten."""
    intervals = list(signal_dict.keys())
    fig, axes = plt.subplots(1, len(intervals), figsize=(18, 5))

    if len(intervals) == 1:
        axes = [axes]  # falls nur ein Intervall vorhanden ist

    for ax, interval in zip(axes, intervals):
        values = signal_dict[interval]
        centers = centers_dict[interval]

        # Labels erzeugen
        y_true = np.array([1 if timestamp_is_uncomfortable(t) else 0 for t in centers])
        y_scores = np.array(values)

        # ROC
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Optimaler Schwellenwert (Youden-Index)
        youden_index = tpr - fpr
        best_youden_idx = np.argmax(youden_index)
        best_youden_threshold = roc_thresholds[best_youden_idx]
        
        # Finde Schwelle mit TPR >= target_tpr, die der Youden-Schwelle am nächsten ist
        idx_sens = np.where(tpr >= target_tpr)[0]
        if len(idx_sens) > 0:
            best_roc_idx = idx_sens[0]  # erste Schwelle, die gewünschte Sensitivität erreicht
            best_roc_threshold = roc_thresholds[best_roc_idx]

        # Precision-Recall
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        # F1-Werte für alle Thresholds berechnen
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # +1e-8 verhindert Division durch 0
        # Index des maximalen F1-Werts finden
        best_f1_idx = np.argmax(f1)
        best_f1 = f1[best_f1_idx]
        best_f1_threshold = pr_thresholds[best_f1_idx]

        # Nächstgelegenen ROC-Punkt für den besten F1-Threshold finden
        roc_idx = np.argmin(np.abs(roc_thresholds - best_f1_threshold))
        
        # Grenzwerte abspeichern
        grenzwerte_gesamt[f'{sig_n}_filt_abs'][methode] = {}
        grenzwerte_gesamt[f'{sig_n}_filt_abs'][methode][interval] = {}
        grenzwerte_gesamt[f'{sig_n}_filt_abs'][methode][interval]['F1'] = best_f1_threshold
        grenzwerte_gesamt[f'{sig_n}_filt_abs'][methode][interval]['Youden'] = best_youden_threshold
        grenzwerte_gesamt[f'{sig_n}_filt_abs'][methode][interval][f'TPR>={target_tpr}'] = best_roc_threshold

        # ROC (linke y-Achse)
        color_roc = 'darkorange'
        ax.plot(fpr, tpr, color=color_roc, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.scatter(fpr[best_youden_idx], tpr[best_youden_idx], color='red', label=f'Youden: {best_youden_threshold:.2f}')
        ax.scatter(fpr[best_roc_idx], tpr[best_roc_idx], color='blue', label=f'TPR ≥ 0.9: {best_roc_threshold:.2f}')
        # Besten F1 Punkt in ROC-Kurve markieren
        ax.plot(fpr[roc_idx], tpr[roc_idx], 'o', 
                color='violet', markersize=8, 
                label=f'Best F1 threshold on ROC')
        ax.set_xlabel('False Positive Rate & Recall')
        ax.set_ylabel('True Positive Rate', color=color_roc)
        ax.tick_params(axis='y', labelcolor=color_roc)
        ax.grid(True)

        # Precision-Recall (rechte y-Achse)
        ax2 = ax.twinx()
        color_pr = 'tab:blue'
        ax2.plot(recall, precision, color=color_pr, lw=2, linestyle='--', label=f'PR (AP = {ap:.2f})')
        ax2.set_ylabel('Precision', color=color_pr)
        ax2.tick_params(axis='y', labelcolor=color_pr)
        # Besten F1 Punkt in der P/R-Kurve markieren
        ax2.plot(recall[best_f1_idx], precision[best_f1_idx], 'o', 
                color='red', markersize=8, 
                label=f'Best F1={best_f1:.2f} @ thr={best_f1_threshold:.2f}')
        # Titel und Legenden
        ax.set_title(f"{interval}s-Intervall")

        # Gemeinsame Legende pro Subplot
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='lower right')

    fig.suptitle(f"ROC & Precision-Recall – {signal_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

def save_thresholds(all_thresholds, person_id, fahrt_id, save_dir="grenzwerte"):
    """
    Speichert die Grenzwerte pro Fahrt und Person als JSON-Datei.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/person_{person_id}_{fahrt_id}.json"

    data = {
        "person_id": person_id,
        "fahrt_id": fahrt_id,
        "thresholds": all_thresholds
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Grenzwerte gespeichert in: {filename}")

def plot_two_normalized_histograms(data1, data2, grenzwert, bins=40, label1="Komfortabel", label2="Unkomfortabel",
                                   title="Normalisiertes Histogramm", xlabel="Wert", color1='tab:blue', color2='tab:orange'):
    """
    Plottet zwei Histogramme in normalisierter Form (Summe der Balkenhöhen = 1).

    Parameter:
        data1, data2: array-ähnlich
            Die beiden Datensätze (z. B. komfortabel / unkomfortabel)
        bins: int oder Sequenz
            Anzahl oder Grenzen der Bins
        label1, label2: str
            Beschriftungen der beiden Histogramme
        title: str
            Titel des Plots
        xlabel: str
            Achsenbeschriftung (x-Achse)
        color1, color2: str
            Farben der beiden Histogramme
    """

    # --- Histogramme berechnen ---
    counts1, bins1 = np.histogram(data1, bins=bins)
    counts2, bins2 = np.histogram(data2, bins=bins)

    # --- Normalisierung (Summe der Balkenhöhen = 1) ---
    counts1 = counts1 / np.sum(counts1) if np.sum(counts1) != 0 else counts1
    counts2 = counts2 / np.sum(counts2) if np.sum(counts2) != 0 else counts2

    if "acc_yaw" in title:
        grenzwert_einheit = 'rad/s²'
    elif "yaw" in title:
        grenzwert_einheit = 'rad/s'
    elif "jerk" in title:
        grenzwert_einheit = 'm/s³'
    elif 'acc' in title:
        grenzwert_einheit = 'm/s²'

    # --- Plot ---
    plt.figure(figsize=(6, 3))

    plt.hist(data1, bins=bins, alpha=0.6, label=label1, color=color1, weights=np.ones(len(data1))/len(data1))
    plt.hist(data2, bins=bins, alpha=0.6, label=label2, color=color2, weights=np.ones(len(data2))/len(data2))
    plt.axvline(grenzwert, color='red', linestyle='--', label=f' 90%-Grenzwert = {grenzwert:.2f} {grenzwert_einheit}')

    plt.xlabel(xlabel)
    plt.ylabel("Relative Häufigkeit (Σ=1)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

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

# Absolutwerte der Signale bilden
signale = ['acc_x_filt', 'acc_y_filt', 'acc_yaw_filt', 'yaw_filt', 'jerk_x_filt', 'jerk_y_filt']
for s in signale:
    df[f'{s}_abs'] = np.abs(df[s])


# =====================================================
# 3️⃣  LABEL ERSTELLEN
# =====================================================
df['discomfort'] = 0  # Default = komfortabel

for start, end in unangenehme_intervalle:
    df.loc[(df['time_rel'] >= start) & (df['time_rel'] <= end), 'discomfort'] = 1

print(f"{df['discomfort'].sum()} Zeitpunkte als 'unkomfortabel' markiert.")

# =====================================================
# 4️⃣  PERZENTIL-GRENZWERTE
# =====================================================
grenzwerte = {
    'acc_x_filt_abs': berechne_grenzwert(df['acc_x_filt_abs'], df['discomfort']),
    'acc_y_filt_abs': berechne_grenzwert(df['acc_y_filt_abs'], df['discomfort']),
    'acc_yaw_filt_abs': berechne_grenzwert(df['acc_yaw_filt_abs'], df['discomfort']),
    'yaw_filt_abs': berechne_grenzwert(df['yaw_filt_abs'], df['discomfort']),
    'jerk_x_filt_abs': berechne_grenzwert(df['jerk_x_filt_abs'], df['discomfort']),
    'jerk_y_filt_abs': berechne_grenzwert(df['jerk_y_filt_abs'], df['discomfort'])
}

print("\nEmpirische 90%-Grenzwerte (aus unangenehmen Phasen):")
for key, val in grenzwerte.items():
    print(f"{key:10s}: {val:.3f} m/s²")

# =====================================================
# 5️⃣  HISTOGRAMM-VERGLEICH
# =====================================================
bins = 30  # Anzahl der Bins anpassbar

plot_two_normalized_histograms(df.loc[df['discomfort'] == 0, 'acc_x_filt_abs'], 
                               df.loc[df['discomfort'] == 1, 'acc_x_filt_abs'], 
                               grenzwerte['acc_x_filt_abs'], 
                               title=f"Histogramm - acc_x_filt", 
                               xlabel='Beschleunigungswerte [m/s²]')
plot_two_normalized_histograms(df.loc[df['discomfort'] == 0, 'acc_y_filt_abs'], 
                               df.loc[df['discomfort'] == 1, 'acc_y_filt_abs'], 
                               grenzwerte['acc_y_filt_abs'], 
                               title=f"Histogramm - acc_y_filt", 
                               xlabel='Beschleunigungswerte [m/s²]')
plot_two_normalized_histograms(df.loc[df['discomfort'] == 0, 'acc_yaw_filt_abs'], 
                               df.loc[df['discomfort'] == 1, 'acc_yaw_filt_abs'], 
                               grenzwerte['acc_yaw_filt_abs'], 
                               title=f"Histogramm - acc_yaw_filt", 
                               xlabel='Rotationsbeschleunigung [rad/s²]')
plot_two_normalized_histograms(df.loc[df['discomfort'] == 0, 'yaw_filt_abs'], 
                               df.loc[df['discomfort'] == 1, 'yaw_filt_abs'], 
                               grenzwerte['yaw_filt_abs'], 
                               title=f"Histogramm - yaw_filt", 
                               xlabel='Rotationsgeschwindigkeit [rad/s]')
plot_two_normalized_histograms(df.loc[df['discomfort'] == 0, 'jerk_x_filt_abs'], 
                               df.loc[df['discomfort'] == 1, 'jerk_x_filt_abs'], 
                               grenzwerte['jerk_x_filt_abs'], 
                               title=f"Histogramm - jerk_x_filt", 
                               xlabel='Jerkwerte [m/s³]')
plot_two_normalized_histograms(df.loc[df['discomfort'] == 0, 'jerk_y_filt_abs'], 
                               df.loc[df['discomfort'] == 1, 'jerk_y_filt_abs'], 
                               grenzwerte['jerk_y_filt_abs'], 
                               title=f"Histogramm - jerk_y_filt", 
                               xlabel='Beschleunigungswerte [m/s³]')


# =====================================================
# 6️⃣  Summen-Berechnung
# =====================================================
# Intervallweise berechnen (1s, 3s, 5s)
t_total = int(df['time_rel'].iloc[-1])
intervals = [1.0, 3.0, 5.0]
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
            if interval_is_uncomfortable(c, interval):
                uncomfortable_vals.append(val)
            else:
                comfortable_vals.append(val)

        # === GRENZWERT berechnen (z. B. 90. Perzentil der unkomfortablen Intervalle) ===
        if len(uncomfortable_vals) > 0:
            grenzwert = np.percentile(uncomfortable_vals, 90)
        else:
            grenzwert = np.nan

        # === PLOT ===
        if sig in ['acc_x', 'acc_y']:
            xlabel = "Summationswert [m/s²]"
        elif sig in ['acc_yaw']:
            xlabel = "Summationswert [rad/s²]"
        elif sig in ['jerk_x', 'jerk_y']:
            xlabel = "Summationswert [m/s³]"
        plot_two_normalized_histograms(comfortable_vals, uncomfortable_vals, grenzwert, bins=bins, title=f"Histogramm Summationswerte ({sig}, {interval:.0f}s-Intervall)", xlabel=xlabel)

# =====================================================
# RMS-Berechnung
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
    if sig == 'acc_yaw':
        plt.ylabel(f"{sig} RMS [rad/s²]")
    else:
        plt.ylabel(f"{sig} RMS [m/s²]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =====================================================
# 6️⃣  ROC-P/R-ANALYSE
# =====================================================
signale = ['acc_x_filt_abs', 'acc_y_filt_abs', 'acc_yaw_filt_abs', 'yaw_filt_abs', 'jerk_x_filt_abs', 'jerk_y_filt_abs']

youden_thresholds = {}
for s in signale:
    y_true = df['discomfort']
    y_scores = df[s]

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 👉 Youden-Index berechnen
    youden_index = tpr - fpr
    best_youden_idx = np.argmax(youden_index)
    best_youden_threshold = roc_thresholds[best_youden_idx]
    youden_thresholds[s] = best_youden_threshold
    
    # Finde Schwelle mit TPR >= target_tpr, die der Youden-Schwelle am nächsten ist
    idx_sens = np.where(tpr >= target_tpr)[0]
    if len(idx_sens) > 0:
        best_roc_idx = idx_sens[0]  # erste Schwelle, die gewünschte Sensitivität erreicht
        best_roc_threshold = roc_thresholds[best_roc_idx]

    # P/R
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # F1-Werte für alle Thresholds berechnen
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # +1e-8 verhindert Division durch 0
    # Index des maximalen F1-Werts finden
    best_f1_idx = np.argmax(f1)
    best_f1 = f1[best_f1_idx]
    best_f1_threshold = pr_thresholds[best_f1_idx]

    # Nächstgelegenen ROC-Punkt für den besten F1-Threshold finden
    roc_idx = np.argmin(np.abs(roc_thresholds - best_f1_threshold))

    # Grenzwerte abspeichern
    grenzwerte_gesamt[s] = {}
    grenzwerte_gesamt [s]['M1'] = {}
    grenzwerte_gesamt[s]['M1']['F1'] = best_f1_threshold
    grenzwerte_gesamt[s]['M1']['Youden'] = best_youden_threshold
    grenzwerte_gesamt[s]['M1'][f'TPR>={target_tpr}'] = best_roc_threshold

    if "acc_yaw" in s:
        grenzwert_einheit = 'rad/s²'
    elif "yaw" in s:
        grenzwert_einheit = 'rad/s'
    elif "jerk" in s:
        grenzwert_einheit = 'm/s³'
    elif 'acc' in s:
        grenzwert_einheit = 'm/s²'

    # ROC-P/R-Plot
    fig, ax1 = plt.subplots()

    # ROC-Kurve (linke Achse)
    color_roc = 'darkorange'
    ax1.plot(fpr, tpr, color=color_roc, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax1.scatter(fpr[best_youden_idx], tpr[best_youden_idx], color='red', label=f'Youden-GW: {best_youden_threshold:.2f} {grenzwert_einheit}')
    ax1.scatter(fpr[best_roc_idx], tpr[best_roc_idx], color='blue', label=f'GW (TPR ≥ 0.9): {best_roc_threshold:.2f}  {grenzwert_einheit}')
    # Besten F1 Punkt in ROC-Kurve markieren
    plt.plot(fpr[roc_idx], tpr[roc_idx], 'o', 
            color='violet', markersize=8, 
            label=f'F1-GW auf ROC')
    ax1.set_xlabel('False Positive Rate & Recall')
    ax1.set_ylabel('True Positive Rate', color=color_roc)
    ax1.tick_params(axis='y', labelcolor=color_roc)
    ax1.grid(True)

    # Precision-Recall-Kurve (rechte Achse)
    ax2 = ax1.twinx()
    color_pr = 'tab:blue'
    ax2.plot(recall, precision, color=color_pr, lw=2, linestyle='--', label=f'PR (AP = {ap:.2f})')
    # Besten F1 Punkt in der P/R-Kurve markieren
    ax2.plot(recall[best_f1_idx], precision[best_f1_idx], 'o', 
            color='red', markersize=8, 
            label=f'Bester F1-Wert={best_f1:.2f} @ GW={best_f1_threshold:.2f} {grenzwert_einheit}')
    ax2.set_ylabel('Precision', color=color_pr)
    ax2.tick_params(axis='y', labelcolor=color_pr)

    # Titel und Legenden
    fig.suptitle(f"ROC & Precision-Recall – {s}")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right')

    plt.tight_layout()
    plt.show()

print("\nOptimale Grenzwerte (Youden-Index):")
for s, val in youden_thresholds.items():
    print(f"{s:10s}: {val:.3f} m/s²")

signale = ['acc_x', 'acc_y', 'acc_yaw', 'jerk_x', 'jerk_y']

for sig in signale:
    plot_roc_pr_for_dicts(sum_all[sig], sum_centers_all[sig], f"Summationswerte - {sig}", sig, 'M2')

for sig in signale[:-2]:
    plot_roc_pr_for_dicts(rms_all[sig], sum_centers_all[sig], f"RMS-Werte - {sig}", sig, 'M3')

# =====================================================
# Grenzwerte abspeichern (JSON)
# =====================================================
save_thresholds(grenzwerte_gesamt, person_id, fahrt_id)
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
print("\nVergleich mit Literatur:")
print(vergleich.round(3))
