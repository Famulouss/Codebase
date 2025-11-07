import json
import glob
import numpy as np
import os

signal = "jerk_y_filt_abs"
szenario = "Szenario 3"

analysed_thresholds = {}    # {'M1': {''}}
m1_thresh = {'F1': [], 'Youden': [], 'AUC': [], 'AP': []}
m2_thresh_1s = {'F1': [], 'Youden': [], 'AUC': [], 'AP': []}
m2_thresh_3s = {'F1': [], 'Youden': [], 'AUC': [], 'AP': []}
m2_thresh_5s = {'F1': [], 'Youden': [], 'AUC': [], 'AP': []}
m3_thresh_1s = {'F1': [], 'Youden': [], 'AUC': [], 'AP': []}
m3_thresh_3s = {'F1': [], 'Youden': [], 'AUC': [], 'AP': []}
m3_thresh_5s = {'F1': [], 'Youden': [], 'AUC': [], 'AP': []}
m2_thresh_all = {'1.0': m2_thresh_1s, '3.0': m2_thresh_3s, '5.0': m2_thresh_5s}
m3_thresh_all = {'1.0': m3_thresh_1s, '3.0': m3_thresh_3s, '5.0': m3_thresh_5s}
sections = [m1_thresh, m2_thresh_1s, m2_thresh_3s, m2_thresh_5s, m3_thresh_1s, m3_thresh_3s, m3_thresh_5s]

def save_thresholds(all_thresholds, signal_name, save_dir="GW-Analysis"):
    """
    Speichert die Grenzwertanalysis pro Signal als JSON-Datei.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{signal_name}.json"

    data = {
        "signal": signal_name,
        "thresholds_analysis": all_thresholds
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Grenzwertanalysis gespeichert in: {filename}")

def load_thresholds(signal_name, szenario='Szenario 1'):
    files = glob.glob("grenzwerte/*.json")
    values = []

    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            thr_dict = data["thresholds"]
            
            # Szenario-Überprüfung (Je nach Szenario sind unterschiedliche Signale zu betrachten)
            if data['fahrt_id'] in szenario:
                # Check, ob das zu analysierende Signal in den Daten vorhanden ist
                if signal_name in thr_dict:
                    for meth, meth_data in thr_dict[signal_name].items():
                        if meth == 'M1':
                            for thresh_name, thresh in meth_data.items():
                                if thresh_name in m1_thresh.keys():
                                    m1_thresh[thresh_name].append(thresh)
                        else:
                            for interv, metrics in meth_data.items():
                                for thresh_name, thresh in metrics.items():
                                    if thresh_name in m1_thresh.keys():
                                        if meth == 'M2':
                                            m2_thresh_all[interv][thresh_name].append(thresh)                                
                                        else:
                                            m3_thresh_all[interv][thresh_name].append(thresh)

    # Analyse aller Daten
    all_thresholds_dict = {'M1': m1_thresh, 'M2': m2_thresh_all, 'M3': m3_thresh_all}
    for meth, meth_data in all_thresholds_dict.items():
        analysed_thresholds[meth] = {}
        if meth == 'M1':
            for thresh_name, thresh in meth_data.items():
                analysed_thresholds[meth][thresh_name] = {}
                analysed_thresholds[meth][thresh_name]['Durchschnitt'] = np.mean(thresh)
                analysed_thresholds[meth][thresh_name]['Standardabweichung'] = np.std(thresh)
                analysed_thresholds[meth][thresh_name]['Anzahl der Werte'] = len(thresh)
        else:
            for interv, thresh_dict in meth_data.items():
                analysed_thresholds[meth][interv] = {}
                for thresh_name, thresh in thresh_dict.items():
                    analysed_thresholds[meth][interv][thresh_name] = {}
                    analysed_thresholds[meth][interv][thresh_name]['Durchschnitt'] = np.mean(thresh)
                    analysed_thresholds[meth][interv][thresh_name]['Standardabweichung'] = np.std(thresh)
                    analysed_thresholds[meth][interv][thresh_name]['Anzahl der Werte'] = len(thresh)
                               

    return analysed_thresholds

# Beispiel
thresholds_analysis = load_thresholds(signal, szenario=szenario)
save_thresholds(thresholds_analysis, signal)