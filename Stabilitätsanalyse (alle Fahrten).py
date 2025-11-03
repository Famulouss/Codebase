import json
import glob
import numpy as np

def load_thresholds(signal_name, method=None, interval=None, metric="Youden", fahrt_id='Szenario 1'):
    files = glob.glob("grenzwerte/*.json")
    values = []

    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            thr_dict = data["thresholds"]
            
            if data['fahrt_id'] in fahrt_id:
                if signal_name in thr_dict:
                    for meth, meth_data in thr_dict[signal_name].items():
                        if method is None or meth == method:
                            for interv, metrics in meth_data.items():
                                if interval is None or interv == interval:
                                    if metric in metrics:
                                        values.append(metrics[metric])

    return np.mean(values), np.std(values), len(values)

# Beispiel
mean, std, n = load_thresholds("acc_x_filt", method="M1", metric="Youden", fahrt_id='Szenario 2')
print(f"acc_x_filt (M1, Youden): mean={mean:.3f}, std={std:.3f}, n={n}")