import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Pfad zur CSV-Datei
csv_file = r'C:\Users\kompa\Documents\Uni\TUM\Bachelorarbeit\Daten\imu_data.csv'  # passe den Pfad ggf. an

# Spaltennamen manuell definierenpip
column_names = [
    'sec', 'nanosec',
    'frame_id',
    'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
    'orientation_cov_00', 'orientation_cov_01', 'orientation_cov_02',
    'orientation_cov_10', 'orientation_cov_11', 'orientation_cov_12',
    'orientation_cov_20', 'orientation_cov_21', 'orientation_cov_22',
    'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
    'angular_velocity_cov_00', 'angular_velocity_cov_01', 'angular_velocity_cov_02',
    'angular_velocity_cov_10', 'angular_velocity_cov_11', 'angular_velocity_cov_12',
    'angular_velocity_cov_20', 'angular_velocity_cov_21', 'angular_velocity_cov_22',
    'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
    'linear_acceleration_cov_00', 'linear_acceleration_cov_01', 'linear_acceleration_cov_02',
    'linear_acceleration_cov_10', 'linear_acceleration_cov_11', 'linear_acceleration_cov_12',
    'linear_acceleration_cov_20', 'linear_acceleration_cov_21', 'linear_acceleration_cov_22',
]

# Einlesen ohne Header und mit manuellen Spaltennamen
df = pd.read_csv(csv_file, header=None, names=column_names)

# Zeit in Sekunden berechnen
df['time_s'] = df['sec'] + df['nanosec'] * 1e-9
df['time_s'] -= df['time_s'].iloc[0]  # Zeit bei 0 starten

# Lineare Beschleunigung plotten
plt.figure(figsize=(10, 5))
plt.plot(df['time_s'], df['linear_acceleration.x'], label='Acc X')
plt.plot(df['time_s'], df['linear_acceleration.y'], label='Acc Y')
plt.plot(df['time_s'], df['linear_acceleration.z'], label='Acc Z')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s²]')
plt.title('IMU Linear Acceleration')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Winkelgeschwindigkeit plotten
plt.figure(figsize=(10, 5))
plt.plot(df['time_s'], df['angular_velocity.x'], label='Gyro X')
plt.plot(df['time_s'], df['angular_velocity.y'], label='Gyro Y')
plt.plot(df['time_s'], df['angular_velocity.z'], label='Gyro Z')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.title('IMU Angular Velocity')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Jerk berechnen
# J = da/dt ≈ (a[i+1] - a[i]) / (t[i+1] - t[i])

# Leere Spalten für Jerk anlegen
for axis in ['x', 'y', 'z']:
    a = df[f'linear_acceleration.{axis}'].values
    t = df['time_s'].values
    jerk = np.gradient(a, t)
    df[f'jerk.{axis}'] = jerk

# Jerk plotten
plt.figure(figsize=(10, 6))
plt.plot(df['time_s'], df['jerk.x'], label='Jerk X')
plt.plot(df['time_s'], df['jerk.y'], label='Jerk Y')
plt.plot(df['time_s'], df['jerk.z'], label='Jerk Z')
plt.xlabel('Time [s]')
plt.ylabel('Jerk [m/s³]')
plt.title('Jerk (Rate of Change of Acceleration)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()