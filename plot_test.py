import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

def smooth_data(traj, steps=10):
    return np.convolve(traj, np.ones(steps), 'same') / steps

def load_and_process(file):
    # load data
    file = pd.read_csv(file)
    ids = file[file['success']==True]['No.']
    id = np.random.choice(ids) - 1
    test_data = {}
    speed = []
    jerk = []
    acceleration = []
    curvature = []

    # extract data
    for v in file.loc[id, 'speed'].strip('[').strip(']').strip().split(','):
        speed.append(float(v))

    for a_x, a_y in zip(file.loc[id, 'linear_acceleration_x'].strip('[').strip(']').replace(' ', '').split(','),
                        file.loc[id, 'linear_acceleration_y'].strip('[').strip(']').replace(' ', '').split(',')):
        acceleration.append(np.sign(float(a_x)) * np.linalg.norm((a_x, a_y)))

    for j_x, j_y in zip(file.loc[id, 'linear_jerk_x'].strip('[').strip(']').replace(' ', '').split(','),
                        file.loc[id, 'linear_jerk_y'].strip('[').strip(']').replace(' ', '').split(',')):
        jerk.append(np.sign(float(j_x)) * np.linalg.norm((j_x, j_y)))

    for i, w in enumerate(file.loc[id, 'angular_velocity_z'].strip('[').strip(']').replace(' ', '').split(',')):
        curvature.append(float(w) / speed[i])

    test_data['speed'] = np.array(speed)
    test_data['acceleration'] = smooth_data(np.array(acceleration), steps=5)
    test_data['curvature'] = smooth_data(np.array(curvature), steps=5)
    test_data['jerk'] = smooth_data(np.array(jerk), steps=5)
    test_data['time'] = np.linspace(0, 0.1*len(speed), len(speed))

    return test_data

# load and process data
parser = argparse.ArgumentParser()
parser.add_argument("test_log", help="path to the test log")
args = parser.parse_args()
results = load_and_process(args.test_log)
metrics = ['speed', 'acceleration', 'curvature']
units = ['m/s', 'm/s^2', '1/m']

# plot
fig, axes = plt.subplots(3, 1)

for i in range(3):
    axes[i].plot(results['time'], results[metrics[i]], linewidth=2)
    axes[i].set_xlabel('Time (s)')
    axes[i].set_xlim([0, results['time'][-1]])
    axes[i].set_ylabel('{metrics[i]} ({units[i]})')
    axes[i].grid()
    if i == 0:
        axes[i].set_ylim(bottom=0)
    else:
        axes[i].axhline(y=0, color='k', linestyle='--', linewidth=1.5)

fig.set_size_inches(8, 10, forward=True)
fig.tight_layout()
plt.show()
