import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse

def load_and_process(exp):
    # load data
    files = glob.glob('{}/*/training_log.csv'.format(exp))
    all_files = [pd.read_csv(file) for file in files] # ['step'] ['success rate'] ['episodic reward']
    processed_files = []

    # smooth data
    for datum in all_files:
        datum['reward'] = datum['episodic reward'].ewm(span=3000).mean()
        datum['success'] = datum['success rate'].ewm(span=3000).mean()
        datum = datum[datum['step']%50==0]
        processed_files.append(datum)    

    # put data together
    data = pd.concat(processed_files)

    return data

# load and process data
parser = argparse.ArgumentParser()
parser.add_argument("algo", help="algorithm to plot")
parser.add_argument("scenario", help="scenario to plot")
parser.add_argument('metric', help='metric to plot')
args = parser.parse_args()

results = load_and_process(f'train_results/{args.scenario}/{args.algo}')

# plot
plt.figure(figsize=(15, 10))
seaborn.set(style="whitegrid", font_scale=2, rc={"lines.linewidth": 3})
seaborn.lineplot(data=results, x='step', y=args.metric, err_style='band')
axes = plt.gca()
axes.set_xlim([0, 1e5])
axes.set_xlabel('Step')

if args.metric == 'success':
    axes.set_ylim([0, 1])
    axes.set_ylabel('Average success rate')
elif args.metric == 'reward' and args.scenario == 'left_turn':
    axes.set_ylim([-1, 2])
    axes.set_ylabel('Average episode reward')
elif args.metric == 'reward' and args.scenario == 'roundabout':
    axes.set_ylim([-1, 3])
    axes.set_ylabel('Average episode reward')
else:
    raise Exception('Undefined metric!')

axes.set_title(f'{args.scenario} ({args.algo})')
plt.tight_layout()
plt.show()

