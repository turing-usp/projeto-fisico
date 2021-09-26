#!/usr/bin/env python

### Modules #############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import json
import hashlib

from pathlib import Path
from scipy.signal import filtfilt


### Params set ##########################################################
PROGRESS_FILE = 'progress.csv'

DEFAULT_COLUMNS = [
    'custom_metrics/agent_checkpoints_mean',
    ','.join([
        'hist_stats/agent_checkpoints',
        'custom_metrics/agent_checkpoints',
    ]),
    ','.join([
        'hist_stats/agent_reward',
        'custom_metrics/agent_reward',
    ]),
    ','.join([
        'env/unity_config/AgentVelocityBonus_CoeffPerSecond',
        'env/unity_config/HazardCountPerChunk',
        'env/unity_config/ChunkDifficulty',
        'env/unity_config/HazardMaxSpeed',
        'env/unity_config/HazardMinSpeed',
    ]),
    'agent_steps_this_phase',
    'info/learner/fisico/entropy',
    ','.join([
        'info/learner/fisico/vf_loss',
        'info/learner/fisico/policy_loss',
        'info/learner/fisico/total_loss',
    ]),
]

HIST_SIZE_PER_STEP = 64
HIST_MARKER_SIZE = 8

CM = plt.cm.get_cmap('tab10')
plt.rc('font', size=9)
plt.rc('legend', fontsize=7)


### Functions ###########################################################

def main():

    ### Parser
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog=prog, width=100))
    parser.add_argument("experiment", type=Path, metavar="EXPERIMENT",
                        help='Experiment directory')
    parser.add_argument("--no-phase", action='store_true',
                        help='Do not plot the phase')
    parser.add_argument("--no-noise", action='store_true',
                        help='Do not not add noise in hist_stats plots')
    parser.add_argument("--window", "-w", type=int, default=10,
                        help="The window size for the rolling averages (default: 10)")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Output DPI (default: 120)")
    parser.add_argument("--no-default-columns", action='store_true',
                        help='Do not plot the default columns')
    parser.add_argument("--column", "-c", type=str, metavar='COL', nargs='*', dest='columns', default=[],
                        help='Additional columns to plot. '
                        'If COL does not exist, will try to plot COL_min, COL_max and COL_mean. '
                        'If COL is a comma-separated list of columns, plot them all together. '
                        '(repeatable)')
    args = parser.parse_args()

    if not args.no_default_columns:
        args.columns = DEFAULT_COLUMNS + [c for c in args.columns if c not in DEFAULT_COLUMNS]

    ### Import data
    df = pd.read_csv(args.experiment / PROGRESS_FILE)
    rows = len(args.columns)

    ### Create figure for plot
    fig = plt.figure(figsize=(6, 1+3*rows))

    ### For each subplot
    for i, cols in enumerate(args.columns):
        plt.subplot(rows, 1, i+1)
        if i == 0:
            plt.suptitle(args.experiment.name, fontsize=12)
            train_iters = df["training_iteration"].iloc[-1]
            total_time = df["time_total_s"].iloc[-1]
            plt.title(f'\n\n{train_iters} iterations    {total_time:.1f} seconds    {total_time/train_iters:.1f} $\pm$ {np.std(df["time_this_iter_s"]):.1f} seconds/iteration\n'
                      f'rolling average exponential window with size {args.window}\n')
        try:
            plot_cols(args, cols, df)
        except KeyError:
            print('Invalid column(s):', cols)
            ### Remove os subplots vazios
            fig.delaxes(plt.gca())

    ### Save plot
    plt.tight_layout()
    plt.savefig((args.experiment / PROGRESS_FILE).with_suffix('.png'), dpi=args.dpi)


def plot_cols(args, cols, df):
    i = 0
    for col in cols.split(','):
        col = col.strip()
        if col in df.columns:
            plot_series(df[col], c=CM(i), window=args.window, add_noise=not args.no_noise)
            i += 1
        else:
            for typ in 'max', 'mean', 'min':
                plot_series(df[col + '_' + typ], c=CM(i), window=args.window, add_noise=not args.no_noise)
                i += 1

    plt.xlabel('Training iteration')
    plt.grid()
    
    if args.no_phase:
        plt.legend()
    else:
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        
        twin_ax = plt.twinx()
        df['env/phase'].plot(c=(.5,0,1), ls='--', dashes=(4,3.5))
        plt.ylabel('env/phase')
        plt.yticks(range(-1, df['env/phase'].iloc[-1] + 1))
        
        twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
        
        leg = ax.legend(handles + twin_handles, labels + twin_labels)


def plot_series(s, *, c, window, add_noise):
    if s.name.startswith('hist_stats/'):
        np.random.seed(int(hashlib.md5(s.name.encode()).hexdigest()[:8], base=16))
        x = []
        y = []
        for i, raw_vals in enumerate(s):
            vals = np.random.choice(json.loads(raw_vals), size=HIST_SIZE_PER_STEP, replace=True)
            x.extend(np.random.uniform(i-.5, i+.5, size=len(vals)))
            y.extend(vals)
        if add_noise:
            y += .015 * (max(y) - min(y)) * np.random.uniform(-1, 1, size=len(y))
        
        scatter = plt.scatter(x, y, color=lighten(c, .1), alpha=min(.5, 1000/len(x)/HIST_MARKER_SIZE), s=HIST_MARKER_SIZE)
        plt.scatter([], [], color=c, s=4, label=s.name)
    elif s.name.startswith('custom_metrics/') or s.name.startswith('info/learner/'):
        plt.plot(s, c=c, alpha=.5, lw=1)
        plt.plot(filtfilt(np.ones(window)/window,1,s.values), label=s.name, c=c)
    else:
        plt.plot(s, label=s.name, c=c)


def lighten(color, amount):
    white = (1, 1, 1, color[3])  # preserve the alpha value
    return tuple(w*amount + c*(1-amount) for w, c in zip(white, color))


if __name__ == '__main__':
    main()
