#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

PROGRESS_FILE = 'progress.csv'
DEFAULT_COLUMNS = [
    'custom_metrics/agent_checkpoints_mean',
    'custom_metrics/agent_checkpoints',
    'custom_metrics/agent_reward',
    'time_this_iter_s',
    'timers/sample_time_ms,timers/learn_time_ms,timers/update_time_ms',
]

CM = plt.cm.get_cmap('tab10')
plt.rc('font', size=9)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog=prog, width=100))
    parser.add_argument("experiment", type=Path,
                        help='Experiment directory')
    parser.add_argument("--no-phase", action='store_true',
                        help='Do not plot the phase')
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

    df = pd.read_csv(args.experiment / PROGRESS_FILE)
    rows = len(args.columns)
    plt.figure(figsize=(6, 3*rows))
    for i, col in enumerate(args.columns):
        plt.subplot(rows, 1, i+1)
        try:
            plot_col(args, col, df)
        except KeyError:
            print('Invalid column:', col)
            return

    plt.tight_layout()
    plt.savefig((args.experiment / PROGRESS_FILE).with_suffix('.png'), dpi=args.dpi)


def plot_col(args, col, df):
    if ',' in col:
        for j, col in enumerate(col.split(',')):
            plot_series(df[col.strip()], c=CM(j), window=args.window)
    elif col in df.columns:
        plot_series(df[col], c=CM(0), window=args.window)
        plt.ylabel(col)
    else:
        for j, typ in enumerate(['max', 'mean', 'min']):
            plot_series(df[col + '_' + typ], label=typ, c=CM(j), window=args.window)
        plt.ylabel(col)

    plt.legend()
    plt.xlabel('training iteration')
    plt.grid()

    if not args.no_phase:
        plot_phase(df)


def plot_series(s, *, label=None, c, window):
    if label is None:
        label = s.name
    plt.plot(s, c=c, alpha=.5, lw=1)
    plt.plot(s.rolling(window, min_periods=1).mean(), label=label, c=c)


def plot_phase(df):
    plt.twinx()
    df['env/phase'].plot(c=CM(3), lw=1)
    plt.ylabel('env/phase')
    plt.yticks(range(-1, df['env/phase'].iloc[-1] + 1))


if __name__ == '__main__':
    main()
