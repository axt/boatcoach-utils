#!/usr/bin/env python3
"""
FTP Calculator for BoatCoach logs.

Analyzes workout data to estimate FTP (Functional Threshold Power) using:
1. Best 20-minute average power × 0.95
2. Best 8-minute average power × 0.90

Usage:
    python calc_ftp.py                    # Analyze all workouts and plot
    python calc_ftp.py 2024/workout.csv   # Analyze specific workout (e.g., after FTP test)
"""

import io
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

matplotlib.use('Agg')  # Non-interactive backend for saving plots

BOATCOACH_LOG_DIR = '../boatcoach-logs/'

# Test durations in seconds
TEST_20MIN = 20 * 60  # 1200 seconds
TEST_8MIN = 8 * 60    # 480 seconds

# Sanity check: realistic power range for rowing (Watts)
MIN_REALISTIC_POWER = 50
MAX_REALISTIC_POWER = 500

# Rolling max window for FTP trend (days)
FTP_ROLLING_WINDOW = 42  # 6 weeks


def load_logfile(fname):
    """Load a BoatCoach CSV log file."""
    r = ""
    first = True
    with open(BOATCOACH_LOG_DIR + '/' + fname, 'rt') as f:
        for line in f:
            if first:
                first = False
            else:
                pos = line.find(',,')
                r += line[0:pos] + '\n'
    return pd.read_csv(io.StringIO(r))


def get_logfiles():
    """Get all log files from the boatcoach-logs directory."""
    logfiles = []
    years = sorted([f for f in os.listdir(BOATCOACH_LOG_DIR)
                    if os.path.isdir(BOATCOACH_LOG_DIR + f) and not f == '.git'])
    for year in years:
        logfiles += sorted([year + '/' + f
                           for f in os.listdir(BOATCOACH_LOG_DIR + year)
                           if os.path.isfile(BOATCOACH_LOG_DIR + year + '/' + f) and f.endswith('csv')])
    return logfiles


def duration_in_sec(d):
    """Convert duration string (H:MM:SS or MM:SS) to seconds."""
    s = 0
    for p in d.split(':'):
        s *= 60
        s += int(p)
    return s


def find_best_power_window(df, window_seconds):
    """
    Find the best average power over a rolling window.

    Args:
        df: DataFrame with strokePower column (1 Hz sampling)
        window_seconds: Size of the window in seconds

    Returns:
        (best_avg_power, start_idx, end_idx) or (None, None, None) if workout too short
    """
    if len(df) < window_seconds:
        return None, None, None

    # Calculate rolling average
    rolling_avg = df['strokePower'].rolling(window=window_seconds).mean()

    # Find the maximum
    best_idx = rolling_avg.idxmax()
    if pd.isna(best_idx):
        return None, None, None

    best_avg = rolling_avg.loc[best_idx]
    start_idx = best_idx - window_seconds + 1

    return best_avg, start_idx, best_idx


def analyze_workout(filepath):
    """
    Analyze a single workout for FTP estimation.

    Returns:
        dict with workout analysis results
    """
    df = load_logfile(filepath)

    # Get workout date from filename (format: YYYY/boatcoach_YYYY-MM-DD_...)
    # Date is at position 15:25 in the filepath
    date = filepath[15:25] if '/' in filepath else filepath[:10]

    # Get workout duration from the workTime column (cumulative)
    df['workTime_sec'] = df['workTime'].apply(duration_in_sec)
    total_duration = df['workTime_sec'].max()

    result = {
        'file': filepath,
        'date': date,
        'duration_min': total_duration / 60,
        'avg_power': df['strokePower'].mean(),
        'max_power': df['strokePower'].max(),
        'ftp_20min': None,
        'ftp_8min': None,
        'best_20min_power': None,
        'best_8min_power': None,
    }

    # Find best 20-minute power (use actual duration, not row count)
    if total_duration >= TEST_20MIN:
        best_20, start_20, end_20 = find_best_power_window(df, TEST_20MIN)
        if best_20 is not None and MIN_REALISTIC_POWER <= best_20 <= MAX_REALISTIC_POWER:
            result['best_20min_power'] = best_20
            result['ftp_20min'] = best_20 * 0.95

    # Find best 8-minute power
    if total_duration >= TEST_8MIN:
        best_8, start_8, end_8 = find_best_power_window(df, TEST_8MIN)
        if best_8 is not None and MIN_REALISTIC_POWER <= best_8 <= MAX_REALISTIC_POWER:
            result['best_8min_power'] = best_8
            result['ftp_8min'] = best_8 * 0.90

    return result


def plot_ftp_progression(results, output_file='ftp_progression.png'):
    """
    Plot FTP estimates over time with rolling max trend.

    Args:
        results: List of workout analysis results
        output_file: Output filename for the plot
    """
    # Filter results with valid FTP estimates and sort by date
    results_20min = [r for r in results if r['ftp_20min'] is not None]
    results_8min = [r for r in results if r['ftp_8min'] is not None]

    if not results_20min and not results_8min:
        print("No valid FTP estimates to plot.")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # --- Plot 1: 20-minute FTP estimates ---
    ax1 = axes[0]
    if results_20min:
        # Sort by date
        results_20min.sort(key=lambda x: x['date'])

        dates_20 = [pd.to_datetime(r['date']) for r in results_20min]
        ftp_20 = [r['ftp_20min'] for r in results_20min]

        # Create DataFrame for rolling calculations
        df_20 = pd.DataFrame({'date': dates_20, 'ftp': ftp_20})
        df_20 = df_20.set_index('date').sort_index()

        # Calculate rolling max (represents current FTP level)
        df_20['rolling_max'] = df_20['ftp'].rolling(
            window=f'{FTP_ROLLING_WINDOW}D', min_periods=1
        ).max()

        # Plot individual estimates as scatter
        ax1.scatter(df_20.index, df_20['ftp'],
                   color='blue', alpha=0.6, s=50, label='Workout FTP estimate', zorder=3)

        # Plot rolling max as line
        ax1.plot(df_20.index, df_20['rolling_max'],
                color='darkblue', linewidth=2, label=f'{FTP_ROLLING_WINDOW}-day rolling max', zorder=2)

        # Fill area under rolling max
        ax1.fill_between(df_20.index, 0, df_20['rolling_max'],
                        alpha=0.1, color='blue')

        # Annotate current FTP
        if len(df_20) > 0:
            current_ftp = df_20['rolling_max'].iloc[-1]
            ax1.axhline(current_ftp, linestyle='--', color='darkblue', alpha=0.5)
            ax1.annotate(f'Current: {current_ftp:.0f}W',
                        xy=(df_20.index[-1], current_ftp),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold')

        ax1.set_ylabel('FTP (Watts)', fontsize=12)
        ax1.set_title('FTP Progression - 20-Minute Test (×0.95)', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        ymin = min(ftp_20) * 0.9
        ymax = max(ftp_20) * 1.1
        ax1.set_ylim(ymin, ymax)
    else:
        ax1.text(0.5, 0.5, 'No workouts >= 20 minutes',
                transform=ax1.transAxes, ha='center', va='center', fontsize=14)
        ax1.set_title('FTP Progression - 20-Minute Test (×0.95)', fontsize=14, fontweight='bold')

    # --- Plot 2: 8-minute FTP estimates ---
    ax2 = axes[1]
    if results_8min:
        # Sort by date
        results_8min.sort(key=lambda x: x['date'])

        dates_8 = [pd.to_datetime(r['date']) for r in results_8min]
        ftp_8 = [r['ftp_8min'] for r in results_8min]

        # Create DataFrame for rolling calculations
        df_8 = pd.DataFrame({'date': dates_8, 'ftp': ftp_8})
        df_8 = df_8.set_index('date').sort_index()

        # Calculate rolling max
        df_8['rolling_max'] = df_8['ftp'].rolling(
            window=f'{FTP_ROLLING_WINDOW}D', min_periods=1
        ).max()

        # Plot individual estimates as scatter
        ax2.scatter(df_8.index, df_8['ftp'],
                   color='green', alpha=0.6, s=50, label='Workout FTP estimate', zorder=3)

        # Plot rolling max as line
        ax2.plot(df_8.index, df_8['rolling_max'],
                color='darkgreen', linewidth=2, label=f'{FTP_ROLLING_WINDOW}-day rolling max', zorder=2)

        # Fill area under rolling max
        ax2.fill_between(df_8.index, 0, df_8['rolling_max'],
                        alpha=0.1, color='green')

        # Annotate current FTP
        if len(df_8) > 0:
            current_ftp = df_8['rolling_max'].iloc[-1]
            ax2.axhline(current_ftp, linestyle='--', color='darkgreen', alpha=0.5)
            ax2.annotate(f'Current: {current_ftp:.0f}W',
                        xy=(df_8.index[-1], current_ftp),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold')

        ax2.set_ylabel('FTP (Watts)', fontsize=12)
        ax2.set_title('FTP Progression - 8-Minute Test (×0.90)', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        ymin = min(ftp_8) * 0.9
        ymax = max(ftp_8) * 1.1
        ax2.set_ylim(ymin, ymax)
    else:
        ax2.text(0.5, 0.5, 'No workouts >= 8 minutes',
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        ax2.set_title('FTP Progression - 8-Minute Test (×0.90)', fontsize=14, fontweight='bold')

    # Format x-axis
    ax2.set_xlabel('Date', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    return fig


def print_summary_table(results, title, ftp_key, power_key):
    """Print a summary table of FTP estimates."""
    filtered = [r for r in results if r[ftp_key] is not None]
    if not filtered:
        print(f"\nNo workouts found for {title}")
        return

    # Sort by date (chronological)
    filtered.sort(key=lambda x: x['date'])

    print(f"\n{'Date':<12} {'Duration':>10} {'Best Power':>12} {'Est. FTP':>10}")
    print("-" * 50)
    for r in filtered:
        print(f"{r['date']:<12} {r['duration_min']:>8.1f}m {r[power_key]:>10.1f}W {r[ftp_key]:>9.1f}W")

    # Statistics
    ftp_values = [r[ftp_key] for r in filtered]
    print("-" * 50)
    print(f"{'Min:':<12} {'':<10} {'':<12} {min(ftp_values):>9.1f}W")
    print(f"{'Max:':<12} {'':<10} {'':<12} {max(ftp_values):>9.1f}W")
    print(f"{'Average:':<12} {'':<10} {'':<12} {np.mean(ftp_values):>9.1f}W")
    print(f"{'Latest:':<12} {'':<10} {'':<12} {ftp_values[-1]:>9.1f}W")


def main():
    if len(sys.argv) > 1:
        # Analyze specific workout
        files = [sys.argv[1]]
        single_file = True
    else:
        # Analyze all workouts
        files = get_logfiles()
        single_file = False

    results = []

    print("Analyzing workouts for FTP estimation...\n")
    print("-" * 80)

    for f in files:
        try:
            result = analyze_workout(f)
            results.append(result)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Filter results with valid FTP estimates
    results_20min = [r for r in results if r['ftp_20min'] is not None]
    results_8min = [r for r in results if r['ftp_8min'] is not None]

    # Display 20-minute results
    print("\n" + "=" * 80)
    print("FTP ESTIMATION FROM 20-MINUTE EFFORTS (×0.95)")
    print("=" * 80)
    print_summary_table(results, "20-minute test", 'ftp_20min', 'best_20min_power')

    # Display 8-minute results
    print("\n" + "=" * 80)
    print("FTP ESTIMATION FROM 8-MINUTE EFFORTS (×0.90)")
    print("=" * 80)
    print_summary_table(results, "8-minute test", 'ftp_8min', 'best_8min_power')

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if results_20min:
        # Use latest 20-min estimate as recommendation
        results_20min.sort(key=lambda x: x['date'])
        latest = results_20min[-1]

        # Also find the best recent (last 42 days worth of data)
        best_recent = max(results_20min[-10:], key=lambda x: x['ftp_20min']) if len(results_20min) > 0 else latest

        print(f"\nLatest FTP estimate: {latest['ftp_20min']:.0f}W (from {latest['date']})")
        print(f"Best recent estimate: {best_recent['ftp_20min']:.0f}W (from {best_recent['date']})")

        recommended_ftp = int(best_recent['ftp_20min'])
        print(f"\nAdd this line to your FTP.txt file:")
        print(f"\n    {datetime.now().strftime('%Y-%m-%d')} {recommended_ftp}")

    elif results_8min:
        results_8min.sort(key=lambda x: x['date'])
        latest = results_8min[-1]
        recommended_ftp = int(latest['ftp_8min'])
        print(f"\nBased on 8-min data, estimated FTP: {recommended_ftp}W")
        print("Consider doing a 20-minute test for better accuracy.")
    else:
        print("\nNot enough data to estimate FTP. Do a workout of at least 8 minutes.")

    # Generate plot if analyzing all workouts
    if not single_file and (results_20min or results_8min):
        print("\n" + "=" * 80)
        print("GENERATING PLOT")
        print("=" * 80)
        plot_ftp_progression(results)


if __name__ == "__main__":
    main()
