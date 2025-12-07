#!/usr/bin/env python3
"""
FTP Calculator for BoatCoach logs.

Analyzes workout data to estimate FTP (Functional Threshold Power) using:
1. Best 20-minute average power × 0.95
2. Best 8-minute average power × 0.90

Usage:
    python calc_ftp.py                    # Analyze all workouts
    python calc_ftp.py 2024/workout.csv   # Analyze specific workout (e.g., after FTP test)
"""

import io
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

BOATCOACH_LOG_DIR = '../boatcoach-logs/'

# Test durations in seconds
TEST_20MIN = 20 * 60  # 1200 seconds
TEST_8MIN = 8 * 60    # 480 seconds


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

    # Get workout date from filename
    date = filepath[5:15] if '/' in filepath else filepath[:10]

    # Get workout duration
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

    # Find best 20-minute power
    best_20, start_20, end_20 = find_best_power_window(df, TEST_20MIN)
    if best_20 is not None:
        result['best_20min_power'] = best_20
        result['ftp_20min'] = best_20 * 0.95

    # Find best 8-minute power
    best_8, start_8, end_8 = find_best_power_window(df, TEST_8MIN)
    if best_8 is not None:
        result['best_8min_power'] = best_8
        result['ftp_8min'] = best_8 * 0.90

    return result


def main():
    if len(sys.argv) > 1:
        # Analyze specific workout
        files = [sys.argv[1]]
    else:
        # Analyze all workouts
        files = get_logfiles()

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

    # Sort by estimated FTP
    results_20min.sort(key=lambda x: x['ftp_20min'], reverse=True)
    results_8min.sort(key=lambda x: x['ftp_8min'], reverse=True)

    # Display results
    print("\n" + "=" * 80)
    print("FTP ESTIMATION FROM 20-MINUTE EFFORTS (×0.95)")
    print("=" * 80)

    if results_20min:
        print(f"\n{'Date':<12} {'Duration':>10} {'Best 20min':>12} {'Est. FTP':>10}")
        print("-" * 50)
        for r in results_20min[:10]:  # Show top 10
            print(f"{r['date']:<12} {r['duration_min']:>8.1f}m {r['best_20min_power']:>10.1f}W {r['ftp_20min']:>9.1f}W")

        best = results_20min[0]
        print(f"\n>>> Best estimate from 20-min test: {best['ftp_20min']:.0f}W")
        print(f"    (from {best['date']}, best 20-min avg: {best['best_20min_power']:.1f}W)")
    else:
        print("\nNo workouts longer than 20 minutes found.")

    print("\n" + "=" * 80)
    print("FTP ESTIMATION FROM 8-MINUTE EFFORTS (×0.90)")
    print("=" * 80)

    if results_8min:
        print(f"\n{'Date':<12} {'Duration':>10} {'Best 8min':>12} {'Est. FTP':>10}")
        print("-" * 50)
        for r in results_8min[:10]:  # Show top 10
            print(f"{r['date']:<12} {r['duration_min']:>8.1f}m {r['best_8min_power']:>10.1f}W {r['ftp_8min']:>9.1f}W")

        best = results_8min[0]
        print(f"\n>>> Best estimate from 8-min test: {best['ftp_8min']:.0f}W")
        print(f"    (from {best['date']}, best 8-min avg: {best['best_8min_power']:.1f}W)")
    else:
        print("\nNo workouts longer than 8 minutes found.")

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if results_20min:
        recommended_ftp = int(results_20min[0]['ftp_20min'])
        print(f"\nAdd this line to your FTP.txt file:")
        print(f"\n    {datetime.now().strftime('%Y-%m-%d')} {recommended_ftp}")
        print(f"\nNote: For more accurate results, do a dedicated 20-minute all-out test")
        print("and then run: python calc_ftp.py <path-to-test-workout.csv>")
    elif results_8min:
        recommended_ftp = int(results_8min[0]['ftp_8min'])
        print(f"\nBased on 8-min data, estimated FTP: {recommended_ftp}W")
        print("Consider doing a 20-minute test for better accuracy.")
    else:
        print("\nNot enough data to estimate FTP. Do a workout of at least 8 minutes.")


if __name__ == "__main__":
    main()
