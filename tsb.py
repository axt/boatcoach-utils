#! /usr/bin/env python3

import io
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

matplotlib.use('QT5Agg')

BOATCOACH_LOG_DIR = '../boatcoach-logs/'

STARTING_CTL = 0
STARTING_ATL = 0    
CTL_DECAY    = 42
ATL_DECAY    = 7
FIRST_DT     = '2019-01-01'
START_DT     = '2019-01-01'
END_DT       = '2021-07-31'
CUR_DT       = date.today().strftime("%Y-%m-%d") 

def load_logfile(fname):
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
    logfiles = []
    years = sorted([f for f in os.listdir(BOATCOACH_LOG_DIR) if os.path.isdir(BOATCOACH_LOG_DIR + f) and not f == '.git'])
    for year in years:
        logfiles += sorted([ year + '/' + f for f in os.listdir(BOATCOACH_LOG_DIR + year) if os.path.isfile(BOATCOACH_LOG_DIR + year + '/' + f) and f.endswith('csv')])
    return logfiles

def load_ftp():
    r = {}
    with open(BOATCOACH_LOG_DIR + '/FTP.txt', 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line[0:line.find('#')].split()
            r[cols[0]] = int(cols[1])
    return r

def duration_in_sec(d):
    s = 0
    for p in d.split(':'):
        s *= 60
        s += int(p)
    return s

def prepare_tsb_data():
    dates = pd.date_range(start=FIRST_DT, end=CUR_DT, freq='D')
    dfagg = pd.DataFrame(index=dates, columns=['TSS', 'FTP', 'ATL', 'CTL', 'TSB'])
    dfagg = dfagg.astype(dtype={'TSS':float, 'FTP': float, 'ATL':float, 'CTL':float, 'TSB':float})
    dfagg['TSS'].fillna(0, inplace=True)

    for k,v in load_ftp().items():
        dfagg.at[k,'FTP'] = v
    
    lastftp = np.nan
    for idx in dfagg.index:
        curftp = dfagg.ix[idx]['FTP']
        if np.isnan(curftp):
            dfagg.at[idx,'FTP'] = lastftp
        else:
            lastftp = curftp
    
    dfagg = dfagg[dfagg.index > START_DT]
    
    for f in get_logfiles():
        dt = f[15:25]
        if (dt < START_DT):
            continue
        df = load_logfile(f)
        df['workTime'] = df['workTime'].apply(duration_in_sec)
        
        workoutType = df['workoutType'][0]
        if not all(df['workoutType'] == workoutType):
            print("ERROR: workout type multivalued, recover manually: ", f)
            exit(1)
        if workoutType not in ['FixedTimeSplits', 'FixedDistanceSplits', 'VariableInterval']:
            print("ERROR: workout type '%s' unknown: %s" % (workoutType, f))
            exit(1)
        
        if workoutType == 'VariableInterval':
            df = df[df['intervalType'] != 'Rest']
            if len(df) == 0:
                print("ERROR: dataframe empty after filtering, unhandled case %s, %s" % (workoutType, f))
                exit(1)
            else:
                duration = df.groupby('intervalCount').max().sum()['workTime']
        else:
                duration = df['workTime'].max()
            
        ftp = dfagg.ix[dt]['FTP']

        mean_power = df['totalAvgPower'].iloc[-1]
        norm_power = np.sqrt(np.sqrt(np.mean(df['strokePower'].rolling(30).mean() ** 4)))
        intensity  = norm_power / ftp
        
        tss_old = int((duration * mean_power) / (ftp * 3600.0) * 100.0)
        tss     = int((duration * norm_power * intensity) / (ftp * 3600.0) * 100.0)
        
        print("%12s\t%d\t%d\t%d\t%.2f\t%d\t%d\t%d\t%.2f" % (dt, tss_old, tss, tss-tss_old, intensity, mean_power, norm_power, duration, tss*60/duration))
        
        dfagg.at[dt,'TSS'] = dfagg.ix[dt]['TSS'] + tss

    atl = STARTING_ATL
    ctl = STARTING_CTL
    for index, row in dfagg.iterrows():
        tss = row['TSS']
        atl = atl + (tss - atl)*(1/ATL_DECAY)
        ctl = ctl + (tss - ctl)*(1/CTL_DECAY)
        dfagg.at[index, 'ATL'] = atl
        dfagg.at[index, 'CTL'] = ctl
        dfagg['TSB'] = dfagg['CTL'] - dfagg['ATL']

    return dfagg

def plot_tsb_data(dfagg):
    last_dt = CUR_DT
    last = dfagg.ix[CUR_DT]
    df = dfagg.reset_index()

    plt.figure(figsize=(20,10))

    plt.plot( 'index', 'CTL', data=df, marker='', color='blue', linewidth=2, label="CTL (fitness)")
    plt.annotate( "%2.1f" % last['CTL'], (mdates.datestr2num(last_dt), last['CTL']), xytext=(30, 30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
    plt.axhline(last['CTL'], linestyle='--', color='blue')

    plt.plot( 'index', 'ATL', data=df, marker='', color='green', linewidth=2, label="ATL (fatigue)" )
    plt.annotate( "%2.1f" % last['ATL'], (mdates.datestr2num(last_dt), last['ATL']), xytext=(30, 30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
    plt.axhline(last['ATL'], linestyle='--', color='green')


    plt.plot( 'index', 'TSB', data=df, marker='', color='red', linewidth=2)
    plt.annotate( "%2.1f" % last['TSB'], (mdates.datestr2num(last_dt), last['TSB']), xytext=(30, -30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
    plt.axhline(last['TSB'], linestyle='--', color='red')

    plt.scatter( 'index', 'TSS', data=df, marker='o', color='gray')
    plt.axhline(last['TSS'], linestyle='--', color='gray')

    plt.grid()
    plt.legend()
    plt.ylim(-50,100)
    plt.xlim(START_DT, END_DT)
    return plt


def plot_tss_agg(dfagg, period, width=5):
    df = dfagg.resample(period).sum()
    df = df.reset_index()
    plt.figure(figsize=(30,15))
    plt.bar( df['index'], df['TSS'], width)
    plt.grid(axis='y')
    plt.legend()
    plt.ylabel('Aggregated TSS')
    plt.xlim(START_DT, END_DT)
    return plt


def main():
    dfagg = prepare_tsb_data()
    print(dfagg)
    plt = plot_tsb_data(dfagg)
    plt.savefig('tsb.png', bbox_inches='tight')
    plt = plot_tss_agg(dfagg, 'M', 20)
    plt.savefig('tss_monthly.png', bbox_inches='tight')
    plt = plot_tss_agg(dfagg, 'W')
    plt.savefig('tss_weekly.png', bbox_inches='tight')
    

if __name__== "__main__":
    main()
