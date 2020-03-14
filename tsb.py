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
START_DT     = '2019-01-01'
END_DT       = '2020-12-31'
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

def get_ftp(dt):
    return 145

def prepare_tsb_data():
    dates = pd.date_range(start=START_DT, end=CUR_DT, freq='D')
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
    
    for f in get_logfiles():
        dt = f[15:25]
        df = load_logfile(f)
        mean_power = df['totalAvgPower'].mean()
        duration   = duration_in_sec(df['workTime'].max())
        tss = int(100 * (mean_power*duration / 3600) / dfagg.ix[dt]['FTP'])
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

def main():
    dfagg = prepare_tsb_data()
    plt = plot_tsb_data(dfagg)
    plt.savefig('tsb.png', bbox_inches='tight')

if __name__== "__main__":
    main()
