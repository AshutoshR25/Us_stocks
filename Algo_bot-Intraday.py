import glob
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
import datetime

# Python for finance #7
def computeRSI(df, time_window):
    diff = df.diff(1).dropna()  # diff in one field(one day)

    up_chg = 0 * diff
    down_chg = 0 * diff

    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[diff < 0]

    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def computeBollingerBands(df, n, m):
    df['TP'] = (df["high"] + df["low"] + df["close"]) / 3

    df['std'] = df['TP'].rolling(n).std(ddof=0)
    df['MA-TP']  = df['TP'].rolling(n).mean()
    df['BOLU'] = df['MA-TP'] + m * df['std']
    df['BOLD'] = df['MA-TP'] - m * df['std']

    return df

def computeProfit(i, Profit, ax):
    if i == 0:
        with open('temp.txt', 'w') as f:
            f.write('0')
    else:
        with open('temp.txt', 'r') as f:
            Current_Profit = float(f.readlines()[0])
            Current_Profit = Current_Profit + Profit
        with open('temp.txt', 'w') as f:
            f.write(str(Current_Profit))

        ax.text(0.83, 1.15, 'Total Profit: $' + str(round(Current_Profit, 3)) + ' per lot',
                bbox=dict(facecolor='#FF7A01', alpha=0.5),
                transform=ax.transAxes, color='white', fontsize=16, fontweight='bold',
                horizontalalignment='left', verticalalignment='center')

def backtest_day(i, data_full):
    time_stamp = date_list[i]
    #Changes made
    year = int(time_stamp[0:4])
    month = int(time_stamp[5:7])
    day = int(time_stamp[8:10])

    data_day = data_full[   (data_full['year'] == year) &
                            (data_full['month'] == month) &
                            (data_full['day'] == day)]

    data_day.reset_index(inplace=True)
    current_date = str(year) + '-' + str(month) + '-' + str(day)
    return data_day, current_date

def MA_Strategy(data):
    data['MA5'] = data["close"].rolling(5).mean()
    data['MA10'] = data["close"].rolling(10).mean()
    data['MA20'] = data["close"].rolling(20).mean()

    Buy=[] #show buy in the graph
    Sell=[] #show sell in the graph
    Record=[] #record buy and sell
    position = False # no short selling

    for i in range(len(data['close'])):
        if pd.notna(data['MA20'][i]):
            if (data['MA5'][i] > data['MA10'][i]) & (data['MA5'][i] > data['MA20'][i]):
                if position == False:
                    Buy.append(data['close'][i])
                    Sell.append(np.nan)
                    position = True
                    Record.append([i, data['close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            elif (data['MA5'][i] < data['MA10'][i]) & (data['MA5'][i] < data['MA20'][i]):
                if position == True:
                    Buy.append(np.nan)
                    Sell.append(data['close'][i])
                    position = False
                    Record.append([i, data['close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    data['Buy'] = Buy
    data['Sell'] = Sell

    return data, Record

def MACD_Strategy(data):
    macd = ta.macd(data["close"])*100
    macd.rename(columns={'MACD_12_26_9': 'MACD', 'MACDh_12_26_9': 'Histogram', 'MACDs_12_26_9': 'Signal'}, inplace=True)
    data = pd.concat([data, macd], axis=1).reindex(data.index)

    Buy=[] #show buy in the graph
    Sell=[] #show sell in the graph
    Record=[] #record buy and sell
    Buy_position = False
    Sell_position = False

    for i in range(len(data['close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['Histogram'][i - 1]):
            if ((data['Histogram'][i - 1]<0) & (data['Histogram'][i]>0)) & \
                ((data['MACD'][i]<0) & (data['Signal'][i]<0)):
                # Check buying signal
                if Sell_position == True:
                    Buy.append(data['close'][i])
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['close'][i])
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            elif ((data['Histogram'][i - 1]>0) & (data['Histogram'][i]<0)) & \
                ((data['MACD'][i]>0) & (data['Signal'][i]>0)):
                # Check selling signal
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['close'][i])
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['close'][i])
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    if len(Record) % 2:
        if Record[-1][2] == 'Buy':
            Buy[Record[-1][0]] = np.nan
        if Record[-1][2] == 'Sell':
            Sell[Record[-1][0]] = np.nan
        Record.pop()

    data['Buy'] = Buy
    data['Sell'] = Sell

    return data, Record

def RSI_Strategy(data):
    data['RSI'] = ta.rsi(data['close'], timeperiod=14)

    Buy=[] #show buy in the graph
    Sell=[] #show sell in the graph
    Record=[] #record buy and sell
    Buy_position = False
    Sell_position = False

    for i in range(len(data['close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['RSI'][i - 1]):
            if ((data['RSI'][i - 1]<30) & (data['RSI'][i]>30)):
                # Check buying signal
                if Sell_position == True:
                    Buy.append(data['close'][i])
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['close'][i])
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            elif ((data['RSI'][i - 1]>70) & (data['RSI'][i]<70)):
                # Check selling signal
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['close'][i])
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['close'][i])
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    if len(Record) % 2:
        if Record[-1][2] == 'Buy':
            Buy[Record[-1][0]] = np.nan
        if Record[-1][2] == 'Sell':
            Sell[Record[-1][0]] = np.nan
        Record.pop()

    data['Buy'] = Buy
    data['Sell'] = Sell

    return data, Record

def BB_Strategy(data):
    bb = ta.bbands(data['close'], length=20, std=2)
    bb.rename(columns={'BBU_20_2.0': 'BBU', 'BBL_20_2.0': 'BBL'}, inplace = True)
    data = pd.concat([data, bb], axis=1).reindex(data.index)

    Buy=[] #show buy in the graph
    Sell=[] #show sell in the graph
    Record=[] #record buy and sell
    Buy_position = False
    Sell_position = False

    for i in range(len(data['close'])):
        if i == 0:
            Buy.append(np.nan)
            Sell.append(np.nan)
        elif pd.notna(data['BBU'][i]):
            if (data['BBL'][i] > (data['close'][i]+data['open'][i])/2):
                # Check buying signal
                if Sell_position == True:
                    Buy.append(data['close'][i])
                    Sell.append(np.nan)
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                elif Buy_position == False:
                    Buy.append(data['close'][i])
                    Sell.append(np.nan)
                    Buy_position = True
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Buy'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            elif (data['BBU'][i] < (data['close'][i]+data['open'][i])/2):
                # Check selling signal
                if Buy_position == True:
                    Buy.append(np.nan)
                    Sell.append(data['close'][i])
                    Buy_position = False
                    Sell_position = False
                    Record.append([i, data['close'][i], 'Sell'])
                elif Sell_position == False:
                    Buy.append(np.nan)
                    Sell.append(data['close'][i])
                    Buy_position = False
                    Sell_position = True
                    Record.append([i, data['close'][i], 'Sell'])
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    if len(Record) % 2:
        if Record[-1][2] == 'Buy':
            Buy[Record[-1][0]] = np.nan
        if Record[-1][2] == 'Sell':
            Sell[Record[-1][0]] = np.nan
        Record.pop()

    data['Buy'] = Buy
    data['Sell'] = Sell

    return data, Record

# Python for finance #3
def figure_design(ax):
    ax.set_facecolor('#091217')
    ax.tick_params(axis='both', labelsize=14, colors='white')
    ax.ticklabel_format(useOffset=False)
    ax.spines['bottom'].set_color('#808080')
    ax.spines['top'].set_color('#808080')
    ax.spines['left'].set_color('#808080')
    ax.spines['right'].set_color('#808080')

# Python for finance #4
def main_plot(data, ax, current_date, showMA=True, showBB=True, showEMA=True, Strategy=True):

    ax.clear()

    if showMA==True:
        data['MA5'] = data["close"].rolling(5).mean()
        data['MA10'] = data["close"].rolling(10).mean()
        data['MA20'] = data["close"].rolling(20).mean()

        ax1.plot(data['MA5'], color='pink', linestyle="-", linewidth=1, label='5 periods SMA')
        ax1.plot(data['MA10'], color='orange', linestyle="-", linewidth=1, label='10 periods SMA')
        ax1.plot(data['MA20'], color='#08a0e9', linestyle="-", linewidth=1, label='20 periods SMA')

    if showEMA == True:
        data['EMA20'] = data["close"].ewm(span=20, adjust=False).mean()
        ax1.plot(data['EMA20'], color='#08a0e9', linestyle="-", linewidth=1, label='20 periods EMA')

    if showBB == True:
        #data = computeBollingerBands(data, 20, 2)
        bb = ta.bbands(data['close'], length=20, std=2)
        data = pd.concat([data, bb], axis=1).reindex(data.index)

        ax1.fill_between(data.index, data['BBU_20_2.0'], data['BBL_20_2.0'],
                         facecolor='#666699', alpha=0.2,
                         label='Bollinger Bands')
        ax1.plot(data['BBU_20_2.0'], color='#666699', linestyle="-", linewidth=0.2)
        ax1.plot(data['BBL_20_2.0'], color='#666699', linestyle="-", linewidth=0.2)

    if (showMA == True) | (showBB == True) | (showEMA == True):
        leg = ax1.legend(loc='upper left', facecolor='#121416', fontsize=10)
        #for text in leg.get_texts():
        #    text.set_color('w')
        plt.setp(leg.get_texts(), color='w')

    if Strategy == True:
        #data, Record = MA_Strategy(data)
        #data, Record = MACD_Strategy(data)
        #data, Record = RSI_Strategy(data)
        data, Record = BB_Strategy(data)

        ax1.scatter(data.index, data['Buy'], label='Buy', marker='^', color='#00FFBD', alpha=1, s=150)
        ax1.scatter(data.index, data['Sell'], label='Sell', marker='v', color='#FF6FFF', alpha=1, s=150)

        Profit=0
        margin = 0.95
        i = 1
        for item in Record:
            message = str(i) + ' ' + str(item[2]) + '@' + str(item[1])
            if item[2] == 'Buy':
                ax1.text(1.01, margin, message,
                         bbox=dict(facecolor='green', alpha=0.5),
                         transform=ax1.transAxes, color='white', fontsize=9, fontweight='bold',
                         horizontalalignment='left', verticalalignment='center')

                Profit = Profit - float(item[1])
            else:
                ax1.text(1.01, margin, message,
                         bbox=dict(facecolor='red', alpha=0.5),
                         transform=ax1.transAxes, color='white', fontsize=9, fontweight='bold',
                         horizontalalignment='left', verticalalignment='center')

                Profit = Profit + float(item[1])

            margin = margin - 0.055
            i = i + 1

        #if Record[-1][2] == 'Buy':
        #    Profit = Profit + float(Record[-1][1])

        ax1.text(0.83, 1.05, 'Daily Profit: $' + str(round(Profit, 3)) + ' per lot',
                 bbox=dict(facecolor='white', alpha=0.5),
                 transform=ax1.transAxes, color='black', fontsize=16, fontweight='bold',
                 horizontalalignment='left', verticalalignment='center')

    else:
        Profit=0

    ####################################################################
    candle_counter = range(len(data["open"]) - 1)
    ohlc = []
    for candle in candle_counter:
        append_me = candle_counter[candle], data["open"][candle], \
                    data["high"][candle], data["low"][candle], \
                    data["close"][candle]
        ohlc.append(append_me)
    ####################################################################
    ####################################################################
    if Strategy == True:
        candlestick_ohlc(ax, ohlc, width=0.4, colorup='#006400', colordown='#8B0000')
    else:
        candlestick_ohlc(ax, ohlc, width=0.4, colorup='#18b800', colordown='#ff3503')
    ####################################################################

    figure_design(ax)

#Given date - us

    ax1.text(0.5, 1.05, 'Date - 05-05-2023' ,
             transform=ax1.transAxes, color='white', fontsize=16, fontweight='bold',
             horizontalalignment='center', verticalalignment='center')
    
#    ax1.text(0.5, 1.05, 'Date ' + current_date,
#             transform=ax1.transAxes, color='white', fontsize=16, fontweight='bold',
#             horizontalalignment='center', verticalalignment='center')

    ax.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    ax.set_xticklabels([])

    return Profit

def subplot_macd(data, ax):
    ax.clear()
    figure_design(ax)

    # 1st change
    macd = ta.macd(data["close"]).fillna(0)  * 100
    data = pd.concat([data, macd], axis=1).reindex(data.index)

    # 2nd change
    ax2.plot(np.where(data['MACD_12_26_9'] == 0, data['MACD_12_26_9'], None), label='MACD', linewidth=1, alpha = 0)
    ax2.plot(np.where(data['MACD_12_26_9'] != 0, data['MACD_12_26_9'], None), label='MACD', linewidth=1, color='white')

    ax2.plot(np.where(data['MACDs_12_26_9'] == 0, data['MACDs_12_26_9'], None), label='signal', linewidth=1,  alpha = 0)
    ax2.plot(np.where(data['MACDs_12_26_9'] != 0, data['MACDs_12_26_9'], None) , label='signal', linewidth=1, color='orange')

    pos = data['MACDh_12_26_9'] > 0
    neg = data['MACDh_12_26_9'] < 0

    ax2.bar(data.index[pos], data['MACDh_12_26_9'][pos], color='#006400', width=0.8, align='center')
    ax2.bar(data.index[neg], data['MACDh_12_26_9'][neg], color='#8B0000', width=0.8, align='center')

    if len(data['MACD_12_26_9']) != 0:
        ax.text(0.01, 0.95, 'MACD(12, 26, 9)', transform=ax.transAxes, color='white',
                fontsize=10, fontweight='bold',
                horizontalalignment='left', verticalalignment='top')

    ax.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    ax.set_xticklabels([])

def subplot_rsi(data, ax):
    ax.clear()
    figure_design(ax)

    ax.axes.yaxis.set_ticks([30, 70])
    ax.set_ylim([-2, 102])

    #data['RSI'] = computeRSI(data["close"], 14)

    # 3rd change
    data['RSI'] = ta.rsi(data['close'], timeperiod=14).fillna(0)
    #data['x_axis'] = list(range(1, len(data['close']) + 1))

    # 4th change
    ax.plot(np.where(data['RSI'] == 0, data['RSI'], None), color='white', alpha = 0)
    ax.plot(np.where(data['RSI'] != 0, data['RSI'], None), color='white', linewidth=1)

    if len(data['RSI']) != 0:
        ax.text(0.01, 0.95, 'RSI(14)', transform=ax.transAxes, color='white',
                fontsize=10, fontweight='bold',
                horizontalalignment='left', verticalalignment='top')

    data['datetime'] = pd.to_datetime(data['datetime'], format="%Y-%m-%d %H:%M:%S")
    xdate = [i for i in data['datetime']]

    def mydate(x, pos=None):
        try:
            t = xdate[int(x)].strftime('%H:%M')
            return xdate[int(x)].strftime('%H:%M')
        except IndexError:
            return ''

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(mydate))

    ax.axhline(30, linestyle='-', color='green', linewidth=0.5)
    ax.axhline(50, linestyle='-', alpha=0.5, color='white', linewidth=0.5)
    ax.axhline(70, linestyle='-', color='red', linewidth=0.5)

    ax.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    ax.tick_params(axis='x', which='major', labelsize=10)

# Python for finance #3
fig = plt.figure()
#fig = plt.figure(figsize=(16.0, 10.0))
fig = plt.figure(figsize=(14.0, 10.0))
fig.patch.set_facecolor('#121416')
gs = fig.add_gridspec(6, 6)
ax1 = fig.add_subplot(gs[0:4, 0:6])
ax2 = fig.add_subplot(gs[4, 0:6])
ax3 = fig.add_subplot(gs[5, 0:6])

#list_of_files = glob.glob('C:\\Users\\daryle\\Desktop\\Acquire AI\\Python Finance - Algo Trade\\AAPL\\check point\\*.csv')
list_of_files = glob.glob('C:\\Users\\ashut\\Final_year-proj_Ashu\\Data_Collect\\VNO_final_c.csv')
data_full = pd.read_csv(list_of_files[-1], header=0)

date_list = [x[0:10] for x in data_full['datetime']]
date_list = sorted(set(date_list))

def animate(i):
    data_day, current_date = backtest_day(i, data_full)

    if not data_day.empty:
        Profit = main_plot(data_day, ax1, current_date, showMA=False, showBB=True, showEMA=True, Strategy=True)
        subplot_macd(data_day, ax2)
        subplot_rsi(data_day, ax3)
        computeProfit(i, Profit, ax1)
    plt.savefig('C:\\Users\\ashut\\Final_year-proj_Ashu\\Data_Collect\\'
                + str(current_date) +'.png', dpi = 300)

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()
