from functions import pressKey
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import sys


# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)


"""
    If a key in inData is "% St Dev", table will exclude it.
    The "% Product" values will be converted from decimal to percentage.
    The order of the keys in inData will have the same order in the table.
"""

# Make table
inEnzyme = f'M{"ᵖʳᵒ"}'
inEnzyme2 = f'M{"ᵖʳᵒ"}2'
inSigFigs = 0
inSubstrates = ['AVLQSGFR', 'VILQSGFR', 'VILQTGFR', 'VILQSPFR',
                'VILHSGFR', 'VIMQSGFR', 'VPLQSGFR', 'NILQSGFR']
inEmptyList = [0 for i in range(1, len(inSubstrates) + 1)]
inData = {
    'Substrates': inSubstrates,

    f'% Product {inEnzyme}': [0.54, 0.66, 0.34, 0.0, 0.26, 0.61, 0.0, 0.22],
    f'Activity Z {inEnzyme}': inEmptyList,
    f'% Product Rank {inEnzyme}': inEmptyList,
    f'% St Dev {inEnzyme}': [0.01, 0.058, 0.025, 0.0, 0.027, 0.044, 0.0, 0.033],
    f'Predicted {inEnzyme}': [0.572, 1.0, 0.015, 0.01, 0.038, 0.51, 0.034, 0.066],
    f'Pred Z {inEnzyme}': inEmptyList,
    # f'Predicted ln {inEnzyme}': inEmptyList,
    f'Predicted Rank {inEnzyme}': inEmptyList,

    f'% Product {inEnzyme2}':      [0.76, 0.82, 0.38, 0.0, 0.32, 0.66, 0.0, 0.23],
    f'Activity Z {inEnzyme2}': inEmptyList,
    f'% Product Rank {inEnzyme2}': inEmptyList,
    f'% St Dev {inEnzyme2}': [0.1, 0.09, 0.02, 0, 0.06, 0.09, 0, 0.05],
    f'Predicted {inEnzyme2}': [0.976, 1.0, 0.035, 0.005, 0.053, 0.362, 0.006, 0.01],
    f'Pred Z {inEnzyme2}': inEmptyList,
    # f'Predicted ln {inEnzyme2}': inEmptyList,
    f'Predicted Rank {inEnzyme2}': inEmptyList,
}
inPlotTable = True
inNatLog = False

# Z-Scores
inCalcZScores = [ # Format: (x, y), use "x" values to calc the Z-Score saved in "y"
    (f'% Product {inEnzyme}', f'Activity Z {inEnzyme}'),
    (f'Predicted {inEnzyme}', f'Pred Z {inEnzyme}'),
    (f'% Product {inEnzyme2}', f'Activity Z {inEnzyme2}'),
    (f'Predicted {inEnzyme2}', f'Pred Z {inEnzyme2}')
]

# Scatter plot
inPlotBoth = True
inScatter = [f'Activity {inEnzyme2}', f'Activity Z {inEnzyme2}',
             f'Pred {inEnzyme2}', f'Pred Z {inEnzyme2}']

# Save Figure
inSavePath = ''
inFigResolution = 600


def normalizeData():
    for key in inData.keys():
        if '% Product' in key and 'Rank' not in key and 'ln' not in key:
            maxValue = max(inData[key])
            norm = []
            for val in inData[key]:
                norm.append(val / maxValue)
            inData[key] = norm


def convertNum(data, key):
    keyStDev = '% St Dev'
    if key != '% St Dev':
        keyStDev = f'% St Dev{key.replace("% Product", "")}'

    activity = [] # Convert to %
    for idx, substrate in enumerate(data['Substrates']):
        act = data[key][idx] * 100
        stdev = int(data[keyStDev][idx] * 100)
        if inSigFigs == 0 or not inSigFigs:
            act = int(act)
        else:
            act = round(act, inSigFigs)
        if stdev == 0.0:
            activity.append(act)
        else:
            activity.append(f'{act} ± {stdev}')
    return activity


def plotTable():
    data = pd.DataFrame(0.0, index=range(len(inData['Substrates'])), columns=[])
    for col in inData.keys():
        if '% st dev' in col.lower():
            continue
        if '% product' in col.lower() and 'rank' not in col.lower():
            data[col] = convertNum(inData, col)
        elif ('predicted' in col.lower() and
              'rank' not in col.lower() and
              'ln' not in col.lower()):
            val = []
            for v in inData[col]:
                if inSigFigs == 0 or not inSigFigs:
                    val.append(int(v * 100))
                else:
                    val.append((v * 100).round(inSigFigs))
            data[col] = val
        elif 'rank' in col.lower():
            key = col.replace(' Rank', '')
            data.loc[:, col] = (
                pd.Series(inData[key]).rank(ascending=False, method='min').astype(int)
            )
        elif 'substrates' in col.lower():
            data[col] = inData[col]
        else:
            print(f'* {col}')
            subs = inData['Substrates']
            print(inData[col])

            for i in range(len(inData[col])):
                data.loc[i, col] = round(inData[col][i], 3)
            print(f'Data:\n{data.loc[:, col]}\n')

    table = pd.DataFrame.from_dict(data)
    table.drop(f'Predicted {inEnzyme}', axis=1, inplace=True)
    table.drop(f'Predicted {inEnzyme2}', axis=1, inplace=True)
    print(f'Table:\n{table}\n')

    # Make figure
    h = (len(table.index) / 2) - 1
    w = len(table.columns) * 2
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis('off')  # hide axes
    tbl = plt.table(cellText=table.values,
                    colLabels=table.columns,
                    loc='center',
                    cellLoc='center',
                    bbox=(0, 0, 1, 1)
                    )
    tbl.scale(1, 2)

    # Bold headers
    for i in range(len(table.columns)):
        tbl[0, i].set_text_props(weight='bold')

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)

    for (row, col), cell in tbl.get_celld().items():
        text = cell.get_text()
        text.set_fontname('Times New Roman')
        text.set_verticalalignment('top')

    fig.tight_layout(pad=0.5)
    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()

    if inSavePath:
        fig.savefig(inSavePath, dpi=inFigResolution)
        print(f'Saving figure at path:\n'
              f'     {inSavePath}\n')
    else:
        print(f'The figure was not saved\n')


def zScore(values, tag):
    print(f'Calculate Z Score: {tag}')
    avg = np.mean(values)
    stdev = np.std(values)
    z = []
    for x in values:
        z.append((x - avg) / stdev)

    rVal = 3
    for i in range(len(values)):
        x, y = round(float(values[i]),rVal), round(float(z[i]),rVal)
        print(f'* Value: {x}, Z Score: {y}')
    print()
    return z


def ln(vals):
    x = []
    for val in vals:
        l = np.log(val)
        x.append(l)
        print(f'*: ln({round(val,3)}) -> {round(l,3)}')
    print()
    return x


from scipy.optimize import curve_fit

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def fitExp(x, y):
    # Fit the curve
    popt, pcov = curve_fit(exp_func, x, y, p0=[1, 1, 0], maxfev=10000)
    a, b, c = popt

    # Generate smooth curve for plotting
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = exp_func(x_fit, *popt)

    # R² for the exponential fit
    y_pred = exp_func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    exp_r2 = 1 - (ss_res / ss_tot)
    return x_fit, y_fit, exp_r2


# ========================================================================================


normalizeData()
# z = True
# if z:
#     def alt(n, m):
#         q = (n + m) * (100 / m)
#         return round(q, 5)
#
#     print(f'Values:\n{inData[f'Predicted {inEnzyme}']}')
#     print('ln()')
#     x = ln(inData[f'Predicted {inEnzyme}'])
#     # x = zScore(x, inEnzyme)
#     for i in range(len(x)):
#         n = x[i]
#         m = 4.605
#         x[i] = alt(n, m)
#     inData[f'Predicted ln {inEnzyme}'] = x
#     # x = zScore(inData[f'Predicted {inEnzyme}'], inEnzyme)
#     x = ln(inData[f'Predicted {inEnzyme2}'])
#     for i in range(len(x)):
#         n = x[i]
#         m = 5.298
#         x[i] = alt(n, m)
#     inData[f'Predicted ln {inEnzyme2}'] = x
# else:
#     inData[f'Predicted ln {inEnzyme}'] = ln(inData[f'Predicted {inEnzyme}'])
#     inData[f'Predicted ln {inEnzyme2}'] = ln(inData[f'Predicted {inEnzyme2}'])


# Calculate Z-Scores
for tags in inCalcZScores:
    inData[tags[1]] = zScore(inData[tags[0]], tags[0])
    # inData[tags[1]] = ln(inData[tags[0]])

# if inPlotTable:
#     plotTable()

# Evaluate the natural log
if inNatLog:
    inData[f'Predicted {inEnzyme}'] = ln(inData[f'Predicted {inEnzyme}'])
    inData[f'Predicted {inEnzyme2}'] = ln(inData[f'Predicted {inEnzyme2}'])

# ========================================================================================
# Evaluate data
data = pd.DataFrame(0.0, index=inData['Substrates'],
                    columns=inScatter)
data[f'Activity {inEnzyme2}'] = inData[f'% Product {inEnzyme2}']
data[f'Activity Z {inEnzyme2}'] = inData[f'Activity Z {inEnzyme2}']
data[f'Pred {inEnzyme2}'] = inData[f'Predicted {inEnzyme2}']
data[f'Pred Z {inEnzyme2}'] = inData[f'Pred Z {inEnzyme2}']

actR2 = round(r2_score(data[f'Activity {inEnzyme2}'], data[f'Pred {inEnzyme2}']), 3)
actZR2 = round(r2_score(data[f'Activity Z {inEnzyme2}'], data[f'Pred Z {inEnzyme2}']), 3)
print(f'\nZ Scores: {inEnzyme2}\n{data}\n')


# Evaluate a second set
if inPlotBoth:
    data[f'Activity {inEnzyme}'] = inData[f'% Product {inEnzyme}']
    data[f'Activity Z {inEnzyme}'] = inData[f'Activity Z {inEnzyme}']
    data[f'Pred {inEnzyme}'] = inData[f'Predicted {inEnzyme}']
    data[f'Pred Z {inEnzyme}'] = inData[f'Pred Z {inEnzyme}']
    print(f'\nZ Scores: {inEnzyme}\n{data}\n')
    actEnzR2 = round(
        r2_score(data[f'Activity {inEnzyme}'], data[f'Pred {inEnzyme}']), 3
    )
    actEnzZR2 = round(
        r2_score(data[f'Activity Z {inEnzyme}'], data[f'Pred Z {inEnzyme}']), 3
    )


# Plot data
inTitleSize = 16
inLabelSize = 14
c1, c2, = '#2E9418', '#BF5700'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# data.plot(
#     x=f'Activity {inEnzyme2}', y=f'Pred {inEnzyme2}', ax=ax1, marker='o',
#     linestyle='none', color=c2, legend=f'R² = {actR2}'
# )
# data.plot(
#     x=f'Activity Z {inEnzyme2}', y=f'Pred Z {inEnzyme2}', ax=ax2, marker='s',
#     linestyle='none', color=c2, legend=f'R² = {actZR2}'
# )
x_fit, y_fit, fitCurve = fitExp(x=data[f'Activity {inEnzyme2}'].values,
                            y=data[f'Pred {inEnzyme2}'].values)
data.plot(
    x=f'Activity {inEnzyme2}', y=f'Pred {inEnzyme2}', ax=ax1, marker='o',
    linestyle='none', color=c2, label=f'R² = {fitCurve:.3f}'
)
ax1.plot(x_fit, y_fit, color=c2, linestyle='-', linewidth=1.5)

x_fit2, y_fit2, fitCurve2 = fitExp(x=data[f'Activity Z {inEnzyme2}'].values,
                                   y=data[f'Pred Z {inEnzyme2}'].values)
data.plot(
    x=f'Activity Z {inEnzyme2}', y=f'Pred Z {inEnzyme2}', ax=ax2, marker='s',
    linestyle='none', color=c2, legend=f'R² = {fitCurve2:.3f}'
)
ax2.plot(x_fit2, y_fit2, color=c2, linestyle='-', linewidth=1.5)


if inPlotBoth:
    # data.plot(
    #     x=f'Activity {inEnzyme}', y=f'Pred {inEnzyme}', ax=ax1, marker='D',
    #     linestyle='none', color=c1, legend=f'R² = {actEnzR2}'
    # )
    # data.plot(
    #     x=f'Activity Z {inEnzyme}', y=f'Pred Z {inEnzyme}', ax=ax2, marker='*',
    #     linestyle='none', color=c1, legend=f'R² = {actEnzZR2}'
    # )
    x_fit3, y_fit3, fitCurve3 = fitExp(x=data[f'Activity {inEnzyme}'].values,
                                       y=data[f'Pred {inEnzyme}'].values)
    data.plot(
        x=f'Activity Z {inEnzyme}', y=f'Pred Z {inEnzyme}', ax=ax1, marker='s',
        linestyle='none', color=c1, legend=f'R² = {fitCurve3:.3f}'
    )
    ax1.plot(x_fit2, y_fit2, color=c1, linestyle='--', linewidth=1.5)

    x_fit4, y_fit4, fitCurve4 = fitExp(x=data[f'Activity Z {inEnzyme}'].values,
                                       y=data[f'Pred Z {inEnzyme}'].values)
    data.plot(
        x=f'Activity Z {inEnzyme}', y=f'Pred Z {inEnzyme}', ax=ax2, marker='s',
        linestyle='none', color=c1, legend=f'R² = {fitCurve4:.3f}'
    )
    ax2.plot(x_fit4, y_fit4, color=c1, linestyle='--', linewidth=1.5)


fig.suptitle(f'Enzyme Activity', fontsize=inTitleSize,
             fontweight='bold', va='top')

# X Axis
ax1.set_xlabel('Experimental Activity', fontsize=inLabelSize)
ax2.set_xlabel('Experimental Activity Z Scores', fontsize=inLabelSize)
ax1.set_xticks(ax1.get_xticks()) # Fix the tick positions
ax1.set_xticklabels(ax1.get_xticklabels(), ha='center', fontsize=inLabelSize-2)
ax2.set_xticks(ax2.get_xticks()) # Fix the tick positions
ax2.set_xticklabels(ax2.get_xticklabels(), ha='center', fontsize=inLabelSize-2)

# Y Axis
ax1.set_ylabel('Predicted Activity', fontsize=inLabelSize)
ax2.set_ylabel('Predicted Z Scores', fontsize=inLabelSize)
ax1.tick_params(axis='y', labelsize=inLabelSize-2)
ax2.tick_params(axis='y', labelsize=inLabelSize-2)

# Legend
if inPlotBoth:
    # ax1.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
    #     [], [], linestyle='None', marker='None',
    #     label=f'R² {inEnzyme}  = {actEnzR2:.3f}\nR² {inEnzyme2} = {actR2:.3f}')],
    #            handletextpad=0, handlelength=0
    # )
    # ax2.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
    #     [], [], linestyle='None', marker='None',
    #     label=f'R² {inEnzyme}  = {actEnzZR2:.3f}\nR² {inEnzyme2} = {actZR2:.3f}')],
    #            handletextpad=0, handlelength=0
    # )
    ax1.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
        [], [], linestyle='None', marker='None',
        label=f'R² {inEnzyme}  = {fitCurve3:.3f}\nR² {inEnzyme2} = {fitCurve:.3f}')],
               handletextpad=0, handlelength=0
               )
    ax2.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
        [], [], linestyle='None', marker='None',
        label=f'R² {inEnzyme}  = {fitCurve4:.3f}\nR² {inEnzyme2} = {fitCurve2:.3f}')],
               handletextpad=0, handlelength=0
               )
else:
    # ax1.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
    #     [], [], linestyle='None', marker='None',
    #     label=f'R² {inEnzyme2} = {actR2:.3f}')], handletextpad=0, handlelength=0
    # )
    # ax2.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
    #     [], [], linestyle='None', marker='None',
    #     label=f'R² {inEnzyme2} = {actZR2:.3f}')], handletextpad=0, handlelength=0
    # )
    ax1.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
        [], [], linestyle='None', marker='None',
        label=f'R² {inEnzyme2} = {fitCurve:.3f}')], handletextpad=0, handlelength=0
               )
    ax2.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
        [], [], linestyle='None', marker='None',
        label=f'R² {inEnzyme2} = {fitCurve2:.3f}')], handletextpad=0, handlelength=0
               )

plt.tight_layout()
fig.canvas.mpl_connect('key_press_event', pressKey)
plt.show()
