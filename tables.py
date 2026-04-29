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
inEnzyme = f'M{"ᵖʳᵒ"}2'
inEnzyme2 = f'M{"ᵖʳᵒ"}'
inSigFigs = 0
inSubstrates = ['AVLQSGFR', 'VILQSGFR', 'VILQTGFR', 'VILQSPFR',
                   'VILHSGFR', 'VIMQSGFR', 'VPLQSGFR', 'NILQSGFR']
inData = {
    'Substrates': inSubstrates,
    f'% Product {inEnzyme}':      [0.76, 0.82, 0.38, 0.0, 0.32, 0.66, 0.0, 0.23],
    f'% Product Rank {inEnzyme}': [0 for i in range(1, len(inSubstrates) + 1)],
    f'% St Dev {inEnzyme}': [0.1, 0.09, 0.02, 0, 0.06, 0.09, 0, 0.05],
    f'Predicted {inEnzyme}': [0.628, 1.000, 0.019, 0.004, 0.054, 0.404, 0.004, 0.048],
    f'Predicted Rank {inEnzyme}': [0 for i in range(1, len(inSubstrates) + 1)],

    f'% Product {inEnzyme2}': [0.54, 0.66, 0.34, 0.0, 0.26, 0.61, 0.0, 0.22],
    f'% Product Rank {inEnzyme2}': [i for i in range(1, len(inSubstrates) + 1)],
    f'% St Dev {inEnzyme2}': [0.01, 0.058, 0.025, 0.0, 0.027, 0.044, 0.0, 0.033],
    f'Predicted {inEnzyme2}': [0.572,1.0,0.015,0.010,0.038,0.510,0.034,0.066],
    f'Predicted Rank {inEnzyme2}': [0 for i in range(1, len(inSubstrates) + 1)],
}

# Scatter plot
inScatter = [f'Activity {inEnzyme}', f'Activity Z {inEnzyme}',
             f'Pred {inEnzyme}', f'Pred Z {inEnzyme}']

# Save Figure
inSavePath = ''
inFigResolution = 600


def normalizeData():
    for key in inData.keys():
        if '% Product' in key and 'Rank' not in key:
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


# Make figure
def plotTable():
    data = pd.DataFrame(0.0, index=range(len(inData['Substrates'])), columns=[])
    for col in inData.keys():
        if '% St Dev' in col:
            continue
        if '% Product' in col and 'Rank' not in col:
            data[col] = convertNum(inData, col)
        elif 'Predicted' in col and 'Rank' not in col:
            val = []
            for v in inData[col]:
                if inSigFigs == 0 or not inSigFigs:
                    val.append(int(v * 100))
                else:
                     val.append((v * 100).round(inSigFigs))
            data[col] = val
        elif 'Rank' in col:
            key = col.replace(' Rank', '')
            data.loc[:, col] = (
                pd.Series(inData[key]).rank(ascending=False, method='min').astype(int)
            )
        else:
            data[col] = inData[col]

    table = pd.DataFrame.from_dict(data)
    print(f'Table:\n{table}\n')

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
        print(f'The figure was not saved')

normalizeData()
plotTable()


# ========================================================================================
def zScore(values):
    avg = np.mean(values)
    stdev = np.std(values)
    z = []
    for x in values:
        z.append((x - avg) / stdev)
    return z


# Evaluate data
data = pd.DataFrame(0.0, index=inData['Substrates'],
                    columns=inScatter)
activity = inData[f'% Product {inEnzyme}']
data[f'Activity {inEnzyme}'] = activity
data[f'Activity Z {inEnzyme}'] = zScore(activity)
pred = inData[f'Predicted {inEnzyme}']
data[f'Pred {inEnzyme}'] = pred
data[f'Pred Z {inEnzyme}'] = zScore(pred)
print(f'\nZ Scores:\n{data}\n')
actR2 = round(r2_score(data[f'Activity {inEnzyme}'], data[f'Pred {inEnzyme}']), 2)
actZR2 = round(r2_score(data[f'Activity Z {inEnzyme}'], data[f'Pred Z {inEnzyme}']), 2)

# Evaluate a second set
if inEnzyme2:
    activity = inData[f'% Product {inEnzyme2}']
    data[f'Activity {inEnzyme2}'] = activity
    data[f'Activity Z {inEnzyme2}'] = zScore(activity)
    pred = inData[f'Predicted {inEnzyme2}']
    data[f'Pred {inEnzyme2}'] = pred
    data[f'Pred Z {inEnzyme2}'] = zScore(pred)
    print(f'\nZ Scores:\n{data}\n')


sys.exit()

inTitleSize = 16
inLabelSize = 14

# Plot data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
data.plot(
    x='Activity', y='Pred', ax=ax1, marker='o',
    linestyle='none', color='#BF5700', legend=f'R² = {actR2}'
)
data.plot(
    x='Activity Z', y='Pred Z', ax=ax2, marker='s',
    linestyle='none', color='#BF5700', legend=f'R² = {actZR2}'
)
fig.suptitle('Substrate Activity', fontsize=inTitleSize, fontweight='bold', va='top')

# X Axis
ax1.set_xlabel('Activity', fontsize=inLabelSize)
ax2.set_xlabel('Activity Z Scores', fontsize=inLabelSize)
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
ax1.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
    [], [], linestyle='None', marker='None',
    label=f'R² = {actR2:.3f}')], handletextpad=0, handlelength=0
)
ax2.legend(prop=FontProperties(size=10, weight='bold'), handles=[Line2D(
    [], [], linestyle='None', marker='None',
    label=f'R² = {actZR2:.3f}')], handletextpad=0, handlelength=0
)

plt.tight_layout()
plt.show()
