from functions import pressKey
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


pd.set_option('display.float_format', '{:,.3f}'.format)


# Make table
inSigFigs = 0
inData = {
    'Substrates': ['AVLQSGFR', 'VILQSGFR', 'VILQTGFR', 'VILQSPFR',
                   'VILHSGFR', 'VIMQSGFR', 'VPLQSGFR', 'NILQSGFR'],
    '% Product': [0.761, 1.000, 0.110, 0.000, 0.222, 0.807, 0.000, 0.175],
    'Predicted': [0.628, 1.000, 0.019, 0.004, 0.054, 0.404, 0.004, 0.048],
    'Predicted Sq Root': [0.793, 1.0, 0.138, 0.063, 0.233, 0.635, 0.060, 0.221],
}

# Save Figure
inSavePath = ''
inFigResolution = 600


# Make figure
def plotTable():
    # Convert to %
    table = pd.DataFrame.from_dict(inData)
    for col in table.columns[1:]:
        if inSigFigs == 0 or not inSigFigs:
            table[col] = table[col] = (table[col] * 100).astype(int)
        else:
            table[col] = (table[col] * 100).round(inSigFigs)
    print(f'{table}\n')

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
# plotTable()


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
                    columns=['Activity', 'Activity Z', 'Pred', 'Pred Z'])
activity = inData['% Product']
data['Activity'] = activity
data['Activity Z'] = zScore(activity)
pred = inData['Predicted']
data['Pred'] = pred
data['Pred Z'] = zScore(pred)
print(f'\nZ Scores:\n{data}\n')

actR2 = round(r2_score(data['Activity'], data['Pred']), 2)
actZR2 = round(r2_score(data['Activity Z'], data['Pred Z']), 2)

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
