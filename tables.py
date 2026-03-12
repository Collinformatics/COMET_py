from functions import pressKey
import matplotlib.pyplot as plt
import pandas as pd


# Make table
inSigFigs = 0
inData = {
    'Substrates': ['AVLQSGFR', 'VILQSGFR', 'VILQTGFR', 'VILQSPFR',
                   'VILHSGFR', 'VIMQSGFR', 'VPLQSGFR', 'NILQSGFR'],
    '% Product': [0.761, 1.000, 0.110, 0.000, 0.222, 0.807, 0.000, 0.175],
    'Predicted': [0.628, 1.000, 0.019, 0.004, 0.054, 0.404, 0.004, 0.048],
    'Predicted Sq Root': [0.793, 1.0, 0.138, 0.063, 0.233, 0.635, 0.060, 0.221],
}


# Convert to %
table = pd.DataFrame.from_dict(inData)
for col in table.columns[1:]:
    if inSigFigs == 0 or not inSigFigs:
        table[col] = table[col] = (table[col] * 100).astype(int)
    else:
        table[col] = (table[col] * 100).round(inSigFigs)
print(f'{table}\n')

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
