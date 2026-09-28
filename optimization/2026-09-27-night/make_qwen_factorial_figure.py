"""Publication-style view of every predeclared oracle factorial condition."""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE


def main():
    data = json.loads((HERE/'qwen_factorial_summary.json').read_text())
    rows = [('plain', '(14, 20)'), ('chat', '(14, 20)'),
            ('plain', '(24,)'), ('chat', '(24,)')]
    cols = [(1., 1), (1., 4), (2., 1), (2., 4)]
    matrix = np.array([[data['configuration_means'][f'{prompt}_layers{layers}_gain{gain}_positions{positions}']
                       for gain, positions in cols] for prompt, layers in rows])
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(7.6,4.4))
    im = ax.imshow(matrix*100, cmap='Blues', vmin=0, vmax=30, aspect='auto')
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{100*matrix[i,j]:.1f}',ha='center',va='center',
                    color='white' if matrix[i,j]>.17 else '#152536',fontweight='bold',fontsize=12)
    ax.set_xticks(range(4),['Gain 1 / 1 pos.','Gain 1 / 4 pos.','Gain 2 / 1 pos.','Gain 2 / 4 pos.'])
    ax.set_yticks(range(4),['Plain / layers 14+20','Native / layers 14+20',
                          'Plain / layer 24','Native / layer 24'])
    ax.tick_params(length=0,pad=8)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title('Qwen3: intervention setting changes steerability',loc='left',pad=15,fontweight='bold')
    colorbar=fig.colorbar(im,ax=ax,fraction=.042,pad=.04)
    colorbar.set_label('Oracle accuracy (%)')
    fig.text(.025,.09,'All 16 conditions on 20 non-development known targets; mined target activations are available.',fontsize=8.5)
    fig.text(.025,.047,'“Native” includes a generic system instruction. Layer setting changes depth and layer count.',fontsize=8.5)
    fig.subplots_adjust(left=.255,right=.9,top=.84,bottom=.25)
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'qwen_oracle_factorial.{ext}',dpi=180)
    with (out/'qwen_oracle_factorial.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'layers', 'gain1_pos1', 'gain1_pos4', 'gain2_pos1', 'gain2_pos4'])
        for (prompt, layers), values in zip(rows, matrix):
            writer.writerow([prompt, layers, *values])
    plt.close(fig)


if __name__=='__main__':main()
