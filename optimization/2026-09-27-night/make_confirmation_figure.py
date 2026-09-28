"""A source-backed scientific figure for the completed original-query check."""
import csv
import hashlib
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import HERE

parser=argparse.ArgumentParser()
parser.add_argument('--split',choices=['confirm','fresh'],default='confirm')
parser.add_argument('--metric',choices=['submitted_prefix','task_case_literal_line'],default='submitted_prefix')
parser.add_argument('--refined-control',action='store_true')
args=parser.parse_args()
source=json.loads((HERE/'extension_comparisons.json').read_text())['splits'][args.split]
if args.refined_control:
    assert (HERE/f'position_{args.split}_done.json').exists()
    control=json.loads((HERE/'live_summary.json').read_text())[f'position_{args.split}']['suites']
    literal=json.loads((HERE/'metric_robustness.json').read_text())['experiments'][f'position_{args.split}']
suites=['nondevelopment_loto','novel','compound']
labels=['Non-development\nLOTO (20 tasks)','Novel\n(15 tasks)','Compound\n(10 tasks)']
methods=[('original','Submitted steering','#929292'),
         ('dictionary_free_refined','Position-tuned dictionary-free' if args.refined_control else 'Initial dictionary-free baseline','#56B4E9'),
         ('combined_no_correction','Contextual: correction off','#A7D6BC'),
         ('combined','Contextual CASS','#007F5F'),('icl4','Four-shot ICL','#E69F00')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(10.4,3.5),gridspec_kw={'width_ratios':[1.55,1]})
x=np.arange(3);width=.155;export=[]
for i,(method,label,color) in enumerate(methods):
    values=[]
    for suite in suites:
        cell=source[suite][args.metric]['means'][method]
        assert cell['n_tasks']=={'nondevelopment_loto':20,'novel':15,'compound':10}[suite]
        value=cell['accuracy'];record_method=method
        if args.refined_control and method=='dictionary_free_refined':
            key='heldout_known' if suite=='nondevelopment_loto' else suite
            entry=control[key] if args.metric=='submitted_prefix' else literal[key][args.metric]
            value=entry['means']['dictionary_free'];record_method='dictionary_free_position'
        values.append(value);export.append(dict(suite=suite,method=record_method,accuracy=value))
    bars=ax.bar(x+(i-2)*width,values,width=.14,label=label,color=color)
    for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.012,f'{v:.2f}',ha='center',va='bottom',fontsize=7)
ax.set_xticks(x,labels);ax.set_ylabel('Accuracy')
ax.set_ylim(0,1.);ax.set_title('(a) Matched four-example adaptation',loc='left',fontsize=10)
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
for i,suite in enumerate(suites):
    cell=source[suite][args.metric]['comparisons']['combined__combined_no_correction']
    point=100*cell['diff'];lo,hi=100*np.asarray(cell['ci95'])
    bx.errorbar(point,2-i,xerr=np.array([[point-lo],[hi-point]]),fmt='o',color='#007F5F',capsize=4,markersize=5)
    bx.text(7.8,2-i,f'{point:+.2f}\n[{lo:.2f}, {hi:.2f}]',ha='right',va='center',fontsize=8)
bx.axvline(0,color='#666666',linestyle='--',linewidth=.8)
bx.set_yticks([2,1,0],['LOTO20','Novel15','Compound10']);bx.set_xlim(-.4,8.);bx.set_ylim(-.5,2.5);bx.set_xticks([0,2,4,6])
bx.set_xlabel('Correction gain (percentage points)')
bx.set_title('(b) Only the correction is disabled',loc='left',fontsize=10)
bx.grid(axis='x',alpha=.18);bx.set_axisbelow(True)
handles,names=ax.get_legend_handles_labels()
fig.legend(handles,names,ncol=3,loc='lower center',bbox_to_anchor=(.5,.06),frameon=False,fontsize=8)
split_label='Original queries' if args.split=='confirm' else '50 additional queries per task'
metric_label='submitted prefix metric' if args.metric=='submitted_prefix' else 'case-aware literal-first-line metric'
fig.text(.5,.015,f'{split_label}; {metric_label}; three seeds. Paired 95% intervals resample task means; no routing.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.2,1,1),w_pad=2.4)
out=HERE/'figures';out.mkdir(exist_ok=True)
stem='confirmation_quality' if args.split=='confirm' and args.metric=='submitted_prefix' else f'{args.split}_{args.metric}_quality'
if args.refined_control:stem+='_position_control'
for ext in ['pdf','svg','png']:fig.savefig(out/f'{stem}.{ext}',dpi=220,bbox_inches='tight')
with (out/f'{stem}.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['suite','method','accuracy']);writer.writeheader();writer.writerows(export)
sources=['extension_comparisons.json']+(['live_summary.json','metric_robustness.json','position_selected.json'] if args.refined_control else [])
(out/f'{stem}.sources.json').write_text(json.dumps({name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in sources},indent=2)+'\n')
print(out/f'{stem}.pdf')
