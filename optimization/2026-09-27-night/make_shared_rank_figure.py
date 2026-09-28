"""Fixed-setting rank control, with source-backed means and paired intervals."""
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE


def main():
    live=json.loads((HERE/'live_summary.json').read_text())
    suites=[('heldout_known','LOTO20',20),('novel','Novel15',15),('compound','Compound10',10)]
    ranks=[0,1,2,4]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(10.4,5.7),sharex=True,sharey=True)
    rows=[];sources={}
    for i,split in enumerate(['confirm','fresh']):
        stem='shared_rank_'+split
        assert (HERE/(stem+'_done.json')).exists()
        source=HERE/(stem+'.jsonl')
        sources[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
        for j,(suite,label,n) in enumerate(suites):
            cell=live[stem]['suites'][suite]
            assert cell['n_complete_tasks']==n
            ax=axes[i,j]
            for suffix,name,color,style in [('', 'Contextual CASS','#007F5F','-'),
                                          ('_off','Correction off','#777777','--')]:
                values=[cell['means'][f'rank{r}{suffix}'] for r in ranks]
                ax.plot(ranks,values,style,marker='o',color=color,label=name,linewidth=1.6,markersize=4)
                for rank,value in zip(ranks,values):
                    rows.append(dict(split=split,suite=suite,rank=rank,method=name,accuracy=value))
            delta=cell['rank0__rank1'];lo,hi=delta['ci95']
            annotation=f'Rank 1 − rank 0: {-100*delta["diff"]:+.2f} pp\n95% CI [{-100*hi:+.2f}, {-100*lo:+.2f}]'
            ax.text(.04,.97,annotation,transform=ax.transAxes,va='top',fontsize=8)
            ax.set_title(label,loc='left',fontsize=10)
            ax.set_ylim(0,.86);ax.set_xlim(-.15,4.15)
            ax.set_xticks(ranks);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if j==0:ax.set_ylabel(('Original queries' if i==0 else 'Additional queries')+'\nAccuracy')
            if i==1:ax.set_xlabel('Shared directions removed per layer')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False)
    fig.text(.5,.012,'Frozen gain, schedule and contrast; all ranks retained. Prefix metric; three seeds; task-bootstrap paired intervals.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.075,1,1),h_pad=1.6)
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'shared_rank_control.{ext}',dpi=220,bbox_inches='tight')
    with (out/'shared_rank_control.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sources['live_summary.json']=hashlib.sha256((HERE/'live_summary.json').read_bytes()).hexdigest()
    (out/'shared_rank_control.sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(out/'shared_rank_control.pdf')


if __name__=='__main__':main()
