"""Source-backed scaling figure with all four update/solver combinations."""
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE


def main():
    sources={};rows=[]
    for method,stem in [('Submitted','scale_eval'),('Contextual','scale_context')]:
        for fixed in [False,True]:
            name=stem+('_fixed_shared' if fixed else '')
            assert (HERE/(name+'_done.json')).exists(),name
            path=HERE/(name+'_analysis.json');data=json.loads(path.read_text())
            sources[name]=dict(analysis_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               prediction_sha256=data['source_sha256'])
            for row in data['scaling']:
                rows.append(dict(method=method,shared='Frozen' if fixed else 'Recomputed',
                                 **{k:v for k,v in row.items() if not isinstance(v,(dict,list))}))
    out=HERE/'figures';out.mkdir(exist_ok=True)
    with (out/'dictionary_scaling.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(12,6.4),sharex=True)
    colors={'Frozen':'#1764a1','Recomputed':'#be4b3a'}
    fields=['accuracy','support_jaccard_to32','coding_median_seconds']
    titles=['Execution accuracy (25 task means)','Support Jaccard to 32 skills','Online coding time (seconds)']
    handles=[];labels=[]
    for i,method in enumerate(['Submitted','Contextual']):
        for j,(field,title) in enumerate(zip(fields,titles)):
            ax=axes[i,j]
            for shared in ['Frozen','Recomputed']:
                for mode in ['full','shortlist32']:
                    selected=sorted([r for r in rows if r['method']==method and r['shared']==shared and r['mode']==mode],key=lambda r:r['size'])
                    label=f"{shared} shared direction / {'full coding' if mode=='full' else 'shortlist 32'}"
                    line,=ax.plot([r['size'] for r in selected],[r[field] for r in selected],
                                  color=colors[shared],linestyle='-' if mode=='full' else '--',
                                  marker='o' if mode=='full' else 's',markersize=4,label=label)
                    if i==0 and j==0:handles.append(line);labels.append(label)
            ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256],labels=['32','64','128','256'])
            ax.grid(alpha=.18)
            if field=='accuracy':ax.set_ylim(0,.8);ax.set_ylabel(method)
            elif field=='support_jaccard_to32':ax.set_ylim(0,1.05)
            else:ax.set_ylim(bottom=0)
            if i==0:ax.set_title(title)
            if i==1:ax.set_xlabel('Dictionary entries')
    fig.suptitle('Correlated dictionary expansion: execution, support and coding cost',fontsize=13,y=.985)
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.04),frameon=False)
    fig.text(.5,.006,'32 original skills + up to 224 synthetic skills; Novel15 + Compound10; three demonstration seeds.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.15,1,.95])
    for ext in ['pdf','svg','png']:fig.savefig(out/f'dictionary_scaling.{ext}',dpi=180)
    (out/'dictionary_scaling_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(out/'dictionary_scaling.png')


if __name__=='__main__':main()
