"""Export a module-level diagram of the submitted CASS information flow."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(12.8,6.2))
    ax.set_xlim(0,13);ax.set_ylim(0,6.3);ax.axis('off')
    blue='#17668D';green='#007F5F';orange='#B56B00';gray='#4B5563'
    def box(x,y,w,h,text,fill='white',edge=gray,fs=10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.07',
                                   facecolor=fill,edgecolor=edge,linewidth=1.15))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color='#17202A',linespacing=1.45)
    def arrow(a,b,color=gray,style='-',connection='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
            linewidth=1.25,color=color,linestyle=style,connectionstyle=connection))
    for x,w,title,note in [(0.15,3.55,'1. Offline mining','Once per model and skill library'),
                           (4.15,3.75,'2. New-task adaptation','Once per task; at most four examples'),
                           (8.35,4.45,'3. Query serving','Reuse the cached task operator')]:
        ax.add_patch(FancyBboxPatch((x,1.45),w,4.55,boxstyle='round,pad=0.07,rounding_size=0.08',
                                   facecolor='#F7F9FB',edgecolor='#CDD4DA',linewidth=.85))
        ax.text(x+.15,5.72,title,fontsize=12,fontweight='bold',va='center')
        ax.text(x+.15,5.39,note,fontsize=8.7,color=gray,va='center')
    box(.48,4.61,2.87,.49,'Known-task activation pairs',fs=9.7)
    box(.48,3.58,2.87,.62,'Remove shared rank-one\ncomponent '+r'$U_0$',fill='#EAF3F8',edge=blue)
    box(.48,2.31,2.87,.72,'Low-rank skill bases '+r'$U_t$'+'\nand anchors '+r'$\mu_t$',fill='#E8F3EF',edge=green)
    arrow((1.92,4.54),(1.92,4.27));arrow((1.92,3.51),(1.92,3.10))
    ax.text(1.92,1.78,'Language-model weights remain frozen',fontsize=8.7,ha='center',color=gray)
    box(4.47,4.61,3.10,.49,'New task demonstrations',fs=9.7)
    box(4.47,3.58,3.10,.62,'Contrastive extraction\n'+r'de-shared signature $z$',fill='#EAF3F8',edge=blue)
    box(4.47,2.31,3.10,.72,'Weighted group LASSO\n'+r'support $S$ and coefficients $c_t$',fill='#E8F3EF',edge=green)
    arrow((6.02,4.54),(6.02,4.27));arrow((6.02,3.51),(6.02,3.10),blue)
    arrow((3.43,3.89),(4.39,3.89),blue)
    ax.text(3.91,4.02,r'$U_0$',ha='center',fontsize=9,color=blue)
    arrow((3.43,2.67),(4.39,2.67),green)
    ax.text(3.91,2.84,'library',ha='center',fontsize=8.5,color=green)
    ax.text(6.02,1.91,r'Signals: $\|z\|$ and reconstruction residual',ha='center',fontsize=8.9,color=gray)
    ax.text(6.02,1.64,'Reconstruction is a diagnostic of library coverage',ha='center',fontsize=8.3,color=gray)
    box(8.68,4.61,3.79,.49,'Query hidden state '+r'$h$',fs=9.7)
    box(8.68,3.43,3.79,.90,'Demonstration direction\n+ gated geometric correction\n'+r'gate $g$ uses $z$ and the anchor',fill='#EAF3F8',edge=blue,fs=9.5)
    box(8.68,2.31,3.79,.65,'Selected bases and weighted anchors\n'+r'projector $P_S$, reference $\mu_S$',fill='#E8F3EF',edge=green,fs=9.5)
    arrow((10.58,4.54),(10.58,4.40))
    arrow((7.65,3.89),(8.60,3.89),blue)
    ax.text(8.13,4.08,r'$z$',ha='center',fontsize=10,color=blue)
    arrow((7.65,2.67),(8.60,2.67),green)
    ax.text(8.13,2.88,r'$S,c_t$',ha='center',fontsize=9,color=green)
    arrow((10.58,3.03),(10.58,3.36),green)
    ax.text(10.58,1.91,'Greedy decoding from the intervened state',ha='center',fontsize=9,color=gray)
    ax.text(10.58,1.64,'No demonstration tokens in the steering query',ha='center',fontsize=8.7,color=blue)
    box(.48,.26,12.00,.73,
        'Separately evaluated serving policies\n'
        r'Weak $\|z\|$  $\rightarrow$  prompt-state replacement'
        '       |       Large residual  '+r'$\rightarrow$'+'  full ICL with demonstrations',
        fill='#FFF7E8',edge=orange,fs=10)
    arrow((6.02,1.39),(6.02,1.07),orange,'--')
    out=Path(__file__).resolve().parent/'figures';out.mkdir(exist_ok=True)
    fig.tight_layout(pad=.4)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/('submitted_method_overview.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(out/'submitted_method_overview.pdf')


if __name__=='__main__':main()
