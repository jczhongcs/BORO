from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(os.environ.get('BORO_FIGURE_OUTPUT', str(ROOT/'results/figures')))
OUT.mkdir(parents=True, exist_ok=True)
ade=pd.read_csv(ROOT/'data/source/ade/ablation/means.csv').set_index('variant')
ade_runs=pd.read_csv(ROOT/'data/source/ade/ablation/per_seed_results.csv').rename(columns={'版本标识':'variant','种子':'seed'})
seed_metrics={'r_at_p95_mean':'R@P95','r_at_p99_mean':'R@P99','auprc_mean':'AP','best_f1_mean':'Best F1'}
prsm=pd.read_csv(ROOT/'data/source/prsm/ablation_per_seed.csv').set_index('variant')
order=['full_boro','uniform_weights','no_boundary_local_selection','hard_example_matched']
selected=['full_boro','uniform_weights','no_boundary_local_selection','hard_example_matched']
labels={'full_boro': 'Full BORO', 'uniform_weights': 'Uniform boundary weights', 'no_boundary_local_selection': 'Global hard-example mining', 'hard_example_matched': 'Loss-scale-matched hard mining'}
assert ade.loc[order,'n_seeds'].eq(5).all()
assert ade_runs.groupby('variant').seed.nunique().loc[order].eq(5).all()
seeds=sorted(ade_runs.seed.unique())
assert len(seeds)==5 and not ade_runs.duplicated(['variant','seed']).any()
for metric,raw_metric in seed_metrics.items():
    assert np.allclose(ade_runs.groupby('variant')[raw_metric].mean().loc[order],ade.loc[order,metric],rtol=0,atol=1e-12),metric
assert prsm.loc[selected,'seed'].eq(1729).all()
plt.rcdefaults()
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.5,'axes.titlesize':7.2,'xtick.labelsize':6.5,'ytick.labelsize':6.9,'xtick.major.size':2.5,'ytick.major.size':0,'text.color':'#263448','axes.labelcolor':'#42516A','xtick.color':'#758297','ytick.color':'#263448','xtick.major.width':.5,'ytick.major.width':.5,'axes.edgecolor':'#C5CED8','savefig.facecolor':'white'})
fig=plt.figure(figsize=(183/25.4,110/25.4),facecolor='white')
left=.335;right=.975
positions=[(.335,.175),(.545,.175),(.76,.09),(.89,.085)]
fullcolor='#364C74';neutral='#BCC8D6'
metrics=[('r_at_p95_mean','Recall at\nprecision ≥95%',(0,.16),[0,.08,.16]),('r_at_p99_mean','Recall at\nprecision ≥99%',(0,.12),[0,.06,.12]),('auprc_mean','Average precision',(0,.75),[0,.35,.7]),('best_f1_mean','Best F1',(0,.7),[0,.3,.6])]
def style(ax, variants, show_labels=True):
    ax.set_ylim(len(variants)-.65,-.65)
    ax.set_yticks(range(len(variants)),[labels[v] for v in variants] if show_labels else [])
    ax.tick_params(axis='y',pad=7)
    for name in ['top','right','left']:ax.spines[name].set_visible(False)
    ax.spines['bottom'].set_linewidth(.5)
    for tick,v in zip(ax.get_yticklabels(),variants):
        if v=='full_boro':tick.set_color(fullcolor);tick.set_fontweight('bold')
for i,(col,title,lim,ticks) in enumerate(metrics):
    pos,w=positions[i]
    ax=fig.add_axes([pos,.55,w,.29])
    style(ax,order,i==0);ax.set_xlim(*lim);ax.set_xticks(ticks);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.set_title(title,pad=11,loc='center',fontweight='bold' if i<2 else 'normal',color=fullcolor if i<2 else '#51627C',fontsize=7.2)
    if i<2:ax.set_facecolor('#F5F7FA')
    for j,v in enumerate(order):
        val=float(ade.loc[v,col])
        ax.barh(j,val,height=.46,color=(fullcolor if i<2 else '#8393A8') if v=='full_boro' else ('#BCC8D6' if i<2 else '#DCE1E7'),zorder=3)
        vals=ade_runs.loc[ade_runs.variant.eq(v)].set_index('seed').loc[seeds,seed_metrics[col]].to_numpy(float)
        assert len(vals)==5 and np.isfinite(vals).all() and (vals>=lim[0]).all() and (vals<=lim[1]).all()
        ax.text(val,j-.38,f'{val:.3f}',ha='right',va='center',fontsize=6.5,color=fullcolor if v=='full_boro' else '#33435C')
fig.text(.035,.95,'a',fontsize=10,fontweight='bold',va='center',color=fullcolor)
fig.text(left,.95,'ADE detection',fontsize=9.5,fontweight='bold',va='center')
ax=fig.add_axes([left,.105,right-left,.245])
style(ax,selected);ax.set_xlim(0,42);ax.set_xticks([0,10,20,30,40])
for j,v in enumerate(selected):
    x=float(prsm.loc[v,'additional'])
    ax.barh(j,x,height=.48,color=fullcolor if v=='full_boro' else neutral,zorder=3)
    ax.text(x+.65,j,f'{x:.2f}',va='center',fontsize=7.5,color=fullcolor if v=='full_boro' else '#42516A',fontweight='normal')
ax.set_xlabel('Mean additional target PrSMs per dataset at 1% FDR',labelpad=9,fontsize=7.5)
fig.text(.035,.425,'b',fontsize=10,fontweight='bold',va='center',color=fullcolor)
fig.text(left,.425,'PrSM rescoring',fontsize=9.5,fontweight='bold',va='center')
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t in fig.findobj(matplotlib.text.Text):
    if t.get_visible() and t.get_text():
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,t.get_text()
stem=OUT/'ed_boundary_controls'
for ext in ['png','pdf','svg','tiff']:
    kwargs={'dpi':600 if ext=='tiff' else 260}
    if ext=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
    fig.savefig(str(stem)+'.'+ext,**kwargs)
print('Saved combined figure')
