from pathlib import Path
import json, hashlib, shutil, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score

parser=argparse.ArgumentParser(); parser.add_argument('--repo',type=Path); parser.add_argument('--output',type=Path); args=parser.parse_args()
ROOT=args.output or Path(__file__).resolve().parent
ROOT.mkdir(parents=True,exist_ok=True)
REPO=Path(r'D:\Codex_output\PathGuard任务化评估与反馈闭环\onsite-ai-construction')
REPO=args.repo or REPO
OUT=ROOT/'figures'; OUT.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.titlesize':13,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.1,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'legend.frameon':False})
C=['#0F4D92','#42949E','#B64342','#9A4D8E']
sources={}; captions=[]
def read(rel):
 p=REPO/rel; sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf-8-sig')) if p.suffix=='.json' else pd.read_csv(p)
def save(fig,name,title,caption,placement):
 fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
 for ext in ['svg','pdf']: fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',facecolor='white')
 plt.close(fig); captions.append(dict(name=name,title=title,caption=caption,placement=placement))
def grid(ax): ax.grid(axis='y',alpha=.17); ax.set_axisbelow(True)

df=read('reports/final/tables/engineering_heldout_scores.csv'); y=1-df.true_label.to_numpy()
assert len(df)==66 and y.sum()==39
methods=[('model_score','学习模型'),('frozen_geometry_rule_score','固定工程规则'),('minimum_clearance_score','最小余量基线')]
fig,axs=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
for (col,label),color,style in zip(methods,C,['-','--',':']):
 s=df[col]; fpr,tpr,_=roc_curve(y,s); pr,rec,_=precision_recall_curve(y,s)
 axs[0].plot(fpr,tpr,label=f'{label}  AUC={roc_auc_score(y,s):.3f}',color=color,ls=style,lw=2)
 axs[1].step(rec,pr,where='post',label=f'{label}  AP={average_precision_score(y,s):.3f}',color=color,ls=style,lw=2)
axs[0].plot([0,1],[0,1],color='#999999',ls=':',label='随机参考')
axs[1].axhline(y.mean(),color='#999999',ls=':',label='失败占比 39/66')
for ax,title,xlab,ylab in zip(axs,['A  跨地图留出 ROC','B  跨地图留出精确率—召回率'],['假阳性率','召回率'],['真阳性率','精确率']):
 ax.set(title=title,xlabel=xlab,ylabel=ylab,xlim=(0,1),ylim=(0,1.03)); ax.legend(fontsize=9,loc='lower right' if ax==axs[0] else 'lower left');grid(ax)
save(fig,'01_heldout_curves','历史留出排序性能', '66条样本、5张留出地图、39条失败；来自已保存的历史分值。AP采用average_precision_score。曲线为既有结果重绘，不是新增独立验证。','第8章 方法比较')

fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained'); K=np.arange(1,67)
for (col,label),color,style in zip(methods,C,['-','--',':']):
 order=df.sort_values([col,'sample_id'],ascending=[False,True]); captured=np.cumsum(1-order.true_label.to_numpy())
 axs[0].step(K,captured,where='post',color=color,ls=style,lw=2,label=label)
 axs[1].plot(K,captured/K,color=color,ls=style,lw=2,label=label)
axs[0].plot(K,K*39/66,color='#888888',ls=':',label='随机选择期望');axs[1].axhline(39/66,color='#888888',ls=':')
axs[0].scatter([10,20,34],[8,17,24],color=C[0],zorder=5)
for k,v in [(10,8),(20,17),(34,24)]: axs[0].annotate(f'({k}, {v})',(k,v),xytext=(4,8),textcoords='offset points',fontsize=9)
axs[0].set(title='A  验证预算与失败发现数',xlabel='验证名额 K',ylabel='累计发现失败数',ylim=(0,41));axs[0].legend(fontsize=9)
axs[1].set(title='B  选中路线的失败比例',xlabel='验证名额 K',ylabel='Precision@K',ylim=(0,1.04))
for ax in axs:grid(ax)
save(fig,'02_budget_capture','有限验证预算的收益曲线','同分按sample_id升序，匹配既有审计口径。蓝色标记为Top-10/20/34。全局留出排序不等于同一规划任务内的批次收益；随机线为解析期望而非一次随机试验。','第8章 Top-K价值解释')

ab=read('reports/final/tables/feature_group_ablation.csv'); mc=read('reports/final/tables/model_comparison.csv')
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
labels=['HGB','Extra Trees','逻辑回归']; vals=mc.cv_pr_auc_failure_mean.to_numpy();std=mc.cv_pr_auc_failure_std.to_numpy()
axs[0].errorbar(vals,np.arange(3),xerr=std,fmt='o',color=C[0],capsize=5,lw=1.7)
axs[0].set(yticks=np.arange(3),yticklabels=labels,xlim=(0,1),xlabel='开发集 AP 均值 ± 折间标准差',title='A  地图分组五折模型比较');axs[0].invert_yaxis();axs[0].grid(axis='x',alpha=.17)
names=['移除路线几何','移除坡度','移除速度与动力代理','移除转向','移除余量','移除局部窗口'];delta=ab.delta_vs_all.iloc[1:].to_numpy()
axs[1].barh(names,delta,color=[C[1] if v>0 else C[2] for v in delta],edgecolor='white');axs[1].axvline(0,color='#333',lw=1);axs[1].invert_yaxis()
for i,v in enumerate(delta):axs[1].text(v+(0.0005 if v>=0 else -0.0005),i,f'{v:+.4f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
axs[1].set(xlim=(-.015,.034),xlabel='移除该组后的 AP 变化',title='B  特征组消融（基准 AP=0.6605）')
save(fig,'03_model_ablation','模型选择与特征组消融','误差线是5折之间的标准差，不是95%置信区间。消融重新训练模型，正值表示移除该组后均值提高；本结果不支持所有特征组均有独立增益的说法。','第8章 消融实验')

cov=read('analysis/boundary_geometry/boundary_coverage_by_structure.csv')
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
axs[0].bar(['五轴','六轴','结构未定'],cov.samples,color=[C[0],C[1],'#A4ADB5'],edgecolor='white')
for i,v in enumerate(cov.samples):axs[0].text(i,v+3,str(v),ha='center')
axs[0].set(title='A  冻结样本结构',ylabel='样本数',ylim=(0,185));grid(axs[0])
stages=['可读取路线','完整左右边界','距离语义通过','完整点列筛查通过']; counts=[340,69,26,3]
axs[1].barh(stages,counts,color=[C[0],C[1],'#7FBFC5','#AADCA9']);axs[1].invert_yaxis()
for i,v in enumerate(counts):axs[1].text(v+4,i,str(v),va='center')
axs[1].set(title='B  空间分析证据逐级筛查',xlabel='满足该条件的样本数',xlim=(0,385))
read('reports/final/boundary_integrity_reaudit_summary.json')
save(fig,'04_evidence_coverage','冻结样本与空间证据覆盖','340条冻结样本可用于路线展示；空间分析需要额外边界条件。69、26、3是嵌套的筛查层次，不能相加；完整点列筛查仍不是连续碰撞检测证明。','第4章 数据基础 / 技术附录')

cases=read('demo/data/tracking_execution_evidence.json')['cases'];cases=[c for c in cases if 'series' in c]
fig,axs=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
for row,c in enumerate(cases[:2]):
 s=pd.DataFrame(c['series']);name='山路运行' if c['map']=='mounarea_1' else '坡道运行'
 axs[row,0].plot(s.time_s,s.lateral_m,color=C[0],lw=1.5);axs[row,1].plot(s.time_s,s.heading_deg,color=C[1],lw=1.5)
 for col,unit in enumerate(['横向偏差 / m','航向偏差 / °']):
  axs[row,col].axhline(0,color='#888',ls=':',lw=1);axs[row,col].set(title=f'{"ABCD"[row*2+col]}  {name}',xlabel='运行时间 / s',ylabel=unit);grid(axs[row,col])
save(fig,'05_tracking_traces','计划轨迹与实际执行的偏差实录','来自两次已核对平台运行的展示序列，曲线经过展示采样，不将采样点当独立试验；全文统计以原始4974/3228帧核查结果为准。平台运行不等同现场实车。','第6章 执行机制 / 第9章 案例')

L=np.linspace(8,24,161);deg=np.linspace(0,10,101);ll,dd=np.meshgrid(L,deg);w=3.74
extra=ll/2*np.sin(np.deg2rad(dd))+w/2*np.cos(np.deg2rad(dd))-w/2
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
im=axs[0].pcolormesh(L,deg,extra,cmap='YlGnBu',shading='auto');fig.colorbar(im,ax=axs[0],label='额外横向占用 / m')
cs=axs[0].contour(L,deg,extra,levels=[.25,.5,1,1.5],colors='white',linewidths=.9);axs[0].clabel(cs,fmt='%.2f',fontsize=9)
axs[0].set(title='A  长度与航向偏差敏感性',xlabel='假设车长 / m',ylabel='假设航向偏差 / °')
for length,col in zip([10,14,18.8,22],C):
 e=length/2*np.sin(np.deg2rad(deg))+w/2*np.cos(np.deg2rad(deg))-w/2
 axs[1].plot(deg,e,color=col,label=f'L={length:g} m',lw=2)
axs[1].set(title='B  固定车长的响应截面',xlabel='假设航向偏差 / °',ylabel='额外横向占用 / m');axs[1].legend();grid(axs[1])
save(fig,'06_geometry_sensitivity','长车身为何放大航向偏差','解析机理图：刚性矩形、参考点居中、宽3.74 m、横向偏差为0。额外占用=L/2·sinθ+W/2·cosθ−W/2；不含轮胎、载荷或控制动力学，不是道路净空、学习模型风险或真实试验结果。','第6章 空间机理')

fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for label,marker,color in [(0,'o',C[1]),(1,'X',C[2])]:
 mask=y==label;axs[0].scatter(df.model_score[mask],df.frozen_geometry_rule_score[mask],label='历史通过' if label==0 else '历史失败',marker=marker,color=color,s=48,alpha=.7)
axs[0].set(title='A  学习排序与规则提示的分歧',xlabel='模型失败排序分值',ylabel='固定规则应力分值',xlim=(0,1),ylim=(0,1));axs[0].legend()
maps=sorted(df.map_id.unique()); matrix=np.array([[int(((df.map_id==m)&(y==label)).sum()) for m in maps] for label in [0,1]])
axs[1].imshow(matrix,cmap='Blues',aspect='auto',vmin=0,vmax=matrix.max())
for (i,j),v in np.ndenumerate(matrix):axs[1].text(j,i,str(v),ha='center',va='center',color='white' if v>matrix.max()/2 else '#172B3A',fontsize=14)
axs[1].set(xticks=range(len(maps)),xticklabels=[m.replace('map_','地图 ') for m in maps],yticks=[0,1],yticklabels=['历史通过','历史失败'],title='B  留出地图的标签构成')
save(fig,'07_score_disagreement','双通道分歧与地图构成','每点对应一条历史留出记录，分值相同的点可能重叠。两个轴的分值语义不同，不能据点到对角线距离判断谁正确；右图解释当前评价受到地图构成影响。','第5章 双通道解释 / 第8章 验证背景')

(ROOT/'provenance.json').write_text(json.dumps({'sources_sha256':sources,'figures':captions,'scope':'Existing results redrawn; no retraining; geometry is an analytic illustration.'},ensure_ascii=False,indent=2),encoding='utf-8')
md=['# PathGuard 科研图集与使用建议','共7幅组合图、16个子图。PNG为300 DPI，PDF/SVG适合印刷及编辑。','建议正文优先使用图1、2、5、6；图3、4、7放技术章节或附录，避免用图表数量替代论证。']
html=['<!doctype html><meta charset="utf-8"><title>PathGuard 科研图集</title><style>body{max-width:1150px;margin:40px auto;font:17px/1.7 sans-serif;color:#173047;background:#f4f7fa}article{background:white;padding:28px;margin:25px 0;border-radius:12px}img{width:100%}p{color:#526170}h1{font-size:32px}</style><h1>PathGuard 科研图集</h1><p>历史实验结果 · 执行证据 · 空间机理</p>']
for c in captions:
 md += [f'\n## {c["title"]}',f'建议位置：{c["placement"]}',c['caption'],f'![{c["title"]}](figures/{c["name"]}.png)']
 html += [f'<article><h2>{c["title"]}</h2><img src="figures/{c["name"]}.png"><p>{c["caption"]}</p><small>建议位置：{c["placement"]} · <a href="figures/{c["name"]}.svg">SVG</a> · <a href="figures/{c["name"]}.pdf">PDF</a></small></article>']
(ROOT/'图集说明与图注.md').write_text('\n\n'.join(md),encoding='utf-8');(ROOT/'科研图集预览.html').write_text('\n'.join(html),encoding='utf-8')
print('Created',len(captions),'figures')

