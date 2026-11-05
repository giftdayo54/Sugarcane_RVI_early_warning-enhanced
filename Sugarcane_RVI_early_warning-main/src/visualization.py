from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.express as px

def save_risk_png(risk_tif,out_png,title='Sugarcane anomaly risk'):
    with rasterio.open(risk_tif) as src: a=src.read(1,masked=True)
    cmap=ListedColormap(['green','yellow','orange','red','darkred']); norm=BoundaryNorm([-0.5,.5,1.5,2.5,3.5,4.5],5)
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(a,cmap=cmap,norm=norm); ax.set_title(title); ax.axis('off')
    cbar=fig.colorbar(im,ax=ax,ticks=range(5)); cbar.ax.set_yticklabels(['Normal','Watch','Moderate','High','Critical'])
    fig.tight_layout(); fig.savefig(out_png,dpi=200,bbox_inches='tight'); return fig

def growth_curve(df,block_id):
    x=df[df.block_id==block_id].sort_values('crop_age_days')
    fig=px.line(x,x='crop_age_days',y=['observed_rvi','expected_rvi'],markers=True,title=f'RVI trajectory: {block_id}')
    if 'expected_std' in x:
        fig.add_scatter(x=x.crop_age_days,y=x.expected_rvi+2*x.expected_std,line=dict(width=0),showlegend=False)
        fig.add_scatter(x=x.crop_age_days,y=x.expected_rvi-2*x.expected_std,fill='tonexty',line=dict(width=0),name='95% envelope')
    return fig
