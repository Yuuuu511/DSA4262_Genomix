"""Assemble five verified panels at 178 mm width; no raw input or analysis writes.

Run: python assemble_report_figures.py --output outputs
Only report_fig1_global_patterns and report_fig2_differential_candidates are saved.
"""
from pathlib import Path
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter, MultipleLocator
from matplotlib import font_manager
import numpy as np
import pandas as pd

from task2_pipeline import KEY, get_palette, nice_limits, build_differential_pair

CELLS = ('A549', 'H9', 'HEYA8', 'Hct116', 'HepG2', 'K562', 'MCF7')
TRANSCRIPT = 'ENST00000299238'
SHARING = [4059, 1978, 1165, 845, 659, 510, 527]
WIDTH_MM = 178


def style():
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
        family = 'Arial'
    except ValueError:
        family = 'DejaVu Sans'
    mpl.rcParams.update({
        'font.family': family, 'font.size': 7.5, 'axes.labelsize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
        'axes.linewidth': .65, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'text.color': '#202124', 'axes.labelcolor': '#202124',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.bbox': None,
    })
    return family


def heading(fig, x, letter, title, subtitle):
    fig.text(x, .96, letter, fontsize=9.5, weight='bold', va='top')
    fig.text(x+.025, .955, title, fontsize=8, va='top')
    fig.text(x, .855 if '\n' in title else .885, subtitle, fontsize=6.7, va='top', color='#444444')


def export(fig, figures, stem, trim_bottom_mm=0):
    # Fixed canvas: no tight cropping, so LaTeX scaling retains the stated font sizes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    # Axes tight boxes consider only ticks actually drawn within view limits;
    # log locators also create out-of-range tick objects which are not rendered.
    for artist in list(fig.axes)+list(fig.texts)+list(fig.legends):
        box = artist.get_tightbbox(renderer)
        if box is not None and box.width and box.height:
            assert box.x0 >= -1 and box.y0 >= trim_bottom_mm/25.4*fig.dpi-1 and box.x1 <= canvas.x1+1 and box.y1 <= canvas.y1+1, f'Clipped artist: {artist}'
    bounds = None
    if trim_bottom_mm:
        # Trim empty canvas only; preserve every artist's physical size/spacing.
        width, height = fig.get_size_inches()
        bounds = mpl.transforms.Bbox.from_bounds(0, trim_bottom_mm/25.4,
                                                 width, height-trim_bottom_mm/25.4)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(figures/f'{stem}.{suffix}', dpi=600, bbox_inches=bounds,
                    metadata={'Creator': 'Task 2 report figure assembly'} if suffix != 'png' else None)
    plt.close(fig)


def global_patterns(tables, figures):
    pairwise = pd.read_csv(tables/'replicate_pairwise.csv')
    cross = pd.read_csv(tables/'cross_cell_pairwise.csv')
    distribution = pd.read_csv(tables/'extended_analysis/a_sharing_distribution.csv')
    baseline = distribution[distribution.threshold.eq(.5)].set_index('n_positive_cells')
    assert baseline.loc[range(1,8), 'n_sites'].tolist() == SHARING
    assert baseline.n_sites.sum() == 440569 and baseline.loc[0, 'n_sites'] == 430826
    assert len(pairwise) == 34 and pairwise.pearson.notna().all()
    assert len(cross) == 21 and not cross.duplicated(['cell_a','cell_b']).any()
    palette = get_palette(CELLS)

    fig = plt.figure(figsize=(WIDTH_MM/25.4, 82/25.4))
    a = fig.add_axes([.075,.265,.24,.55])
    b = fig.add_axes([.385,.235,.32,.59])
    c = fig.add_axes([.785,.265,.205,.55])
    heading(fig,.06,'A','Within-cell prediction agreement','Shared sites within each run pair')
    heading(fig,.39,'B','Cross-cell score similarity','Shared eligible sites (≥2 runs per cell)')
    heading(fig,.755,'C','Sharing of predicted-\npositive sites','Positive cutoff = 0.5')

    for i, cell in enumerate(CELLS):
        values = pairwise.loc[pairwise.cell_line.eq(cell),'pearson'].to_numpy()
        points = a.scatter(np.full(len(values),i),values,s=13,alpha=.5,
                           color=palette[cell],edgecolors='none')
        np.testing.assert_array_equal(np.asarray(points.get_offsets())[:,1],values)
    a.set(xlim=(-.5,6.5),ylim=nice_limits(pairwise.pearson,min_span=.05),
          ylabel='Pearson correlation',xlabel='Cell line',xticks=range(7),xticklabels=CELLS)
    a.tick_params(axis='x',rotation=50,pad=2)
    for label in a.get_xticklabels(): label.set_ha('right')
    a.yaxis.set_major_locator(MultipleLocator(.04))

    matrix = pd.DataFrame(np.nan,index=CELLS,columns=CELLS)
    for row in cross.itertuples():
        matrix.loc[row.cell_a,row.cell_b] = row.pearson
        matrix.loc[row.cell_b,row.cell_a] = row.pearson
    lo, hi = nice_limits(matrix.stack(),min_span=.04)
    cmap = mpl.colormaps['Blues'].copy(); cmap.set_bad('#F0F1F3')
    image = b.pcolormesh(np.arange(7),np.arange(7),np.ma.masked_invalid(matrix.to_numpy()),
                        cmap=cmap,vmin=lo,vmax=hi,shading='nearest',rasterized=False)
    b.invert_yaxis()
    b.set_aspect('equal')
    for i in range(7):
        for j in range(7):
            value = matrix.iloc[i,j]
            if i == j:
                b.text(j,i,'N/A',ha='center',va='center',fontsize=6,color='#73777D')
            else:
                rgba = cmap(image.norm(value))
                luminance = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                b.text(j,i,f'{value:.3f}',ha='center',va='center',fontsize=6.5,
                       color='white' if luminance<.53 else '#17202A')
    b.set(xticks=range(7),yticks=range(7),xticklabels=CELLS,yticklabels=CELLS)
    b.tick_params(length=0,pad=3)
    b.tick_params(axis='x',rotation=50)
    for label in b.get_xticklabels(): label.set_ha('right')
    b.set_xticks(np.arange(-.5,7),minor=True); b.set_yticks(np.arange(-.5,7),minor=True)
    b.grid(which='minor',color='white',linewidth=.5); b.tick_params(which='minor',length=0)
    for spine in b.spines.values(): spine.set_visible(False)
    cbax=fig.add_axes([.43,.078,.23,.022])
    cb=fig.colorbar(image,cax=cbax,orientation='horizontal',ticks=[.74,.76,.78])
    cb.solids.set_rasterized(False)
    cb.set_label('Pearson correlation',fontsize=7,labelpad=2)
    cb.ax.tick_params(labelsize=6.7,pad=1,length=2)

    bars=c.bar(range(1,8),SHARING,color='#4477AA',width=.72)
    for bar,n in zip(bars,SHARING):
        c.text(bar.get_x()+bar.get_width()/2,n+85,f'{n:,}',ha='center',fontsize=6.4)
        assert bar.get_height() == n
    c.set(xlabel='Number of positive cell lines',ylabel='Predicted-positive sites',
          xticks=range(1,8),ylim=(0,4700),xlim=(.35,7.65))
    c.yaxis.set_major_locator(MultipleLocator(1000))
    c.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    fig.text(.075,.075,'Each dot represents one run pair.',fontsize=6.6)
    fig.text(.755,.125,'440,569 sites; ≥2 runs/cell in all seven.\nOnly sites positive in ≥1 cell shown;\n430,826 all-negative sites omitted.',
             fontsize=6.5,linespacing=1.25,va='top')
    export(fig,figures,'report_fig1_global_patterns')
    return pairwise, cross


def differential_candidates(tables, figures):
    # Existing cached consensus join only; never call the source figure function,
    # which also writes scientific tables. Do not read any raw prediction CSVs.
    d, selected, strict = build_differential_pair(tables,'Hct116','K562',2,.3)
    assert (len(d),len(selected),len(strict)) == (810786,2736,2230)
    for name, expected in [('selected',selected),('strict',strict)]:
        saved=pd.read_csv(tables/f'differential_{name}_Hct116_vs_K562.csv')
        assert set(map(tuple,saved[KEY].to_numpy())) == set(map(tuple,expected[KEY].to_numpy()))
    x=np.maximum(d.max_score_Hct116-d.min_score_Hct116,d.max_score_K562-d.min_score_K562)
    dd=d[d.transcript_id.eq(TRANSCRIPT)].sort_values('transcript_position')
    chosen=selected[selected.transcript_id.eq(TRANSCRIPT)]
    assert len(chosen)==15 and chosen.replicate_supported.sum()==13

    fig=plt.figure(figsize=(WIDTH_MM/25.4,85/25.4))
    a=fig.add_axes([.075,.275,.405,.55])
    cbax=fig.add_axes([.49,.275,.012,.55])
    b=fig.add_axes([.625,.275,.36,.55])
    heading(fig,.06,'A','Between-cell differences and run variability','810,786 matched sites; ≥2 runs per cell')
    heading(fig,.61,'B','Example differential transcript',TRANSCRIPT)
    hb=a.hexbin(x,d.abs_delta_mean,gridsize=65,mincnt=1,bins='log',cmap='Greys',
                linewidths=0,rasterized=True)
    np.testing.assert_equal(int(hb.get_array().sum()),len(d))
    for supported,color,n in [(False,'#EE8866',506),(True,'#AA3377',2230)]:
        mask=d.selected & d.replicate_supported.eq(supported)
        assert mask.sum()==n
        a.scatter(x[mask],d.loc[mask,'abs_delta_mean'],s=2.5,c=color,alpha=.5,
                  edgecolors='none',rasterized=True)
    a.axhline(.3,color='#666666',ls=':',lw=.8)
    a.plot([0,1],[0,1],color='#777777',ls='--',lw=.8)
    a.set(xlim=(0,1),ylim=(0,1),xlabel='Maximum within-cell observed score range',
          ylabel='Absolute consensus-score difference, D')
    cb=fig.colorbar(hb,cax=cbax)
    cb.solids.set_rasterized(False)
    cb.set_label('Sites per hexagon (log scale)',fontsize=7,labelpad=3)
    cb.ax.tick_params(labelsize=6.5,pad=2)
    cb.ax.minorticks_off()
    handles=[Line2D([],[],marker='o',linestyle='none',color='#AA3377',markersize=3,label='Strict separation (2,230)'),
             Line2D([],[],marker='o',linestyle='none',color='#EE8866',markersize=3,label='Overlapping ranges (506)'),
             Line2D([],[],color='#666666',ls=':',lw=.8,label='D = 0.30 cutoff'),
             Line2D([],[],color='#777777',ls='--',lw=.8,label='D = maximum range')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.065,.165),
               frameon=False,ncol=2,fontsize=6.7,handlelength=1.4,columnspacing=1,
               handletextpad=.5,labelspacing=.6)

    for cell,color in [('Hct116','#4477AA'),('K562','#EE8866')]:
        points=b.scatter(dd.transcript_position,dd[f'mean_score_{cell}'],s=10,
                         color=color,alpha=.75,zorder=2,label=cell)
        np.testing.assert_array_equal(np.asarray(points.get_offsets())[:,1],dd[f'mean_score_{cell}'])
        rings=b.scatter(chosen.transcript_position,chosen[f'mean_score_{cell}'],s=27,
                        facecolors='none',edgecolors=color,linewidths=.85,zorder=4)
        assert len(rings.get_offsets())==15
    b.axhline(.5,color='#777777',lw=.8,ls='--')
    b.set(xlabel='Transcript position',ylabel='Mean prediction score',ylim=(0,1))
    b.xaxis.set_major_locator(MultipleLocator(500))
    h=[Line2D([],[],marker='o',ls='none',color='#4477AA',markersize=3,label='Hct116'),
       Line2D([],[],marker='o',ls='none',color='#EE8866',markersize=3,label='K562'),
       Line2D([],[],marker='o',ls='none',color='#555555',markerfacecolor='none',markersize=4,label='Selected site'),
       Line2D([],[],color='#777777',ls='--',lw=.8,label='Score = 0.50')]
    fig.legend(handles=h,loc='upper left',bbox_to_anchor=(.615,.165),ncol=2,frameon=False,
               fontsize=6.7,handlelength=1.4,columnspacing=1.5,labelspacing=.6)
    export(fig,figures,'report_fig2_differential_candidates',trim_bottom_mm=4)
    print(f'Figure 2: {len(d):,} matched; {len(selected):,} selected; {len(strict):,} strict; transcript {len(dd)} matched positions, {len(chosen)} selected.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs'),help='Existing output root')
    args=parser.parse_args()
    family=style()
    tables=args.output/'tables'; figures=args.output/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    pairwise,cross=global_patterns(tables,figures)
    differential_candidates(tables,figures)
    print(f'Figure 1: {len(pairwise)} run pairs; {len(cross)} cross-cell pairs; sharing counts {SHARING}. Font: {family}. Width: {WIDTH_MM} mm.')


if __name__=='__main__':
    main()
