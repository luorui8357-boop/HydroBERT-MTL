"""Draw metrics and absolute-error boxes from actual, supplied predictions only."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import regression

def plot(predictions, output):
    frame = pd.read_csv(predictions)
    required = {"Record_ID", "Model", "Mask_rate", "Y_true", "Prediction"}
    if not required <= set(frame):
        raise ValueError(f"Missing prediction columns: {required-set(frame)}")
    # A repeated-CV file needs a declared aggregation policy instead of duplicate rows.
    if frame.duplicated(["Record_ID", "Model", "Mask_rate"]).any():
        raise ValueError("Duplicate predictions; select a single repetition before plotting")
    ids = [set(z.Record_ID) for _, z in frame.groupby(["Model", "Mask_rate"])]
    if not all(i == ids[0] for i in ids):
        raise ValueError("Every model and masking rate must evaluate identical test records")
    if frame.groupby("Record_ID").Y_true.nunique().max() != 1:
        raise ValueError("Target values differ across models")
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family":"Arial", "font.weight":"bold", "font.size":15,
                         "axes.labelweight":"bold", "axes.labelsize":17, "axes.titleweight":"bold"})
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=.43, wspace=.28)
    a, b, c = fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1]), fig.add_subplot(grid[1,:])
    rates = sorted(frame.Mask_rate.unique())
    models = list(frame.Model.unique())
    colors = plt.get_cmap("tab10").colors
    palette = {"HydroBERT-MTL":"#e60012","Ridge":"#4f78bd","RF":"#b67a19","XGBoost":"#239b8d",
               "FT-Transformer":"#ff7f0e","TabPFN":"#9467bd","TabR":"#17becf","CatBoost":"#7c8e4b"}
    rows = []
    width = .82 / len(models)
    for j, model in enumerate(models):
        color = palette.get(model, colors[j % len(colors)])
        scores = []
        errors = []
        for rate in rates:
            z = frame[frame.Model.eq(model) & frame.Mask_rate.eq(rate)]
            s = regression(z.Y_true, z.Prediction)
            scores.append(s)
            errors.append(abs(z.Y_true.to_numpy()-z.Prediction.to_numpy()))
            rows.append(dict(Model=model, Mask_rate=rate, **s))
        a.plot(np.array(rates)*100, [s["R2"] for s in scores], "o-", color=color, label=model, linewidth=2)
        b.plot(np.array(rates)*100, [s["RMSE"] for s in scores], "o-", color=color, linewidth=2)
        positions = np.arange(len(rates)) + (j-(len(models)-1)/2)*width
        boxes = c.boxplot(errors, positions=positions, widths=width*.85, patch_artist=True, showfliers=False,
                          manage_ticks=False)
        for box in boxes["boxes"]:
            box.set(facecolor=color, alpha=.8)
    for ax in (a,b):
        ax.set_xticks(np.array(rates)*100, [f"{r*100:g}%" for r in rates])
        ax.set_xlabel("Structural Data Missing Rate (%)")
    a.set_ylabel("R²")
    b.set_ylabel("RMSE (wt.%)")
    c.set_xticks(np.arange(len(rates)), [f"{r*100:g}%" for r in rates])
    c.set_xlabel("Structural Data Missing Rate (%)")
    c.set_ylabel("|Error| (wt.%)")
    for ax, label in zip((a,b,c), ("(a)","(b)","(c)")):
        ax.text(-.065 if ax is c else -.12, 1.05, label, transform=ax.transAxes, fontsize=19, fontweight="bold")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
    fig.subplots_adjust(top=.86, bottom=.09, left=.08, right=.98)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(dest/f"masking_comparison.{suffix}", dpi=300)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(dest/"plotted_metrics.csv", index=False)

def plot_repetitions(frame, output):
    """Fig. S3: one real repetition per dot; error bars are sample SD."""
    dest=Path(output)
    models=frame.Model.unique()
    if len(models)!=1:
        raise ValueError("Plot repeated CV for one model at a time")
    rates=sorted(frame.Mask_rate.unique())
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    rng=np.random.default_rng(42)
    colors=["#75b9d2","#efad79"]
    for ax,metric in zip(axes,["R2","RMSE"]):
        for i,rate in enumerate(rates):
            values=frame.loc[frame.Mask_rate.eq(rate),metric].to_numpy(float)
            if len(values)<2:
                raise ValueError("At least two repetitions required")
            ax.scatter(i+rng.uniform(-.12,.12,len(values)),values,s=17,alpha=.65,color=colors[i%2])
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='o',color='black',capsize=5)
        ax.set_xticks(range(len(rates)),["Complete inputs" if r==0 else f"{100*r:g}% structural masking" for r in rates])
        ax.set_ylabel("R²" if metric=="R2" else "RMSE (wt%)")
    for suffix in ('png','pdf','svg'):
        fig.savefig(dest/f"Fig_S3_repeated_cv.{suffix}",dpi=300)
    plt.close(fig)

def plot_lopo(detail, output):
    """Fig. S4: all publications retained, including extreme negative R2."""
    dest=Path(output)
    desired=["HydroBERT-MTL","Ridge","RF","XGBoost","FT-Transformer","TabPFN","TabR","CatBoost"]
    models=[m for m in desired if m in set(detail.Model)]+sorted(set(detail.Model)-set(desired))
    rates=sorted(detail.Mask_rate.unique())
    if rates != [0.,1.]:
        raise ValueError("Fig. S4 requires complete inputs and 100% masking only")
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained',sharey='row')
    rng=np.random.default_rng(42)
    for col,rate in enumerate(rates):
        for row,metric in enumerate(['R2','RMSE']):
            ax=axes[row,col]
            values=[detail.loc[detail.Model.eq(m)&detail.Mask_rate.eq(rate),metric].to_numpy() for m in models]
            boxes=ax.boxplot(values,patch_artist=True,showfliers=False)
            for i,(box,v) in enumerate(zip(boxes['boxes'],values)):
                color='#d43d51' if models[i]=='HydroBERT-MTL' else plt.get_cmap('tab10')(i%10)
                box.set(facecolor=color,alpha=.25)
                ax.scatter(i+1+rng.uniform(-.15,.15,len(v)),v,color=color,s=15,alpha=.8)
            ax.set_xticks(range(1,len(models)+1),models,rotation=45,ha='right')
            ax.set_ylabel('R²' if metric=='R2' else 'Material-balanced RMSE (wt%)')
            if row==0:ax.set_yscale('symlog',linthresh=1)
            ax.set_title(('Complete inputs' if rate==0 else '100% structural masking'))
    for suffix in ('png','pdf','svg'):
        fig.savefig(dest/f"Fig_S4_lopo.{suffix}",dpi=300)
    plt.close(fig)
