"""
David Clark
December 2019
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy.special import xlogy
from scipy.ndimage import gaussian_filter1d
import data_util
import matplotlib.pyplot as plt
import style

from imp import reload
reload(style)

def get_p_str(p_val, asterisks=True):
    if p_val >= .05:
        return "n.s."
    e = 0
    min_e = -3
    while p_val < 10**(e-1) and e > min_e:
        e -= 1
    if e == -1:
        if asterisks:
            p_str = "*"
        else:
            p_str = "$p$ < 0.05"
    else:
        if asterisks:
            p_str = "*" * (-e)
        else:
            p_str = "$p$ < " + str(10**e)
    return p_str

def add_pval(ax, pval, **kwargs):
    ax.text(0.025, 0.025, get_p_str(pval), size=style.pval_fontsize,
        transform=ax.transAxes, ha="left", va="bottom", **kwargs)

def prettyify_ax(ax):
    for pos in ("right", "top"):
        ax.spines[pos].set_visible(False)
    for pos in ("left", "bottom"):
        ax.spines[pos].set_linewidth(style.spine_linewidth)
    ax.tick_params(axis="both", labelsize=style.ticklabel_fontsize,
        pad=style.ticklabel_pad, width=style.tick_linewidth)

def plot_linregress_result(ax, linregress_result, xlim, ylim=None, **kwargs):
    slope, intcpt = linregress_result[:2]
    x = np.linspace(xlim[0], xlim[1], 1000)
    y = intcpt+slope*x
    if ylim is not None:
        yrange = ylim[1] - ylim[0]
        ymid = (ylim[0] + ylim[1]) / 2.
        yrange_shrunk = yrange * .975
        ylim_shrunk = (ymid - yrange_shrunk/2., ymid + yrange_shrunk/2.)
        mask = (y >= ylim_shrunk[0]) & (y <= ylim_shrunk[1])
        x = x[mask]
        y = y[mask]
    ax.plot(x, y, **kwargs)

def compute_ax_lim(vals, scale=1.05, symmetric=True):
    if symmetric:
        max_abs_val = np.max(np.abs(vals))
        x1, x2 = -max_abs_val, max_abs_val
    else:
        x1 = np.min(vals)
        x2 = np.max(vals)
    mid = (x1 + x2) / 2.
    tot = x2 - x1
    x1 = mid - tot*scale/2.
    x2 = mid + tot*scale/2.
    return (x1, x2)

def plot_scatter(x, y, linregress_result, xlabel=None, ylabel=None, title=None,
    xlim=None, ylim=None,  ax=None, figsize=(2.5, .85),
    scatter_color="gray", line_color="black"):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=200)
    #Exclude nan's.
    not_nan_mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[not_nan_mask], y[not_nan_mask]
    #Get ax lims.
    if xlim is None: xlim = compute_ax_lim(x)
    if ylim is None: ylim = compute_ax_lim(y)
    #Plot stuff.
    ax.scatter(x, y, marker=".", color=scatter_color, lw=0, s=style.scatter_markersize)
    draw_line(ax, xlim, linregress_result, c=line_color, lw=style.line_linewidth, ylim=ylim)
    #Pretty-ify ax.
    prettyify_ax(ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, size=style.title_fontsize, pad=style.title_pad)
    ax.set_xlabel(xlabel, size=style.label_fontsize, labelpad=style.label_pad)
    ax.set_ylabel(ylabel, size=style.label_fontsize, labelpad=style.label_pad)
    return ax

def plot_two_scatters(x1, y1, linregress_result_1, x2, y2, linregress_result_2,
    xlabel, ylabel, ax1_title, ax2_title, xlim=None, ylim=None, lw=None,
    figsize=(2.5, .85)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=200)
    fig.subplots_adjust(wspace=.2)
    ax1, ax2 = axes

    #Exclude nan's.
    not_nan_mask_1 = (~np.isnan(x1)) & (~np.isnan(y1))
    not_nan_mask_2 = (~np.isnan(x2)) & (~np.isnan(y2))
    x1, y1 = x1[not_nan_mask_1], y1[not_nan_mask_1]
    x2, y2 = x2[not_nan_mask_2], y2[not_nan_mask_2]

    #Get ax lims.
    if xlim is None: xlim = compute_ax_lim(np.concatenate((x1, x2)))
    if ylim is None: ylim = compute_ax_lim(np.concatenate((y1, y2)))

    #Plot stuff.
    ax1.scatter(x1, y1, marker=".", color=style.day_0_color_light, lw=0,
        s=style.scatter_markersize)
    ax2.scatter(x2, y2, marker=".", color=style.day_4_color_light, lw=0,
        s=style.scatter_markersize)
    draw_line(ax1, xlim, linregress_result_1, c=style.day_0_color, lw=lw)
    draw_line(ax2, xlim, linregress_result_2, c=style.day_4_color, lw=lw)

    #Pretty-ify axes.
    for ax in axes:
        prettyify_ax(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax1.set_title(ax1_title, size=style.title_fontsize, pad=style.title_pad)
    ax2.set_title(ax2_title, size=style.title_fontsize, pad=style.title_pad)
    ax1.set_ylabel(ylabel, size=style.label_fontsize)
    ax1.set_xlabel(xlabel, size=style.label_fontsize)
    ax2.set_xlabel(xlabel, size=style.label_fontsize)

    return axes

def plot_lines_with_errorbars(x, y, yerr, xlabel=None, ylabel=None, title=None,
    figsize=(1.5, 1.), colors=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    if y.ndim == 1:
        y = y[np.newaxis, :]
        yerr = yerr[np.newaxis, :]
    N_lines = len(y)
    if colors is None:
        colors = ["black" for _ in range(N_lines)]
    for i in range(N_lines):
        ax.errorbar(x=x, y=y[i], yerr=yerr[i], lw=style.line_linewidth,
            capsize=style.errorbar_capsize, capthick=style.errorbar_capthick,
            color=colors[i])
    prettyify_ax(ax)
    if title is not None:
        ax.set_title(title, size=style.title_fontsize,pad=style.title_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel, size=style.label_fontsize)
    if ylabel is not None:
            ax.set_ylabel(ylabel, size=style.label_fontsize)
    return ax

def plot_lines_with_errorbars_grid(x, y, yerr, xlabel, ylabels, colors,
    figsize=(1.25, 2.), xlim=None, ylims=None, xticks=None):
    N_plots = len(y)
    fig, axes = plt.subplots(N_plots, 1, figsize=figsize, dpi=200)
    fig.subplots_adjust(hspace=.4)
    for i in range(N_plots):
        ax = axes[i]
        xlabel_for_ax = xlabel if i == N_plots-1 else None
        plot_lines_with_errorbars(x, y[i], yerr[i], xlabel_for_ax, ylabels[i],
            colors=colors, ax=ax)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylims is not None:
            ax.set_ylim(ylims[i])
            #ax.set_ylim(ylims[i])
        if xticks is not None:
            ax.set_xticks(xticks)
    fig.align_ylabels(axes)
    return axes

def plot_mean_and_ind_lines(x, y, xlabel, ylabel, title=None,
    figsize=(1.5, 1.)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    ax.plot(x, y.T, c="gray", lw=style.line_linewidth / 2)
    ax.plot(x, y.mean(axis=0), c="black", lw=style.line_linewidth, marker=".",
        ms=style.line_markersize)
    prettyify_ax(ax)
    if title is not None:
        ax.set_title(title, size=style.title_fontsize,pad=style.title_pad)
    ax.set_ylabel(ylabel, size=style.label_fontsize)
    ax.set_xlabel(xlabel, size=style.label_fontsize)
    return ax

def plot_mean_and_ind_lines_grid(x, y, titles, xlabel, ylabels, figsize=(6, 1),
    xticks=None, xlim=None, hspace=.2):
    #shape(y) = (N_vars, N_days, N_mouse, N_pos)
    N_rows, N_cols = y.shape[0], y.shape[1]
    fig, axes = plt.subplots(N_rows, N_cols, figsize=figsize, dpi=200)
    fig.subplots_adjust(hspace=hspace)
    for col_idx in range(N_cols):
        for row_idx in range(N_rows):
            ax = axes[row_idx, col_idx]
            ax.plot(x, y[row_idx, col_idx].T, c="gray",
                lw=style.line_linewidth/2.)
            ax.plot(x, y[row_idx, col_idx].mean(axis=0), c="black",
                lw=style.line_linewidth)
            ax.set_ylim([0, np.max(y[row_idx])*1.05])
            if xlim is not None: ax.set_xlim(xlim)
            if xticks is not None: ax.set_xticks(xticks)
            if row_idx == 0:
                ax.set_title(titles[col_idx], size=style.title_fontsize,
                    pad=style.title_pad)
            elif row_idx == N_rows - 1:
                ax.set_xlabel(xlabel, size=style.label_fontsize,
                    labelpad=style.label_pad)
            if col_idx == 0:
                ax.set_ylabel(ylabels[row_idx], size=style.label_fontsize,
                    labelpad=style.label_pad)
            else:
                ax.set_yticks([])
            prettyify_ax(ax)
    fig.align_ylabels(axes[:, 0])
    return axes
     
def log_hist(ax, vals, xlim, bins):
    if np.sum(vals == 0) > 0:
        print("Warning: zeros in log-histogram")
    logbins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), bins)
    ax.hist(vals, bins=logbins, color="0.2")
    ax.set_xscale("log")

def plot_hists(figsize, vals, xlim, xticks, ylim, yticks, titles, bins, ylabel,
    vlines=None, xlabel=None):
    N_plots = len(vals)
    fig, axes = plt.subplots(1, N_plots, figsize=figsize, dpi=200)
    fig.subplots_adjust(wspace=.4)
    for i in range(N_plots):
        ax = axes[i]
        log_hist(ax, vals[i], xlim, bins)
        #ax.hist(vals[i], bins=bins, range=xlim, color="0.2")
        ax.set_title(titles[i], size=style.title_fontsize, pad=style.title_pad)
        if xlabel is not None:
            ax.set_xlabel(xlabel, size=style.label_fontsize,
                pad=style.label_pad)
        if vlines is not None:
            ax.axvline(vlines[i], c="red", lw=style.line_linewidth,
                color=style.day_4_color, linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(xticks)
        if i == 0:
            ax.set_yticks(yticks)
            ax.set_ylabel(ylabel, labelpad=style.label_pad,
                size=style.label_fontsize)
        else:
            ax.set_yticks([])
        prettyify_ax(ax)
        #Reduce the x ticklabel size...
        ax.tick_params(axis="x",
            labelsize=style.ticklabel_fontsize * .9)
    return axes


def plot_pc_shapes(figsize, x, place_fields, envelope):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    ax.plot(x, place_fields.T, lw=style.line_linewidth, color="black")
    ax.plot(x, envelope, lw=style.line_linewidth, color=style.pretty_blue, linestyle="-")
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 1.1])
    ax.set_yticks([0, 1])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("Distance (m)", size=style.label_fontsize, labelpad=style.label_pad)
    ax.set_ylabel("model PC activity", size=style.label_fontsize, labelpad=style.label_pad)
    prettyify_ax(ax)
    return ax


def plot_pc_width(figsize, x, width, ylim, yticks):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    ax.plot(x, width, lw=style.line_linewidth, color=style.pretty_blue)
    ax.set_xlim([0, 4])
    ax.set_ylim(ylim)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks(yticks)
    ax.set_xlabel("Distance (m)", size=style.label_fontsize, labelpad=style.label_pad)
    ax.set_ylabel("model PC width (m)", size=style.label_fontsize, labelpad=style.label_pad)
    prettyify_ax(ax)
    return ax













