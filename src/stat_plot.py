#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>
#
# MIT License
#
# Copyright (c) 2021 Milan Ondrašovič
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numbers

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _grouped_boxplots(data_groups, ax=None, max_width=0.8, pad=0.05, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    max_group_size = max(len(item) for item in data_groups)
    total_padding = pad * (max_group_size - 1)
    width = (max_width - total_padding) / max_group_size
    # kwargs['widths'] = width
    kwargs['widths'] = 0.13
    
    def positions(group, i):
        span = width * len(group) + pad * (len(group) - 1)
        ends = (span - width) / 2
        x = np.linspace(-ends, ends, len(group))
        return x + i
    
    artists = []
    for i, group in enumerate(data_groups, start=1):
        artist = ax.boxplot(group.T, positions=positions(group, i), **kwargs)
        artists.append(artist)
    
    ax.margins(0.05)
    ax.set(xticks=np.arange(len(data_groups)) + 1)
    ax.autoscale()
    
    return artists


def _plot_improvement_over_baseline_boxes(
        ax, data_groups, labels, x_label_suffix="", ax_y_lim=None
):
    assert data_groups.shape[1] == len(labels)
    
    baseline_artist = ax.axhline(
        0, label="baseline", alpha=0.8, linestyle='--',
        linewidth=1, aa=True
    )
    for sep_pos in (np.arange(1, len(data_groups)) + 0.5):
        ax.axvline(sep_pos, alpha=0.5, linewidth=0.5, c='black', aa=True)
    
    meanlineprops = dict(linestyle='-', linewidth=2, color='red')
    groups = _grouped_boxplots(
        data_groups, ax, max_width=0.6, patch_artist=True, notch=False,
        meanline=True, showmeans=True, showfliers=False,
        meanprops=meanlineprops, whis=0.5
    )
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for item in groups:
        assert len(item['boxes']) <= len(colors)
        
        for color, patch, median_line in zip(
                colors, item['boxes'], item['medians']
        ):
            patch.set(facecolor=color, alpha=0.8)
            plt.setp(median_line, color='black')
    
    x_tick_labels = list(map(str, range(1, len(data_groups) + 1)))
    
    def minus_percent_formatter_(y, pos):
        text = f"{abs(y):.0%}"
        if y < 0:
            text = u'\u2212' + text
        return text
    
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(minus_percent_formatter_))
    
    if ax_y_lim:
        ax.set_ylim(ax_y_lim)
        ax.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1])
    
    ax.set(
        xlabel="Used $k$-th best homography" + x_label_suffix,
        ylabel="Relative improvement",
        axisbelow=True, xticklabels=x_tick_labels)
    proxy_artists = groups[-1]['boxes']
    ax.legend(
        proxy_artists + [baseline_artist], labels + ['baseline'],
        loc='lower left', prop={'size': 8}
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def plot_improvement_over_baseline_boxes(
        data_groups, labels, x_label_suffix="", ax_y_lim=None
):
    fig, ax = plt.subplots(figsize=(5, 3))
    _plot_improvement_over_baseline_boxes(
        ax, data_groups, labels, x_label_suffix, ax_y_lim
    )
    fig.tight_layout()
    
    return fig


def _add_stats_plot(data, ax):
    min_val, max_val, mean, std_dev, median = (
        np.min(data), np.max(data), np.mean(data), np.std(data),
        np.median(data)
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    text_str = '\n'.join((
        rf"$\mathrm{{pixels\ no.}} = {len(data.flatten())}$",
        rf"$\mathrm{{min}} = {min_val:.3f}$",
        rf"$\mathrm{{max}} ={max_val:.3f}$",
        rf"$\mu ={mean:.3f}$",
        rf"$\sigma = {std_dev:.3f}$",
        rf"$\mathrm{{median}} = {median:.3f}$")
    )
    
    ax.text(
        0.02, 0.97, text_str, transform=ax.transAxes, fontdict=None,
        verticalalignment='top', bbox=props
    )


def plot_heat_map(
        error_grid, boxes=None, color=0.5, colorbar_range=None,
        x_label_suffix=""
):
    if isinstance(color, numbers.Number):
        color = [color]
    
    if len(color) == 1:
        color = color * len(boxes)
    else:
        if len(color) != len(boxes):
            raise ValueError("no. of colors needs to match the no. of boxes")
    
    fig, ax = plt.subplots()
    
    ax.set_xlabel("Image width" + x_label_suffix)
    ax.set_ylabel("Image height")
    
    image = ax.imshow(error_grid, cmap='gist_yarg')
    xs = np.arange(0, error_grid.shape[1])
    ys = np.arange(0, error_grid.shape[0])
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, error_grid, linewidths=1, antialiased=True, color='black')
    if colorbar_range is not None:
        image.set_clim(*colorbar_range)
    cbar = fig.colorbar(image)
    cbar.ax.set_ylabel("Reprojection error (pixel-wise $L_2$ norm)")
    
    if boxes is not None:
        draw_params = dict(closed=True, linewidth=3, antialiased=True)
        for box, curr_color in zip(boxes, color):
            polygon = patches.Polygon(box, color=curr_color, **draw_params)
            ax.add_patch(polygon)
    
    _add_stats_plot(error_grid, ax)
    
    fig.tight_layout()
    
    return fig


def plot_error_vals_hist(ax, scenar_error_vals, x_label=None, y_label=None):
    for error_vals in scenar_error_vals:
        ax.hist(error_vals, alpha=0.6, bins=100, histtype='step', linewidth=1.4)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set(xlabel=x_label, ylabel=y_label)
