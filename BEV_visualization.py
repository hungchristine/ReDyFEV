# coding: utf-8

# # Mapping of production and consumption mixes in Europe and their effect on carbon footprint of electric vehicles

# This code plots the following:
#     -  CF of production vs consumption mixes of each country with visual indication of relative contribution to total European electricity production] [Figure_1]
#     -  imports, exports and net trade [extra figure]
#     -  trade matrix heatmap [extra figure]

#     -  change in BEV CF with domestic production [Figure 4, horizontal and vertical formats]
#     -  CF of BEVs for 2 size segments and production and consumption mixes] [#Figure 2]
#     -  mitigation potential of BEVs for 4 size segments] [#Figure 3]
#     -  ratio of BEV:ICEV lifecycle impacts [# Extra figure]

# - Requires the following files to run:
#      - country-specific indirect_el.pkl (from BEV_footprints_calculation.py)
#      - country-specific indirect_BEV.pkl (from BEV_footprints_calculation.py)
#      - country-specific indirect_BEV_impacts_consumption.pkl (from BEV_footprints_calculation.py)
#      - country-specific indirect_ICEV_impacts.pkl (from BEV_footprints_calculation.py)
#      - label_pos.csv

# %% Import packages

import os
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import country_converter as coco

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from matplotlib import cm
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import AutoMinorLocator, FixedLocator

import geopandas as gpd
import seaborn as sns
# from palettable.cubehelix import Cubehelix
from cmcrameri import cm


from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


# %%  Set up functions to run for visualizations for experiments
fp = os.path.curdir
fp_data = os.path.join(fp, 'data')
fp_output = os.path.join(fp, 'output')
fp_results = os.path.join(fp, 'results')
fp_figures = os.path.join(fp_results, 'figures')


def visualize(experiment, export_figures=True, include_TD_losses=True, plot_ei=False, plot_el=False):
    # this function used for fullyear and subannual period experiments
    fp_figure, CFEL, results, ICEV_total_impacts, mapping_data = setup(experiment)

    ei_countries = CFEL.loc[CFEL['Total production (TWh)'] == 0].index.tolist() # countries whose el mix were obtained from ecoinvent (and therefore have no production data)
    print(ei_countries)
    if plot_ei:
        ei_CFEL = CFEL.loc[ei_countries]
    else:
        ei_CFEL = None

    if plot_el:
        CFEL.drop(index=ei_countries, inplace=True)  # drop ei-only countries regardless due to special treatment in Figure 2 (plotted separately)
        plot_el_figs(experiment, fp_figure, CFEL, include_TD_losses, export_figures, ei_CFEL)

    plot_all(experiment, fp_figure, results, ICEV_total_impacts, mapping_data, ei_countries, plot_ei, export_figures=export_figures)


def country_footprint(experiment, params_country, timestamp, export_figures=True):
    # this funcion used for country footprints experiments
    # timestamp variable is closest ENTSO-E sample period to user query
    fp_figure, CFEL, results, ICEV_total_impacts, mapping_data = setup(experiment)
    country = params_country['country']
    start = params_country['start'] # user query for time period
    segment = params_country['segment']
    plot_country_footprint(experiment, fp_figure, country, segment, start, timestamp, mapping_data, export_figures=True)


def setup(experiment):
    # load required data
    fp_figure = os.path.join(fp_results, 'figures')
    fp_experiment = os.path.join(fp_figure, experiment)

    if not os.path.exists(fp_results):
        os.mkdir(fp_results)
    if not os.path.exists(fp_experiment):
        os.mkdir(fp_experiment)

    # Load all relevant results
    fp_pkl = os.path.join(fp_output,
                          experiment + ' country-specific indirect_el.pkl')
    CFEL = pickle.load(open(fp_pkl, 'rb'))

    fp_pkl = os.path.join(fp_output,
                          experiment + ' country-specific indirect_BEV.pkl')
    results = pd.read_pickle(fp_pkl)

    fp_pkl = os.path.join(fp_output,
                          experiment + ' country-specific indirect_ICEV_impacts.pkl')
    ICEV_total_impacts = pickle.load(open(fp_pkl, 'rb'))

    mapping_data = map_prep(CFEL, results)

    return fp_experiment, CFEL, results, ICEV_total_impacts, mapping_data

def plot_el_figs(exp, fp_figure, CFEL, include_TD_losses, export_figures, ei_CFEL):
    sns.set_style("whitegrid")
    plot_fig2(exp, fp_figure, CFEL, include_TD_losses, export_figures, ei_CFEL)
    plot_el_trade(exp, fp_figure, CFEL, export_figures)  # figure S1 in supplementary information

# def plot_all(exp, fp_figure, CFEL, results, ICEV_total_impacts, mapping_data, plot_ei=False, include_TD_losses=True, export_figures=False):
def plot_all(exp, fp_figure, results, ICEV_total_impacts, mapping_data, ei_countries, plot_ei=False, export_figures=False):

    """ Reproduce figures from article. """
    if not plot_ei:
        mapping_data.loc[mapping_data['ISO_A2'].isin(ei_countries), 'Total production (TWh)':] = np.nan
        ei_countries = None
    sns.set_style('white')
    plot_fig3(exp, fp_figure, mapping_data, export_figures, ei_countries, 2)
    plot_fig4(exp, fp_figure, mapping_data, export_figures, ei_countries, 2)
    plot_fig5a(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures, ei_countries, 2)
    plot_fig5b(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures, ei_countries, 2)

    # # maps with all segments for SI
    plot_fig3(exp, fp_figure, mapping_data, export_figures, ei_countries, 4)
    plot_fig4(exp, fp_figure, mapping_data, export_figures, ei_countries, 4)
    plot_fig5a(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures, ei_countries, 4)
    plot_fig5b(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures, ei_countries, 4)
    sns.set_style("whitegrid")
    plot_fig7(exp, fp_figure, results, export_figures, orientation='horizontal')


# %% Helper class for asymmetric normalizing colormaps

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work their way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # makes symmetric colormap to deal with asymmetric max-min ranges (i.e., where abs(min) != abs(max))
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#%% Helper function
def cmap_map(function, cmap):
    """Apply function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.

    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return colors.LinearSegmentedColormap('colormap',cdict,1024)

# %% Plot Figure 2 - Production vs Consumption electricity

def plot_fig2(exp, fp_figure, CFEL, include_TD_losses, export_figures, ei_CFEL=None):
    """ Set up different options for production vs consumption electricity figure"""
    # Prepare variables for Figure 2
    CFEL_sorted = CFEL.sort_values(by='Production mix intensity')

    # determine net importers and exporters for using colormap; positive if net importer, negative if net exporter
    net_trade = (CFEL_sorted.iloc[:, 1] - CFEL_sorted.iloc[:, 0])
    pct_trade = CFEL_sorted['Trade percentage, gross']*100

    # for plotting with transmission losses
    net_trade_TL = (CFEL_sorted.iloc[:, 1] - CFEL_sorted.iloc[:, 0]).values

    print("number of importers: " + str(sum(net_trade > 0)))
    print("number of exporters: " + str(sum(net_trade < 0)))

    # Finally, plot Figure 2
    cmap = cm.oslo_r
    mark_color = cmap

    # Prep country marker sizes; for now, use same size for all axis types
    size_ind = 'production'
    if size_ind == 'consumption':
        marker_size = (CFEL_sorted.iloc[:, 1] / (CFEL_sorted.iloc[:, 1].sum()) * 1200)**2  # marker size = diameter **2
    elif size_ind == 'production':
        marker_size = (CFEL_sorted.iloc[:, 0] / (CFEL_sorted.iloc[:, 0].sum()) * 1200)**2  # marker size = diameter **2
    else:
        print('invalid value for size_ind')

    if include_TD_losses:
        plot_lim = 1300
    else:
        plot_lim = 1300

    fig2_generator(exp, fp_figure, CFEL_sorted, ei_CFEL, plot_lim, export_figures,
                   'linear', size_ind, marker_size, mark_color, pct_trade, xlim=500, ylim=500)

# %% Figure 2 generator (carbon footprint of electricity production mix vs consumption mix)

def fig2_generator(exp, fp_figure, CFEL_sorted, ei_CFEL, plot_maxlim, export_figures, axis_type, size_ind, marker_size, marker_clr, net_trade=None, xlim=None, ylim=None):
    """ Generate Figure 2; carbon footprint of production vs consumption
    mixes. Can be plotted on linear, log or semilog axes
    """

    # Sort by marker size to have smallest markers drawn last
    marker_size = marker_size.sort_values(ascending=False)
    CFEL_markers = CFEL_sorted.copy()
    CFEL_markers = CFEL_markers.reindex(marker_size.index)
    fig, ax = plt.subplots(1, figsize=(16, 12))

    # Control for input axis types linear, semilog, logs
    if axis_type == "log":
        ax.set_yscale(axis_type)
    elif axis_type == "semilog":
        axis_type = "log"
    ax.set_xscale(axis_type)

    # Plot data
    norm = colors.Normalize(vmax=100)
    plot = ax.scatter(CFEL_markers.iloc[:, 2], CFEL_markers.iloc[:, 3],
                      s=marker_size, alpha=0.5, norm=norm, c=net_trade, cmap=marker_clr, label='_nolegend_')

    ax.scatter(CFEL_markers.iloc[:, 2], CFEL_markers.iloc[:, 3],
               s=2, c='k', alpha=0.9, edgecolor='k', label='_nolegend_')  # Include midpoint in figure
    if ei_CFEL is not None:
        ax.scatter(ei_CFEL.iloc[:, 2], ei_CFEL.iloc[:, 3], s=100, marker='*', label='_nolegend_')

    # Hack to have darker marker edges
    ax.scatter(CFEL_markers.iloc[:, 2], CFEL_markers.iloc[:, 3],
               s=marker_size, alpha=0.7, norm=norm, c="None", edgecolor='k', linewidths=0.7,
               label='_nolegend_')

    ### Configure axis ticks and set minimum limits to 0 (was -60)
    ax.tick_params(which='minor', direction='out', length=4.0)
    ax.tick_params(which='major', direction='out', length=6.0, labelsize=16.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xlim(left=0, right=plot_maxlim)
    ax.set_ylim(bottom=0, top=plot_maxlim)

    ### Size legend
    legend_sizes = [(x / 100 * 1200)**2 for x in [15, 10, 5, 1]]
    legend_labels = ["15%", "10%", "5%", "1%"]

    # The below is for having the legend in the top left position
    ada = AnchoredDrawingArea(300, 300, 0, 0, loc=2, frameon=True)
    if size_ind == 'consumption':
        legend_title = AnchoredText('% of total European consumption',
                                    frameon=False, loc=9, bbox_to_anchor=(0.18, 0.990),
                                    bbox_transform=ax.transAxes,
                                    prop=dict(size=17, weight='medium'))
    elif size_ind == 'production':
        legend_title = AnchoredText('% of total European production',
                                    frameon=False, loc=9, bbox_to_anchor=(0.174, 0.990),
                                    bbox_transform=ax.transAxes,
                                    prop=dict(size=17, weight='medium'))

    add_year = AnchoredText('2020', frameon=False, loc=9, bbox_to_anchor=(0.185, 0.961),
                            bbox_transform=ax.transAxes, prop=dict(size=17, weight='medium'))

    # The below is for having the legend in the bottom right position
    # ada = AnchoredDrawingArea(250,200,0,0,loc=4,frameon=True)
    # legend_title = AnchoredText('% of total European consumption',frameon=False,loc=9, bbox_to_anchor=(0.87,0.3), bbox_transform=ax.transAxes,prop=dict(size=14))
    # add_year= AnchoredText('2011',frameon=False,loc=9, bbox_to_anchor=(0.87,0.27), bbox_transform=ax.transAxes,prop=dict(size=14))

    ada.drawing_area.add_artist(legend_title)
    ada.drawing_area.add_artist(add_year)

    for i, area in enumerate(legend_sizes):
        radius_pts = np.sqrt(area / 3.14)
        c1 = Circle((150, np.sqrt(area / 3.14) + 10), radius_pts,
                    fc = marker_clr(0.5), #'#89BEA3',
                    ec='k', lw=0.6, alpha=0.4)
        c2 = Circle((150, np.sqrt(area / 3.14) + 10), radius_pts,
                    fc="None", ec='k', lw=0.7, alpha=0.7)
        ada.drawing_area.add_artist(c1)
        ada.drawing_area.add_artist(c2)

        leg_ann = plt.annotate(s=legend_labels[i], xy=[0.5, 1], xycoords=c1, xytext=[120, 12],
                               textcoords="offset points", fontsize=13,
                               arrowprops=dict(arrowstyle="->", color="k", lw=1,
                                               connectionstyle="arc,angleA=45,angleB=45,armA=0,armB=20,rad=0"))
        leg_ann.set_zorder(20)

    ax.add_artist(ada)

    ### 10%, 20%, 50% shading and x=y line
    # x=y
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], color="grey", alpha=0.6)
    # 10%
    plt.fill_between([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1] * 1.1],
                     [0, ax.get_xlim()[1] * 0.9], color="grey", edgecolor='k', alpha=0.13, zorder=10)
    # 20%
    plt.fill_between([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1] * 1.2],
                     [0, ax.get_xlim()[1] * 0.8], color="grey",edgecolor='k', alpha=0.1, zorder=9)
    # 50%
    plt.fill_between([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1] * 1.5],
                     [0, ax.get_xlim()[1] * 0.5], color='grey', edgecolor='k', alpha=0.07, zorder=8)
    ax.annotate(r'$\pm$ 10%', xy=(0, 0), xytext=(0.881, 0.95), xycoords='axes fraction', fontsize=13, rotation=40)
    ax.annotate(r'$\pm$ 20%', xy=(0, 0), xytext=(0.808, 0.95), xycoords='axes fraction', fontsize=13, rotation=40)
    ax.annotate(r'$\pm$ 50%', xy=(0, 0), xytext=(0.64, 0.95), xycoords='axes fraction', fontsize=13, rotation=40)
    ax.annotate(r'x = y', xy=(0, 0), xytext=(0.95, 0.96), xycoords='axes fraction', fontsize=13, rotation=40)

    ax.set_xlabel("Carbon footprint of production mix \n (g CO$_2$ kWh$^{-1}$)", fontsize=20, labelpad=14)
    ax.set_ylabel("Carbon footprint of consumption mix \n (g CO$_2$ kWh$^{-1}$)", fontsize=20, labelpad=14)

    ### Make and format inset figure
    # ax2 = fig.add_subplot(339)  # for inlay in bottom right position inset subplot
    # ax2.axis([400, xlim, 400, ylim])

    # Set formatting for zoomed inlay figure

    # markersize = 40, s = 40*2, pi/4*s = marker area
    # Linear factor between main fig and inset: x1/x2 = z*(l1/l2) --> 1100/150 = 33.75/10.85*x --> x=2.357
    # marker_size_ratio = (ax.get_xlim()[1] / (ax2.get_xlim()[1] - ax2.get_xlim()[0]))

    # ax2.scatter(CFEL_markers.iloc[:, 2], CFEL_markers.iloc[:, 3],
    #             s= (np.sqrt(marker_size) * np.sqrt(marker_size_ratio))**2, alpha=0.5,
    #             norm=norm, c=net_trade, cmap=marker_clr, edgecolor='k', label='_nolegend_')
    # ax2.scatter(CFEL_markers.iloc[:, 2], CFEL_markers.iloc[:, 3],
    #             s=(np.sqrt(marker_size) * np.sqrt(marker_size_ratio))**2, alpha=0.9,
    #             norm=norm, c="None", edgecolor='k', linewidths=0.7, label='_nolegend_')  # Hack for darker edge colours
    # ax2.scatter(CFEL_markers.iloc[:, 2], CFEL_markers.iloc[:, 3],
    #             s=2, c='k', alpha=0.9, edgecolor='k', label='_nolegend_')  # Include midpoint in figure

    # ax2.xaxis.tick_top()
    # ax2.yaxis.tick_right()
    # ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
    # ax2.yaxis.set_major_locator(ticker.MultipleLocator(50))
    # ax2.tick_params(which="major", labelsize=12)
    # ax2.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], color="grey", alpha=0.6)
    # plt.fill_between([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1] * 1.1],
    #                  [0, ax2.get_xlim()[1] * 0.9], color="grey", alpha=0.13)
    # plt.fill_between([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1] * 1.2],
    #                  [0, ax2.get_xlim()[1] * 0.8], color="grey", alpha=0.1)
    # plt.fill_between([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1] * 1.5],
    #                  [0, ax2.get_xlim()[1] * 0.5], color='grey', alpha=0.07)

    ### Add country text labels
    for (country, country_data) in CFEL_markers.iterrows():
        # Inset figure labelling
        # if country_data["Production mix intensity"] >= (ax2.get_xlim()[0 ] * 0.9) and country_data["Production mix intensity"] <= (ax2.get_xlim()[1]*1.1) and country_data["Consumption mix intensity"] >= (ax2.get_ylim()[0]*0.9) and country_data["Consumption mix intensity"] <= (ax2.get_ylim()[1]*1.1): #Inset labels
        #     if country in ['DE', 'IT', 'NL']:
        #         pass  # ignore the largest circles in the inset, as they don't need labelling
        #     else:
        #         ax2.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
        #                      xytext=(np.sqrt(marker_size[country]*marker_size_ratio*(np.pi/4))/2+6,-7),
        #                      textcoords=("offset points"), size=15,
        #                      path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)])

        # Left corner of main figure
        if country_data["Production mix intensity"] <= 100 and country_data["Consumption mix intensity"] <= 100:
            # Adjust for countries in bottom left
            if country == "SE":
                ax.annotate(country, xy=(country_data["Production mix intensity"],
                                         country_data["Consumption mix intensity"]),
                            xytext=(np.sqrt(marker_size[country]) / 2 + 2, -2), textcoords=("offset points"),
                            path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)
            elif country == "NO":
                ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                            xytext=(np.sqrt(marker_size[country]) / 2 + 2, -14),
                            textcoords=("offset points"),
                            path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)
            elif country == 'FR':
                ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                            xytext=(np.sqrt(marker_size[country]) / 2 - 2, -38), textcoords=("offset points"),
                            path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)

        # Rest of figure; avoid overlapping labels for LU, MT
        elif country == 'LU':
            ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                        xytext=(np.sqrt(marker_size[country]) / 2 + 5, -4), textcoords=("offset points"),
                        path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)
        elif country == 'MT':
            ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                        xytext=(-25, 2), textcoords=("offset points"),
                        path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)
        else:
            ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                        xytext=(-9.5, -14 - np.sqrt(marker_size[country]) / 2), textcoords=("offset points"),
                        path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)
    if ei_CFEL is not None:
        for (country, country_data) in ei_CFEL.iterrows():
             ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                        xytext=(-9.5, -20), textcoords=("offset points"),
                        path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)

    ### Make colorbar
    ax_cbar = fig.add_axes([0.925, 0.125, 0.03, 0.755])  # place colorbar on own Ax

    ### Calculate minimum and maximum labels for colorbar (rounded to nearest 5 within the scale)
    cbar_min = 0
    cbar_max = 100

    # Begin plotting colorbar
    cbar = plt.colorbar(plot, cax=ax_cbar, drawedges=False, extend='max')
    cbar.set_label('Gross electricity traded, as % of net production', fontsize=18, rotation=90, labelpad=8)

    cbar.set_alpha(1)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(5)
    cbar.outline.set_color('k')
    cbar.draw_all()

    # Manually tack on semi-transparent rectangle on colorbar to match alpha of plot; workaround for weird rendering of colorbar with alpha
    r1 = Rectangle((9, 0), 85, 500, fc='w', alpha=0.3)
    ax_cbar.add_patch(r1)

    if export_figures:
        keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure, 'Fig_2 - ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, 'Fig_2 - ' + keeper + '.png'), bbox_inches='tight')

    plt.show()


# %% Figure 7 (differences for domestic production)
def plot_fig7(exp, fp_figure, results, export_figures, orientation='both'):
    A_diff = (results['BEV footprint, EUR production - Segment A - Consumption mix'] -
              results['BEV footprint - Segment A - Consumption mix']) / results['BEV footprint - Segment A - Consumption mix']
    JE_diff = (results['BEV footprint, EUR production - Segment JE - Consumption mix'] -
              results['BEV footprint - Segment JE - Consumption mix']) / results['BEV footprint - Segment JE - Consumption mix']

    fig_data = pd.DataFrame([A_diff, JE_diff], index=['A segment', 'JE segment'])
    fig_data = fig_data.T

    sorted_fig_data = fig_data.iloc[::-1].sort_values(by='JE segment')
    fig4_cmap = colors.ListedColormap(['xkcd:cadet blue', 'k'])

    if orientation in ('horizontal', 'both'):
        # Horizontal variation of Figure 4
        ax = sorted_fig_data.iloc[::-1].plot(kind='barh', figsize=(16, 12), position=0.5, stacked=True,
                                             cmap=fig4_cmap, width=0.95)
        ax.set_xticklabels(['{:.0%}'.format(x) for x in ax.get_xticks()])
        ax.set_xlabel('Difference in BEV carbon intensity by shifting to domestic battery production', fontsize=16, labelpad=12)

        ax.yaxis.tick_right()
        ax.tick_params(axis='y', direction='out', color='k')
        ax.tick_params(which='major', length=4.0, labelsize=16, pad=10)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(b=True, which='minor', axis='x', linestyle='-', color='gainsboro', linewidth=0.8, alpha=0.9)
        ax.grid(b=True, which='major', axis='x', linewidth=2.5)
        ax.legend(['A segment (36.8 kWh battery)', 'JE segment (95 kWh battery)'], loc=3, facecolor='white', edgecolor='k', fontsize=18, frameon=True, borderpad=1, framealpha=1)

        if export_figures:
            keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
            plt.savefig(os.path.join(fp_figure, 'Fig_7_horizontal ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(fp_figure, 'Fig_7_horizontal ' + keeper + '.png'), bbox_inches='tight')

        plt.show()

    if orientation in ('vertical', 'both'):
        # Vertical variation of Figure 4
        ax = sorted_fig_data.plot(kind='bar', figsize=(16, 12), position=0.5, stacked=True,
                                  cmap=fig4_cmap, alpha=0.7, rot=50, width=0.95)
        ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
        ax.tick_params(axis='x', direction='out', color='k')
        ax.legend(facecolor='white', edgecolor='k', fontsize='large', framealpha=1, frameon=True)
        ax.set_ylabel('Difference in BEV carbon intensity by shifting to domestic battery production', labelpad=12)

        if export_figures:
            keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
            plt.savefig(os.path.join(fp_figure, 'Fig_7_vertical ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(fp_figure, 'Fig_7_vertical ' + keeper), bbox_inches='tight')
    if orientation not in ('horizontal', 'vertical', 'both'):
        raise Exception('Invalid orientation for Figure 4')

    plt.show()

# %% map_prep methods

def map_prep(CFEL, results):
    # Load map shapefile; file from Natural Earth
    fp_map = os.path.join(fp_data, 'maps', 'ne_10m_admin_0_countries.shp')
    country_shapes = gpd.read_file(fp_map)
    europe_shapes = country_shapes[(country_shapes.CONTINENT == "Europe")]  # restrict to Europe
    europe_shapes.loc[:, 'ISO_A2'] = coco.CountryConverter().convert(names=list(europe_shapes['ADM0_A3'].values), to='ISO2')  # convert to ISO A2 format for joining with data

    # Join results to Europe shapefile and drop irrelevant countries (i.e., those without results)
    li = results.columns.tolist()

    # Create nicer header names for calculated results
    col = pd.Index([e[0] + " - Segment " + e[1] + " - " + e[2] for e in li])
    results.columns = col

    # Combined electricity footprint and vehicle footprint results
    all_data = pd.concat([CFEL, results], axis=1)

    # Add to mapping shapes
    mapping_data = europe_shapes.join(all_data, on="ISO_A2")

    return mapping_data

# %%  Helper functions for map plotting

# function to round to nearest given multiple
def round_up_down(number, multiple, direction):
    if direction == "up":
        return int(number + (multiple - (number % multiple)))
    elif direction == "down":
        return int(number - (number % multiple))
    else:
        print("Incorrect direction")

# function for annotating maps

fp_label_pos = os.path.join(fp_data, 'label_pos.csv')
label_pos = pd.read_csv(fp_label_pos, index_col=[0, 1], skiprows=0, header=0)  # read in coordinates for label/annotation positions
lined_countries = ['PT', 'BE', 'NL', 'DK', 'SI', 'GR', 'ME', 'MK', 'MD', 'EE', 'LV', 'BA', 'MT', 'LU', 'AL', 'HR']  # countries using leader lines

def annotate_map(ax, countries, mapping_data, values, cmap_range, threshold=0.8, round_dig=1, fontsize=7.5, ei_countries=None):
    for loc, label in zip(countries, values):

        country = label_pos.loc[loc].index

        if loc in label_pos.index:
            # perform rounding of label values
            if round_dig == 0:
                label = int(label)
            else:
                label = round(label, round_dig)
                if (str(label) == '-0.0') or (str(label) == '0.0'):
                    label = 0

            if (ei_countries is not None) and (country.isin(ei_countries)):
                # special case for ecoinvent proxy countries
                lc = 'k'  # leader line colour
                color = 'darkblue'  # annotation text colour
            elif (threshold < 0):
                # if dark colours are at left of colorbar
                if (label / cmap_range <= (threshold)):
                    if country.isin(lined_countries):
                        lc = 'w'
                        color = 'k'
                    else:
                        lc = 'w'
                        color = 'w'
                else:
                    lc = 'k'
                    color = 'k'
            elif (label / cmap_range) >= threshold:
                # if dark colours are at right of colorbar
                if country.isin(lined_countries):
                    lc = 'w'
                    color = 'k'
                else:
                    color = 'w'
                    lc = 'w'
            else:
                lc = 'k'
                color = 'k'

            x = label_pos.loc[loc].x
            y = label_pos.loc[loc].y
            x_cent = mapping_data.loc[loc].geometry.centroid.x
            y_cent = mapping_data.loc[loc].geometry.centroid.y

            if country.isin(['CH']): # modified start point for pointer line
                ax.annotate(label,
                            xy=(x_cent, y_cent),
                            xytext=(x, y),
                            textcoords='data', size=fontsize-0.8, color=color, va='bottom', ha='center', zorder=10,
                            bbox=dict(pad=0.03, facecolor="none", edgecolor="none"),
                            arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=0.2, shrinkB=0, relpos=(0.9, 0)))
            elif country.isin(['DK']): # modified start point for pointer line
                ax.annotate(label,
                            xy=(x_cent - 0.75, y_cent),
                            xytext=(x, y),
                            textcoords='data', size=fontsize-0.8, color='k', va='baseline', zorder=10,
                            bbox=dict(pad=0.01, facecolor="none", edgecolor="none"),
                            arrowprops=dict(color=color, arrowstyle='-', lw='0.45', shrinkA=0.2, shrinkB=0, relpos=(1, 0.5)))
            elif country.isin(['BE', 'NL']):
                ax.annotate(label,
                            xy=(x_cent, y_cent),
                            xytext=(x, y),
                            textcoords='data', size=fontsize-0.8, color='k', va='baseline', ha='center', zorder=10,
                            bbox=dict(pad=0.02, facecolor="none", edgecolor='none'),
                            arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=-0.2, shrinkB=0, relpos=(0.55, 0)))
            elif country.isin(['MK','SK']):
                if (ei_countries is not None) and (country == 'MK'):
                    x = x + 7.8
                    y = y + 0.5
                    fc = 'w'
                else:
                    fc = 'none'
                ax.annotate(label,
                            xy=(x_cent, y_cent),
                            xytext=(x, y),
                            textcoords='data', size=fontsize-0.8, color='k', va='center', ha='center', zorder=10,
                            bbox=dict(pad=0.03, facecolor=fc, edgecolor='none', alpha=0.75),
                            arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=0.8, shrinkB=0))
            elif country.isin(['PT']):
                ax.annotate(label,
                            xy=(x_cent+0.2, y_cent),
                            xytext=(x, y),
                            textcoords='data', size=fontsize, color='k', va='baseline', ha='center', zorder=10,
                            bbox=dict(pad=0.03, facecolor="none", edgecolor='none'),
                            arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=0.8, shrinkB=0))
            elif country.isin(['GR']): # modified start point for pointer line
                if ei_countries is not None:
                    x = x + 7.5
                    fc = 'w'
                    fs = fontsize
                else:
                    fc = 'none'
                    fs = fontsize-0.7
                ax.annotate(label,
                            xy=(x_cent-1.5, y_cent+0.9),
                            xytext=(x, y),
                            textcoords='data', size=fs, color='k', va='bottom', ha='center', zorder=10,
                            bbox=dict(pad=0.03, facecolor=fc, edgecolor="none", alpha=0.75),
                            arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=1.5, shrinkB=0, zorder=100))
            elif country.isin(['MD']):
                ax.annotate(label,
                            xy=(x_cent, y_cent),
                            xytext=(x, y),
                            textcoords='data', size=fs, color='k', va='bottom', ha='left', zorder=10,
                            bbox=dict(pad=0.03, facecolor=fc, edgecolor="none", alpha=0.75),
                            arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=1.5, shrinkB=0, zorder=100))
            elif country.isin(['LV', 'EE']):
                if ei_countries is not None:
                    relpos = (0, 0.55)
                    if country=='LV':
                        x = 30.1
                        y = 56.8
                elif country=='LV':
                    relpos=(0.5, 1)
                elif country=='EE':
                    relpos = (0, 0.55)
                ax.annotate(label,
                            xy=(x_cent+1.3, y_cent),
                            xytext=(x, y),
                            textcoords='data', size=fontsize, color='k', va='baseline', ha='center', zorder=10,
                            bbox=dict(pad=0.03, facecolor="none",edgecolor='none'),
                            arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=0.5, shrinkB=0, relpos=relpos, zorder=100))
            elif country.isin(['SI', 'ME', 'BA', 'HR']): # countries requiring smaller annotations
                if ei_countries is not None:
                    if country=='BA':
                        x = x-0.5
                        y = y-1.5
                    elif country=='ME':
                        y = y+0.5
                if country=='HR':
                    y_cent = y_cent + 0.6
                ax.annotate(label,
                            xy=(x_cent, y_cent),
                            xytext=(x, y),
                            textcoords='data', size=fontsize-0.8, color='k', va='baseline', ha='center', zorder=10,
                            bbox=dict(boxstyle='round', pad=0.03, facecolor="w", edgecolor="none", alpha=0.75, zorder=8),
                            arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=0.5, shrinkB=0, zorder=100))
            # countries without leader lines requiring smaller annotations
            elif country.isin(['LT','CZ','IE','HU','AT','BG','RS','IT']):
                ax.annotate(label,
                            xy=(x, y),
                            textcoords='data', size=fontsize-0.8, color=color, ha='center', zorder=10)
            # special cases for countries with proxy ecoinvent el-mixes
            elif (ei_countries is not None) and (country.isin(ei_countries)):
                if country.isin(lined_countries):
                    if country == 'LU':
                        fc = 'none'
                        relpos = (0, 0.9)
                    elif country == 'HR':
                        y_cent = y_cent + 0.6
                        relpos = (0.95, 0.95)
                    else:
                        fc = 'w'
                        relpos = (0.5, 1)

                    ax.annotate(str(label) +'$^*$',
                                xy=(x_cent, y_cent),
                                xytext=(x, y),
                                zorder=12,
                                textcoords='data', size=fontsize-1.3, style='italic', color=color, va='baseline', ha='center',
                                bbox=dict(boxstyle='round', pad=0.03, facecolor=fc, edgecolor="none", alpha=0.75),
                                arrowprops=dict(color=lc, arrowstyle='-', ls='-', lw='0.5', shrinkA=0.5, shrinkB=0, relpos=relpos, zorder=10))
                else:
                    ax.annotate(str(label)+ '$^*$',
                            xy=(x_cent, y_cent),
                            xytext=(x, y),
                            zorder=10,
                            textcoords='data', size=fontsize-1.3, style='italic', color=color, va='center', ha='center')
            else:
                if country == 'DE':
                    y = y + 0.4
                    x = x + 0.4
                ax.annotate(label, xy=(x, y),
                            textcoords='data', size=fontsize, color=color, zorder=10, ha='center')
        else:
            print(f'{mapping_data.loc[loc]["ISO_A2"]} has no label coordinates')

# %% Figure 3 - BEV footprints by country

def plot_fig3(exp, fp_figure, mapping_data, export_figures, ei_countries, num_panels):
    # min/max extremes for colorbar
    vmin = 50
    vmax = 375
    mpl.rcParams['hatch.linewidth'] = 0.2  # for plotting ei_countries (optional)

    cmap = colors.ListedColormap(["#c6baca",  # light purple
                                  "#83abce",
                                  "#6eb668",
                                  "#9caa41",
                                  "#815137",
                                  "#681e3e"  # red
                                  ])

    cmap_col = [cmap(i) for i in np.linspace(0, 1, 6)]  # retrieve colormap colors
    cmap = cmap_col

    # Make manual boundaries for cmap
    # range of negative values approximately 1/3 of that of positive values;
    # cut-off colormap for visual 'equality'
    boundaries = [i for i in np.arange(vmin, vmax, 50)]  # define boundaries of colormap transitions
    threshold = boundaries[-2] / boundaries[-1]  # threshold for switching annotation colours
    cmap_BEV, norm = colors.from_levels_and_colors(boundaries, colors=[cmap[0]]+ cmap + [cmap_col[-1]], extend='both')

    max_fp = max(mapping_data['BEV footprint - Segment A - Consumption mix'].max(),
                 mapping_data['BEV footprint - Segment JE - Consumption mix'].max())

    col_list = ['BEV footprint - Segment A - Consumption mix',
                'BEV footprint - Segment C - Consumption mix',
                'BEV footprint - Segment JC - Consumption mix',
                'BEV footprint - Segment JE - Consumption mix']

    captions = {'2-panel':['(a) A-segment (mini)', '(b) JE-segment (mid-size SUV)'],
                '4-panel':['(a) A-segment (mini)',
                     '(b) C-segment (medium)',
                     '(c) JC-segment (compact SUV)',
                     '(d) JE-segment (mid-size SUV)']
                }

    if num_panels == 2:
        nrows = 1
        col_list = [col_list[0]] + [col_list[-1]]
        captions = captions['2-panel']
        figsize= (9.5, 5)
    elif num_panels == 4:
        nrows = 2
        captions = captions['4-panel']
        figsize= (9.5, 8)

    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True, squeeze=True,
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.03},
                             figsize=figsize, dpi=600)

    # Plot maps
    for col, ax in zip(col_list, axes.flatten()):
        mapping_data[mapping_data[col].isna()].plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
        mapping_data[mapping_data[col].notna()].plot(ax=ax, column=col, cmap=cmap_BEV, vmax=vmax, vmin=vmin, norm=norm, edgecolor='k', linewidth=0.3, alpha=0.8)

        if ei_countries is not None:
            mapping_data.loc[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax, column=col, facecolor='none', edgecolor='darkgrey', linewidth=0.3, hatch=5*'.', alpha=0.5, zorder=1)
            mapping_data[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax, linewidth=0.3, facecolor='none', edgecolor='k', alpha=1)

        annotate_map(ax,
                     mapping_data[mapping_data[col].notna()].index.to_list(),
                     mapping_data,
                     mapping_data[mapping_data[col].notna()][col].values,
                     max_fp,
                     threshold=threshold,
                     round_dig=0,
                     ei_countries=ei_countries
                     )

    plt.xlim((-12, 34))
    plt.ylim((32, 75))

    # Label axes
    for i, a in enumerate(fig.axes):
        a.annotate(captions[i], xy=(0.02, 0.92), xycoords='axes fraction', fontsize=12)

    plt.yticks([])
    plt.xticks([])

    sns.reset_orig()
    cb = plt.cm.ScalarMappable(cmap=cmap_BEV, norm=norm)
    cb.set_array([])

    if num_panels == 2:
        # horizontal colorbar (1x2 figure)
        orientation = 'horizontal'
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.13, 0.13, 0.76, 0.035])
        rotation = 0
        keeper = 'Fig_3_2p ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
    elif num_panels == 4:
        # for vertical colorbar (2x2 figure)
        orientation = 'vertical'
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.815, 0.13, 0.025, 0.75])
        rotation = 90
        keeper = 'Fig_3_4p ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())


    cbar = fig.colorbar(cb, cax=cbar_ax, extend='both', orientation=orientation)
    cbar.set_alpha(0.7)
    cbar.set_label('Lifecycle BEV carbon intensity, \n g CO$_2$ eq/km', labelpad=9, fontsize=12, rotation=rotation)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # hack to remove minor ticks on colorbar extending past min/max values (bug in matplotlib?)
    minorticks = cbar.ax.get_yticks(minor=True)
    minorticks = [i for i in minorticks if (i >= 0 and i <= 1)]
    cbar.ax.yaxis.set_ticks(minorticks, minor=True)

    cbar.ax.tick_params(labelsize=9, pad=4)

    if export_figures:
        plt.savefig(os.path.join(fp_figure, keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, keeper), bbox_inches='tight')

    plt.show()

#%%  Figure 4 - Plot share of production emissions
def plot_fig4(exp, fp_figure, mapping_data, export_figures, ei_countries, num_panels):
    threshold = 0.75  # set up threshold for changing the annotation colour

    col_list = ['Production as share of total footprint - Segment A - Consumption mix',
                'Production as share of total footprint - Segment C - Consumption mix',
                'Production as share of total footprint - Segment JC - Consumption mix',
                'Production as share of total footprint - Segment JE - Consumption mix']

    captions = {'2-panel':['(a) A-segment (mini)', '(b) JE-segment (mid-sized SUV)'],
                '4-panel':['(a) A-segment (mini)',
                         '(b) C-segment (medium)',
                         '(c) JC-segment (compact SUV)',
                         '(d) JE-segment (mid-sized SUV)']
                }

    if num_panels == 2:
        # if only 2 panels are presented, present segments A and F
        nrows = 1
        col_list = [col_list[0]] + [col_list[-1]]
        captions = captions['2-panel']
        figsize= (9.5, 5)
    elif num_panels == 4:
        # present all segments
        nrows = 2
        captions = captions['4-panel']
        figsize= (9.5, 8)
        orientation = 'vertical'

    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True, squeeze=True,
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.03},
                             figsize=figsize, dpi=600)  # for 2x2 figure: figsize=(9.5, 8)


    for col, ax in zip(col_list, axes.flatten()):
        mapping_data[mapping_data[col].isna()].plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
        mapping_data[mapping_data[col].notna()].plot(ax=ax, column=col, cmap=cm.batlow_r, vmax=100, vmin=0, edgecolor='k', linewidth=0.3, alpha=0.8)

        if ei_countries is not None:
            mapping_data.loc[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax, column=col, facecolor='none', edgecolor='darkgrey', linewidth=0.3, hatch=5*'.', alpha=0.5, zorder=1)
            mapping_data[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax, linewidth=0.3, facecolor='none', edgecolor='k', alpha=1)

        annotate_map(ax,
                     mapping_data[mapping_data[col].notna()].index.to_list(),
                     mapping_data,
                     mapping_data[mapping_data[col].notna()][col].values,
                     100,
                     threshold=threshold,
                     round_dig=0,
                     ei_countries=ei_countries
                     )

    plt.xlim((-12, 34))
    plt.ylim((32, 75))

    plt.yticks([])
    plt.xticks([])

    for i, a in enumerate(fig.axes):
        a.annotate(captions[i], xy=(0.02, 0.92), xycoords='axes fraction', fontsize=12)

    sns.reset_orig()
    cb = plt.cm.ScalarMappable(cmap=cm.batlow_r, norm=colors.Normalize(0,100))
    cb.set_array([])

    if num_panels == 2:
        # horizontal colorbar (1x2 figure)
        orientation = 'horizontal'
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.13, 0.13, 0.76, 0.035])
        rotation = 0
        keeper = 'Fig_4_2p ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
    elif num_panels == 4:
        # for vertical colorbar (2x2 figure)
        orientation = 'vertical'
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.815, 0.13, 0.025, 0.75])
        rotation = 90
        keeper = 'Fig_4_4p ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())

    cbar = fig.colorbar(cb, cax=cbar_ax, format=ticker.PercentFormatter(), orientation=orientation)
    cbar.set_alpha(0.7)
    cbar.set_label('Production share of total lifecycle emissions, \n in %', labelpad=9, fontsize=12, rotation=rotation)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    if export_figures:
        plt.savefig(os.path.join(fp_figure, keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, keeper), bbox_inches='tight')


# %% Figure 5 - Absolute mitigation by electrification
def plot_fig5a(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures, ei_countries, num_panels):
    # First, with A-segment ratios and the difference of A- and F-segments (delta)
    # Second, with A-segment and F-segment ratios (original Figure 3) + separate delta figure
    # Third, with absolute difference between BEV and ICEV for both panels

    # Calculate values and set up "shortcut"
    mapping_data['abs_diff_A'] = ICEV_total_impacts['A'].reindex_like(mapping_data, method='pad').subtract(mapping_data['BEV impacts - Segment A - Consumption mix'])
    mapping_data['abs_diff_C'] = ICEV_total_impacts['C'].reindex_like(mapping_data, method='pad').subtract(mapping_data['BEV impacts - Segment C - Consumption mix'])
    mapping_data['abs_diff_JC'] = ICEV_total_impacts['JC'].reindex_like(mapping_data, method='pad').subtract(mapping_data['BEV impacts - Segment JC - Consumption mix'])
    mapping_data['abs_diff_JE'] = ICEV_total_impacts['JE'].reindex_like(mapping_data, method='pad').subtract(mapping_data['BEV impacts - Segment JE - Consumption mix'])

    abs_diff = mapping_data[mapping_data['abs_diff_A'].notna()]
    abs_diff.set_index('ADM0_A3', inplace=True)
    abs_diff = abs_diff.iloc[:, -4:]

    # Calculate extreme values for colourmap normalization
    rmax = max(mapping_data['abs_diff_A'].max(),
               mapping_data['abs_diff_C'].max(),
               mapping_data['abs_diff_JC'].max(),
               mapping_data['abs_diff_JE'].max()) - 2

    rmin = min((mapping_data['abs_diff_A'].min(),
                mapping_data['abs_diff_C'].min(),
                mapping_data['abs_diff_JC'].min(),
                mapping_data['abs_diff_JE'].min())) + 2

    N = 7  # Number of sections from full colormap; must be odd number for proper normalization
    if N==7:
        cmap_diff = colors.ListedColormap(['#6a0000',
                                           '#b94d30',
                                           '#f0a079',
                                           '#ffffe0',
                                           '#99bb8a',
                                           '#47793d',
                                           '#003900'])
    else:
        cmap_diff = plt.get_cmap(cmap_map(lambda x: x*0.75, cm.RdYlGn), N)


    cmap_col = [cmap_diff(i) for i in np.linspace(0, 1, N)]  # retrieve colormap colors

    # Make manual boundaries for cmap
    # range of negative values approximately 1/3 of that of positive values;
    # cut-off colormap asymmetrically for visual 'equality'
    # cmap = cmap_col[2:]  # "trim" bottom section of colormap colors

    if np.abs(rmin) > rmax:
        lower_bound = [i for i in np.linspace(int(rmin), -2.5, 4)]
        boundaries = lower_bound + [-i for i in lower_bound[::-1] if -i<= rmax] + [rmax] # define boundaries of colormap transitions
    else:
        upper_bound = [i for i in np.linspace(2.5, int(rmax), 4)]
        # boundaries = [rmin, -2.5] + upper_bound  # define boundaries of colormap transitions
        boundaries = [rmin] + [-i for i in upper_bound[::-1] if -i>= rmin] + upper_bound # define boundaries of colormap transitions
    n = N - (len(boundaries) - 1)
    if np.abs(rmin) > rmax and n>0:
        cmap = cmap_col[:-(n)]  # "trim" upper section of colormap colors
    elif np.abs(rmin) < rmax and n>0:
        cmap = cmap_col[n:]
    else:
        cmap = cmap_col

    cmap_colors, norm = colors.from_levels_and_colors(boundaries, colors=[cmap[0]] + cmap + [cmap[-1]], extend='both')

    # Plot maps; start with countries not included, then countries with values
    col_list = ['abs_diff_A', 'abs_diff_C', 'abs_diff_JC', 'abs_diff_JE']
    captions = {'2-panel':['(a) A-segment (mini)', '(b) JE-segment (mid-size SUV)'],
                '4-panel':['(a) A-segment (mini)',
                         '(b) C-segment (medium)',
                         '(c) JC-segment (compact SUV)',
                         '(d) JE-segment (mid-size SUV)']
                }

    if num_panels == 2:
        # if only 2 panels are presented, present segments A and F
        nrows = 1
        col_list = [col_list[0]] + [col_list[-1]]
        captions = captions['2-panel']
        figsize= (9.5, 6.5)
    elif num_panels == 4:
        # present all segments
        nrows = 2
        captions = captions['4-panel']
        figsize= (9.5, 8)
        orientation = 'vertical'

    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True, squeeze=True,
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.03}, figsize=figsize, dpi=600)

    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.01, hspace=0.01)

    threshold = boundaries[-2] / rmax  # threshold value to determine annotation text color

    for col, ax in zip(col_list, axes.flatten()):
        mapping_data[mapping_data[col].isna()].plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
        mapping_data[mapping_data[col].notna()].plot(ax=ax, column=col, cmap=cmap_colors, alpha=0.8, edgecolor='k', linewidth=0.3, norm=norm, vmax=rmax, vmin=rmin)

        if ei_countries is not None:
            mapping_data[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax,
                         column=col, facecolor='none', edgecolor='darkgrey', linewidth=0.1, hatch=5*'.', alpha=0.5, zorder=1)
            mapping_data[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax, linewidth=0.3, facecolor='none', edgecolor='k', alpha=1)

        annotate_map(ax,
                     mapping_data[mapping_data[col].notna()].index.to_list(),
                     mapping_data,
                     mapping_data[mapping_data[col].notna()][col].values,
                     max(rmax, np.abs(rmin)),
                     threshold=threshold,
                     ei_countries=ei_countries)

    # Label panels
    for i, a in enumerate(fig.axes):
        a.annotate(captions[i], xy=(0.02, 0.92), xycoords='axes fraction', fontsize=12)

    # Format axes and set axis limits
    plt.xlim((-12, 34))
    plt.ylim((32, 75))

    plt.yticks([])
    plt.xticks([])

    # Add colorbar legend
    sns.reset_orig()

    cb = plt.cm.ScalarMappable(cmap=cmap_colors, norm=norm)
    cb.set_array([])

    if num_panels == 2:
        # horizontal colorbar (1x2 figure)
        orientation = 'horizontal'
        fig.subplots_adjust(bottom=0.1)
        cbar_ax = fig.add_axes([0.13, 0.13, 0.76, 0.035], frame_on=True)
        rotation = 0
        keeper = 'Fig_5_2p_abs ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
    elif num_panels == 4:
        # for vertical colorbar (2x2 figure)
        orientation = 'vertical'
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.815, 0.13, 0.025, 0.75])
        rotation = 90
        keeper = 'Fig_5_4p_abs ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())

    ticks_lower = round_up_down(rmin, 5, 'up')
    ticks_upper = round_up_down(rmax, 5, 'down')
    ticks = [i for i in range(ticks_lower, ticks_upper+1, 5)]

    cbar = fig.colorbar(cb, cax=cbar_ax, extend='both', spacing='proportional', ticks=ticks, orientation=orientation)
    cbar.set_label('Lifecycle CO$_2$ mitigated through electrification, \n t CO$_2$-eq/vehicle', rotation=rotation, labelpad=9, fontsize=12)

    r1 = Rectangle((-200, -50), 500, 85, fc='w', alpha=0.2)
    cbar_ax.add_patch(r1)

    minorticks = np.arange(int(cbar.vmin), int(cbar.vmax) + 1, 1)
    # minorticks = (minorticks - cbar.vmin) / (cbar.vmax - cbar.vmin)  # normalize minor ticks to [0,1] scale
    if num_panels == 2:
        cbar.ax.xaxis.set_minor_locator(FixedLocator(minorticks))  # minor ticks on horizontal
    elif num_panels == 4:
        cbar.ax.yaxis.set_minor_locator(FixedLocator(minorticks))  # minor ticks on vertical
    cbar.ax.tick_params(labelsize=9, pad=4)

    if export_figures:
        plt.savefig(os.path.join(fp_figure, keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, keeper), bbox_inches='tight')

    plt.show()


def plot_fig5b(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures, ei_countries, num_panels):
    # First, with A-segment ratios and the difference of A- and F-segments (delta)
    # Second, with A-segment and F-segment ratios (original Figure 3) + separate delta figure
    # Third, with absolute difference between BEV and ICEV for both panels

    # Calculate values and set up "shortcut"
    mapping_data['rel_diff_A'] = mapping_data['BEV impacts - Segment A - Consumption mix'].subtract(ICEV_total_impacts['A'].reindex_like(mapping_data, method='pad')).div(ICEV_total_impacts['A'].reindex_like(mapping_data, method='pad'))*100
    mapping_data['rel_diff_C'] = mapping_data['BEV impacts - Segment C - Consumption mix'].subtract(ICEV_total_impacts['C'].reindex_like(mapping_data, method='pad')).div(ICEV_total_impacts['C'].reindex_like(mapping_data, method='pad'))*100
    mapping_data['rel_diff_JC'] = mapping_data['BEV impacts - Segment JC - Consumption mix'].subtract(ICEV_total_impacts['JC'].reindex_like(mapping_data, method='pad')).div(ICEV_total_impacts['JC'].reindex_like(mapping_data, method='pad'))*100
    mapping_data['rel_diff_JE'] = mapping_data['BEV impacts - Segment JE - Consumption mix'].subtract(ICEV_total_impacts['JE'].reindex_like(mapping_data, method='pad')).div(ICEV_total_impacts['JE'].reindex_like(mapping_data, method='pad'))*100

    rel_diff = mapping_data[mapping_data['rel_diff_A'].notna()]
    rel_diff.set_index('ADM0_A3', inplace=True)
    rel_diff = rel_diff.iloc[:, -4:]

    # Calculate extreme values for colourmap normalization
    rmax = max(mapping_data['rel_diff_A'].max(),
               mapping_data['rel_diff_C'].max(),
               mapping_data['rel_diff_JC'].max(),
               mapping_data['rel_diff_JE'].max()) - 5

    rmin = min((mapping_data['rel_diff_A'].min(),
                mapping_data['rel_diff_C'].min(),
                mapping_data['rel_diff_JC'].min(),
                mapping_data['rel_diff_JE'].min())) + 5

    N = 7  # Number of sections from full colormap; must be odd number for proper normalization
    if N==7:
        cmap_diff = colors.ListedColormap(['#003900',
                                           '#47793d',
                                           '#99bb8a',
                                           '#ffffe0',
                                           '#f0a079',
                                           '#b94d30',
                                           '#6a0000'])
    else:
        cmap_diff = plt.get_cmap(cmap_map(lambda x: x*0.75, cm.RdYlGn), N) #plt.get_cmap('RdYlGn', N)


    cmap_col = [cmap_diff(i) for i in np.linspace(0, 1, N)]  # retrieve colormap colors

    # Make manual boundaries for cmap
    # range of negative values and positive values is asymmetric;
    if np.abs(rmin) > rmax:
        lower_bound = [i for i in np.linspace(int(rmin), -10, 4)]
        boundaries = lower_bound + [-i for i in lower_bound[::-1] if -i<= rmax] + [rmax] # define boundaries of colormap transitions
    else:
        upper_bound = [i for i in np.linspace(10, int(rmax), 4)]
        # boundaries = [rmin, -2.5] + upper_bound  # define boundaries of colormap transitions
        boundaries = [rmin] + [-i for i in upper_bound[::-1] if -i>= rmin] + upper_bound # define boundaries of colormap transitions

    n = N - (len(boundaries) - 1)
    if np.abs(rmin) > rmax and n>0:
        cmap = cmap_col[:-(n)]  # "trim" upper section of colormap colors
    elif np.abs(rmin) < rmax and n>0:
        cmap = cmap_col[n:]
    else:
        cmap = cmap_col
    cmap_colors, norm = colors.from_levels_and_colors(boundaries, colors=[cmap[0]] + cmap + [cmap[-1]], extend='both')

    # Plot maps; start with countries not included, then countries with values
    col_list = ['rel_diff_A', 'rel_diff_C', 'rel_diff_JC', 'rel_diff_JE']
    captions = {'2-panel':['(a) A-segment (mini)', '(b) JE-segment (mid-size SUV)'],
                '4-panel':['(a) A-segment (mini)',
                         '(b) C-segment (medium)',
                         '(c) JC-segment (compact SUV)',
                         '(d) JE-segment (mid-size SUV)']
                }
    if num_panels == 2:
        # if only 2 panels are presented, present segments A and JE
        nrows = 1
        col_list = [col_list[0]] + [col_list[-1]]
        captions = captions['2-panel']
        figsize= (9.5, 6.5)
    elif num_panels == 4:
        # present all segments
        nrows = 2
        captions = captions['4-panel']
        figsize= (9.5, 8)
        orientation = 'vertical'

    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True, squeeze=True,
                           gridspec_kw={'wspace': 0.03, 'hspace': 0.03}, figsize=figsize, dpi=600)

    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.01, hspace=0.01)

    threshold = (boundaries[1] / np.abs(rmin))  # threshold value to determine annotation text color

    for col, ax in zip(col_list, axes.flatten()):
        mapping_data[mapping_data[col].isna()].plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
        mapping_data[mapping_data[col].notna()].plot(ax=ax, column=col, cmap=cmap_colors, alpha=0.8, edgecolor='k', linewidth=0.3,  norm=norm, vmax=rmax, vmin=rmin)

        if ei_countries is not None:
            mapping_data[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax,
                         column=col, facecolor='none', edgecolor='darkgrey', linewidth=0.1, hatch=5*'.', alpha=0.5, zorder=1)
            mapping_data[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax, linewidth=0.3, facecolor='none', edgecolor='k', alpha=1)

        annotate_map(ax,
                     mapping_data[mapping_data[col].notna()].index.to_list(),
                     mapping_data,
                     mapping_data[mapping_data[col].notna()][col].values,
                     max(rmax, np.abs(rmin)),
                     round_dig=0,
                     threshold=threshold,
                     ei_countries=ei_countries)

    # Label panels
    for i, a in enumerate(fig.axes):
        a.annotate(captions[i], xy=(0.02, 0.92), xycoords='axes fraction', fontsize=12)

    # Format axes and set axis limits
    plt.xlim((-12, 34))
    plt.ylim((32, 75))

    plt.yticks([])
    plt.xticks([])

    # Add colorbar legend
    cb = plt.cm.ScalarMappable(cmap=cmap_colors, norm=norm)
    cb.set_array([])

    if num_panels == 2:
        # horizontal colorbar (1x2 figure)
        orientation = 'horizontal'
        fig.subplots_adjust(bottom=0.10)
        cbar_ax = fig.add_axes([0.13, 0.13, 0.76, 0.035], frame_on=True)
        rotation = 0
        keeper = 'Fig_5_2p_rel ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
    elif num_panels == 4:
        # for vertical colorbar (2x2 figure)
        orientation = 'vertical'
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.815, 0.13, 0.025, 0.75])
        rotation = 90
        keeper = 'Fig_5_4p_rel ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())


    ticks_lower = round_up_down(rmin, 10, 'up')
    ticks_upper = round_up_down(rmax, 10, 'down')
    ticks = [i for i in np.arange(ticks_lower, ticks_upper+1, 10)]

    cbar = fig.colorbar(cb, cax=cbar_ax, extend='both', spacing='proportional', ticks=ticks, orientation=orientation)
    cbar.set_label('Lifecycle CO$_2$ mitigated through electrification, \n % difference from ICEV', rotation=rotation, labelpad=9, fontsize=12)

    minorticks = np.arange(ticks_lower, ticks_upper, 5)

    # minorticks = (minorticks - cbar.vmin) / (cbar.vmax - cbar.vmin)  # normalize minor ticks to [0,1] scale
    if num_panels == 2:
        cbar.ax.xaxis.set_minor_locator(FixedLocator(minorticks))
        r1 = Rectangle((-200, -130), 500, 200,  fc='w', alpha=0.2)
    elif num_panels == 4:
        cbar.ax.yaxis.set_minor_locator(FixedLocator(minorticks))
        r1 = Rectangle((-140, -190), 200, 500,  fc='w', alpha=0.2)
    cbar.ax.tick_params(labelsize=9, pad=4)

    # Manually tack on semi-transparent rectangle on colorbar to match alpha of plot; workaround for weird rendering of colorbar with alpha
    cbar_ax.add_patch(r1)

    if export_figures:
        plt.savefig(os.path.join(fp_figure, keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, keeper), bbox_inches='tight')

    plt.show()

#%%
def plot_fig5_old(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures, ei_countries, num_panels):
    # # Make multiple versions of Figure 3
    #
    # First, with A-segment ratios and the difference of A- and F-segments (delta)
    # Second, with A-segment and F-segment ratios (original Figure 3) + separate delta figure
    # Third, with absolute difference between BEV and ICEV for both panels

    # Calculate values and set up "shortcut"
    mapping_data['abs_diff_A'] = ICEV_total_impacts['A'].reindex_like(mapping_data, method='pad').subtract(mapping_data['BEV impacts - Segment A - Consumption mix'])
    mapping_data['abs_diff_C'] = ICEV_total_impacts['C'].reindex_like(mapping_data, method='pad').subtract(mapping_data['BEV impacts - Segment C - Consumption mix'])
    mapping_data['abs_diff_JC'] = ICEV_total_impacts['JC'].reindex_like(mapping_data, method='pad').subtract(mapping_data['BEV impacts - Segment JC - Consumption mix'])
    mapping_data['abs_diff_JE'] = ICEV_total_impacts['JE'].reindex_like(mapping_data, method='pad').subtract(mapping_data['BEV impacts - Segment JE - Consumption mix'])

    abs_diff = mapping_data[mapping_data['abs_diff_A'].notna()]
    abs_diff.set_index('ADM0_A3', inplace=True)
    abs_diff = abs_diff.iloc[:, -4:]

    # Calculate extreme values for colourmap normalization
    rmax = max(mapping_data['abs_diff_A'].max(),
               mapping_data['abs_diff_C'].max(),
               mapping_data['abs_diff_JC'].max(),
               mapping_data['abs_diff_JE'].max()) - 2

    rmin = min((mapping_data['abs_diff_A'].min(),
                mapping_data['abs_diff_C'].min(),
                mapping_data['abs_diff_JC'].min(),
                mapping_data['abs_diff_JE'].min())) + 2

    N = 7  # Number of sections from full colormap; must be odd number for proper normalization
    if N==7:
        cmap_diff = colors.ListedColormap(['#6a0000',
                                           '#b94d30',
                                           '#f0a079',
                                           '#ffffe0',
                                           '#99bb8a',
                                           '#47793d',
                                           '#003900'])
    else:
        cmap_diff = plt.get_cmap(cmap_map(lambda x: x*0.75, cm.RdYlGn), N) #plt.get_cmap('RdYlGn', N)


    cmap_col = [cmap_diff(i) for i in np.linspace(0, 1, N)]  # retrieve colormap colors

    # Make manual boundaries for cmap
    # range of negative values approximately 1/3 of that of positive values;
    # cut-off colormap asymmetrically for visual 'equality'
    cutoff = int(int(N/2) - ((np.abs(rmin) - 0) / (rmax - 0)) * int(N/2))
    cmap = cmap_col[2:]  # "trim" bottom section of colormap colors
    upper_bound = [i for i in np.linspace(2.5, int(rmax), 4)]
    boundaries = [rmin, -2.5] + upper_bound  # define boundaries of colormap transitions

    cmap_colors, norm = colors.from_levels_and_colors(boundaries, colors=[cmap[0]] + cmap + [cmap[-1]], extend='both')

    # Plot maps; start with countries not included, then countries with values
    col_list = ['abs_diff_A', 'abs_diff_C', 'abs_diff_JC', 'abs_diff_JE']
    captions = {'2-panel':['(a) A-segment (mini)', '(b) JE-segment (mid-size SUV)'],
                '4-panel':['(a) A-segment (mini)',
                         '(b) C-segment (medium)',
                         '(c) JC-segment (compact SUV)',
                         '(d) JE-segment (mid-size SUV)']
                }
    if num_panels == 2:
        # if only 2 panels are presented, present segments A and F
        nrows = 1
        col_list = [col_list[0]] + [col_list[-1]]
        captions = captions['2-panel']
        figsize= (9.5, 6.5)
    elif num_panels == 4:
        # present all segments
        nrows = 2
        captions = captions['4-panel']
        figsize= (9.5, 8)
        orientation = 'vertical'

    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True, squeeze=True,
                           gridspec_kw={'wspace': 0.03, 'hspace': 0.03}, figsize=figsize, dpi=600)

    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.01, hspace=0.01)

    threshold = boundaries[-2] / rmax  # threshold value to determine annotation text color

    for col, ax in zip(col_list, axes.flatten()):
         mapping_data[mapping_data[col].isna()].plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
         mapping_data[mapping_data[col].notna()].plot(ax=ax, column=col, cmap=cmap_colors, alpha=0.8, edgecolor='k', linewidth=0.3, norm=norm, vmax=rmax, vmin=rmin)

         if ei_countries is not None:
             mapping_data[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax,
                          column=col, facecolor='none', edgecolor='darkgrey', linewidth=0.1, hatch=5*'.', alpha=0.5, zorder=1)
             mapping_data[mapping_data['ISO_A2'].isin(ei_countries)].plot(ax=ax, linewidth=0.3, facecolor='none', edgecolor='k', alpha=1)

         annotate_map(ax,
                      mapping_data[mapping_data[col].notna()].index.to_list(),
                      mapping_data,
                      mapping_data[mapping_data[col].notna()][col].values,
                      max(rmax, np.abs(rmin)),
                      threshold=threshold,
                      ei_countries=ei_countries)

    # Label panels
    for i, a in enumerate(fig.axes):
        a.annotate(captions[i], xy=(0.02, 0.92), xycoords='axes fraction', fontsize=12)

    # Format axes and set axis limits
    plt.xlim((-12, 34))
    plt.ylim((32, 75))

    plt.yticks([])
    plt.xticks([])

    # Add colorbar legend

    cb = plt.cm.ScalarMappable(cmap=cmap_colors, norm=norm)
    cb.set_array([])

    if num_panels == 2:
        # horizontal colorbar (1x2 figure)
        orientation = 'horizontal'
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.13, 0.13, 0.76, 0.035])
        rotation = 0
        keeper = 'Fig_5_2p ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
    elif num_panels == 4:
        # for vertical colorbar (2x2 figure)
        orientation = 'vertical'
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.815, 0.13, 0.025, 0.75])
        rotation = 90
        keeper = 'Fig_5_4p ' + exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())

    ticks_lower = round_up_down(rmin, 5, 'up')
    ticks_upper = round_up_down(rmax, 5, 'down')
    ticks = [i for i in range(ticks_lower, ticks_upper+1, 5)]
    cbar = fig.colorbar(cb, cax=cbar_ax, extend='both', spacing='proportional', ticks=ticks, orientation=orientation)
    cbar.set_label('Lifecycle CO$_2$ mitigated through electrification, \n t CO$_2$-eq/vehicle', rotation=rotation, labelpad=9, fontsize=12)
    minorticks = np.arange(int(cbar.vmin), int(cbar.vmax) + 1, 1)
    minorticks = (minorticks - cbar.vmin) / (cbar.vmax - cbar.vmin)  # normalize minor ticks to [0,1] scale
    if num_panels == 2:
        cbar.ax.xaxis.set_minor_locator(FixedLocator(minorticks))
    elif num_panels == 4:
        cbar.ax.yaxis.set_minor_locator(FixedLocator(minorticks))
    cbar.ax.tick_params(labelsize=9, pad=4)

    if export_figures:
        plt.savefig(os.path.join(fp_figure, keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, keeper), bbox_inches='tight')

    plt.show()

#%% Figure 6 - sensitivity with vehicle lifetimes
def plot_fig6(results, fig_dict, export_figures, fp_figure=fp_figures):
    sns.set_style('whitegrid')

    clrs = ['#2e78b8','#206619', '#555c23', '#b8145b']  # colours for ICEV lines
    clrs2 = ['#4e7496', '#3F8638', '#737D2F', '#681E3E']  # colours for BEV ranges

    for el_approach in fig_dict.keys():
        baseline, short_BEV_life, long_BEV_life = fig_dict[el_approach]
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), gridspec_kw={'wspace': 0.05, 'hspace': 0.075}, sharex=True, sharey=True)
        labels = results.index.tolist()  # country labels for x-ticks
        x_coords = np.arange(len(results.index))  # proxy x-coordinates for countries

        for ax, seg, clr, clr2 in zip(axes.flatten(), results[baseline].columns, clrs, clrs2):

            # Plot ICEV values
            ax.plot(x_coords, results[('ICEV 180k', seg)], c=clr)
            ax.plot(x_coords, results[('ICEV 200k', seg)], ls='--', c=clr)
            ax.plot(x_coords, results[('ICEV 250k', seg)], ls=':', c=clr)

            # Plot BEV values
            baseline_marker = ax.scatter(x_coords, results[(baseline, seg)],
                                         c='k', s=11, marker='x', zorder=10,
                                         label='Baseline BEV intensity (180k km lifetime)'
                                         )

            diff = results[(short_BEV_life, seg)] - results[(long_BEV_life, seg)]  # range of intensities for sensitivity
            ax.bar(x_coords, diff, bottom=results[(long_BEV_life, seg)], color=clr2, edgecolor=None, alpha=0.6)

            ax.tick_params(axis='x', which='major', pad=3)
            ax.set_xticks(x_coords)
            ax.set_xticklabels(labels, rotation=90, va='baseline')

            if seg == 'D' or seg=='F':
                ax.set_xlabel('Country', fontsize=13, labelpad=10)

        ax.set_xlim(left=-0.6, right=29.6)

        # set up segment labels for each subplot
        captions = ['(a) A-segment (mini)', '(b) C-segment (medium)',
                    '(c) JC-segment (compact SUV)', '(d) JE-segment (mid-size SUV)']

        for i, a in enumerate(fig.axes):
            a.annotate(captions[i], xy=(0.02, 0.92), xycoords='axes fraction', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="square,pad=0.12", fc="w", ec="w", lw=2))

        # Set up legend
        # First, make handles for each - range bars, markers and ICEV lines
        range_int = mpatches.Patch(color=clrs2[0], ec=None, alpha=0.6,
                                   label='BEV intensity, 150-250k km lifetime (range)')

        ICEV_baseline =  mlines.Line2D([], [], color=clrs[0], label='ICEV intensity (180k km lifetime)')
        ICEV_200 =  mlines.Line2D([], [], ls='--', color=clrs[0], label='ICEV intensity (200k km lifetime)')
        ICEV_250 =  mlines.Line2D([], [], ls=':', color=clrs[0], label='ICEV intensity (250k km lifetime)')

        handles_BEV = [range_int,]
        handles_BEV2 = [baseline_marker]
        handles_ICEV = [ICEV_baseline, ICEV_200, ICEV_250]

        plt.subplots_adjust(bottom=0.175)  # make extra room under figure for legend

        # draw each "column" of the legend separately
        fig.legend(handles=handles_BEV, loc='center left', bbox_to_anchor=(0.0755, 0.065),
                   borderaxespad=0.1, handletextpad=1., columnspacing=6,
                   fontsize=12, handlelength=1, handleheight=4, frameon=False)
        fig.text(0.128, 0.0645, '150k', fontsize=8.5, zorder=10)  # labels for range extremities
        fig.text(0.128, -0.0145, '250k', fontsize=8.5, zorder=10)  # labels for range extremities
        fig.legend(handles=handles_BEV2, loc='center left', bbox_to_anchor=(0.35, 0.065),
                   borderaxespad=0.05, handletextpad=1., columnspacing=6,
                   fontsize=12, markerscale=2.5, frameon=False)
        fig.legend(handles=handles_ICEV, loc='center left', bbox_to_anchor=(0.637, 0.065),
                   borderaxespad=0.1, handletextpad=1., columnspacing=6,
                   fontsize=12, frameon=False)

        # Draw legend border
        fig.patches.extend([plt.Rectangle((0.122, -0.02), width=.775, height=0.11, figure=fig,
                                          transform=fig.transFigure, fill=False, ec='lightgrey', zorder=10)])

        # Common y-axis labels
        fig.text(0.056, 0.5, 'Vehicle lifecycle CO$_2$ intensity', va='center', rotation='vertical', fontsize=13)
        fig.text(0.072, 0.5, 'g CO$_2$-eq/km', va='center', rotation='vertical', fontsize=13)

        if export_figures:
            keeper = " run {:%d-%m-%y, %H_%M}".format(datetime.now())
            plt.savefig(os.path.join(fp_figure, 'Fig 6 - sensitivity ' + el_approach + keeper + '.pdf'), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(fp_figure, 'Fig 6 - sensitivity ' + el_approach + keeper), bbox_inches='tight')

        plt.show()


def sensitivity_plot_lecture(results, export_figures, fp_figure=fp_results):
    # Reproduction of Figure 6 with larger text for presentation slides
    clrs = ['#2e78b8','#206619', '#555c23', '#b8145b']  # colours for ICEV lines
    clrs2 = ['#4e7496', '#3F8638', '#737D2F', '#681E3E']  # colours for BEV ranges
    fig, axes = plt.subplots(2, 2, figsize=(20, 11.5), gridspec_kw={'wspace': 0.05, 'hspace': 0.075}, sharex=True, sharey=True)
    labels = results.index.tolist()  # country labels for x-ticks
    x_coords = np.arange(len(results.index))  # proxy x-coordinates for countries

    for ax, seg, clr, clr2 in zip(axes.flatten(), results['baseline'].columns, clrs, clrs2):

        # set up segment labels for each subplot
        captions = ['(a) A-segment (mini)', '(b) C-segment (medium)',
                    '(c) JC-segment (compact SUV)', '(d) JE-segment (mid-size SUV)']

        for i, a in enumerate(fig.axes):
            a.annotate(captions[i], xy=(0.02, 0.92), xycoords='axes fraction', fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle="square,pad=0.12", fc="w", ec="w", lw=2))

        ax.set_ylim(bottom=0, top=375)
        if seg=='A':
            plt.savefig(os.path.join(fp_figure, 'Fig 5 - sensitivity base'), bbox_inches='tight')

        # Plot ICEV values
        ax.plot(x_coords, results[('ICEV 180', seg)], c=clr)
        ax.plot(x_coords, results[('ICEV 200', seg)], ls='--', c=clr)
        ax.plot(x_coords, results[('ICEV 250', seg)], ls=':', c=clr)
        if seg=='A':
            plt.savefig(os.path.join(fp_figure, 'Fig 5 - sensitivity ICEVs'), bbox_inches='tight')

        # Plot BEV values
        baseline_marker = ax.scatter(x_coords, results[('baseline', seg)],
                                     c='k', s=12, marker='x', zorder=10,
                                     label='Baseline BEV intensity (180k km lifetime)'
                                     )
        if seg=='A':
            plt.savefig(os.path.join(fp_figure, 'Fig 5 - sensitivity baseline'), bbox_inches='tight')

        diff = results[('short_BEV_life', seg)] - results[('long_BEV_life', seg)]  # range of intensities for sensitivity
        ax.bar(x_coords, diff, bottom=results[('long_BEV_life', seg)], color=clr2, edgecolor=None, alpha=0.6)

        ax.tick_params(axis='x', which='major', pad=4)
        ax.set_xticks(x_coords)
        ax.set_xticklabels(labels, rotation=90, va='baseline', fontsize=13)
        ax.tick_params(axis='y', labelsize=13)

        if seg == 'D' or seg=='F':
            ax.set_xlabel('Country', fontsize=15, labelpad=10)

    ax.set_xlim(left=-0.6, right=29.6)


    # Set up legend
    # First, make handles for each - range bars, markers and ICEV lines
    range_int = mpatches.Patch(color=clrs2[0], ec=None, alpha=0.6,
                               label='BEV intensity, 150-250k km lifetime (range)')

    ICEV_baseline =  mlines.Line2D([], [], color=clrs[0], label='ICEV intensity (180k km lifetime)')
    ICEV_200 =  mlines.Line2D([], [], ls='--', color=clrs[0], label='ICEV intensity (200k km lifetime)')
    ICEV_250 =  mlines.Line2D([], [], ls=':', color=clrs[0], label='ICEV intensity (250k km lifetime)')

    handles_BEV = [range_int,]
    handles_BEV2 = [baseline_marker]
    handles_ICEV = [ICEV_baseline, ICEV_200, ICEV_250]

    plt.subplots_adjust(bottom=0.175)  # make extra room under figure for legend

    # draw each "column" of the legend separately
    fig.legend(handles=handles_BEV, loc='center left', bbox_to_anchor=(0.0755, 0.065),
               borderaxespad=0.1, handletextpad=1., columnspacing=6,
               fontsize=15, handlelength=1, handleheight=4, frameon=False)
    fig.text(0.128, 0.0645, '150k', fontsize=8.5, zorder=10)  # labels for range extremities
    fig.text(0.128, -0.0135, '250k', fontsize=8.5, zorder=10)  # labels for range extremities
    fig.legend(handles=handles_BEV2, loc='center left', bbox_to_anchor=(0.35, 0.065),
               borderaxespad=0.05, handletextpad=1., columnspacing=6,
               fontsize=15, markerscale=2.5, frameon=False)
    fig.legend(handles=handles_ICEV, loc='center left', bbox_to_anchor=(0.637, 0.065),
               borderaxespad=0.1, handletextpad=1., columnspacing=6,
               fontsize=15, frameon=False)

    # Draw legend border
    fig.patches.extend([plt.Rectangle((0.122, -0.02), width=.775, height=0.11, figure=fig,
                                      transform=fig.transFigure, fill=False, ec='lightgrey', zorder=10)])

    # Common y-axis labels
    fig.text(0.056, 0.5, 'Vehicle lifecycle CO$_2$ intensity', va='center', rotation='vertical', fontsize=16)
    fig.text(0.072, 0.5, 'g CO$_2$-eq/km', va='center', rotation='vertical', fontsize=16)

    if export_figures:
        keeper = " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure, 'Fig 6 - sensitivity ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, 'Fig 6 - sensitivity ' + keeper), bbox_inches='tight')

    plt.show()

#%% Plot country footprint with specific segment
def plot_country_footprint(exp, fp_figure, country, segment, start, timestamp, mapping_data, export_figures):
    """Produce country footprint based on user query."""

    # use same colour coding as for Figure 3 in paper
    cmap = colors.ListedColormap(["#c6baca",  # light purple
                                  "#83abce",
                                  "#6eb668",
                                  "#9caa41",
                                  "#815137",
                                  "#681e3e"  # red
                                  ])
    vmin=50
    vmax=375
    cmap_col = [cmap(i) for i in np.linspace(0, 1, 6)]  # retrieve colormap colors
    cmap = cmap_col
    boundaries = [i for i in np.arange(vmin, vmax, 50)]  # define boundaries of colormap transitions

    cmap_BEV, norm = colors.from_levels_and_colors(boundaries, colors=[cmap[0]]+ cmap + [cmap_col[-1]], extend='both')

    col_list = ['BEV footprint - Segment A - Consumption mix', 'BEV footprint - Segment C - Consumption mix',
                'BEV footprint - Segment JC - Consumption mix', 'BEV footprint - Segment JE - Consumption mix']

    col = [col for col in col_list if 'Segment '+ segment in col]
    if len(col) == 1:
        col = col[0]
    country_ind = mapping_data.loc[mapping_data['ISO_A2'] == country]
    if country_ind[col].values >= (4/6*(vmax-vmin)):
        color = 'grey'
    else:
        color = '#363737'

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=600)
    mapping_data.plot(ax=ax, color=color, edgecolor='darkgrey', linewidth=0.3)
    country_ind.plot(ax=ax, column=col, cmap=cmap_BEV, norm=norm, edgecolor='k', linewidth=0.3, alpha=0.8)

    plt.xlim((-12, 34))
    plt.ylim((32, 75))

    plt.yticks([])
    plt.xticks([])

    # add annotations
    timestamp = timestamp.strftime('%Y-%m-%d %H:%M %z')
    caption = f'Carbon footprint for segment {segment} BEV in {country} \n at {timestamp} (g CO$_2$e /vkm) \n (User query for {start})'
    ax.annotate(caption, xy=(0.5, 1.02), xycoords='axes fraction', fontsize=9, ha='center')

    annotate_map(ax,
                country_ind[country_ind[col].notna()].index.to_list(),
                country_ind,
                country_ind[country_ind[col].notna()][col].values,
                300,
                threshold=0,
                round_dig=0,
                )

    if export_figures:
        keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure, 'Fig_1 ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, 'Fig_1 ' + keeper), bbox_inches='tight')

    plt.show()


#%% Optional figures for plotting

def plot_el_trade(exp, fp_figure, CFEL, export_figures):
    """Plot import, exports and net trade for each country."""

    plot_trades = pd.DataFrame([CFEL['imports'], -CFEL['exports'], CFEL['imports'] - CFEL['exports']])
    plot_trades.index = ['Imports', 'Exports', 'Net trade']
    plot_trades = plot_trades.T

    fig, ax = plt.subplots()

    # plot imports and exports
    plot_trades.iloc[:, 0:2].plot(kind='bar', color=['xkcd:cadet blue','k'],
                                       stacked=True, width=1, rot=45, figsize=(20,8),
                                       grid=True, ax=ax, use_index=True, legend=False)

    # plot points for net trade
    plot_trades.iloc[:, 2].plot(kind='line', style='o', markersize=8, mfc='#681E3E', mec='w', alpha=0.8, fontsize=15, grid=True, ax=ax)

    plt.ylabel('Electricity traded, TWh', fontsize=14)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis='y', which='major', labelsize=14)

    # set up secondary axis for trade as % of production
    trade_pct = CFEL['Trade percentage, gross'] * 100
    ax2 = ax.twinx()
    ax2.set_xlim(ax.get_xlim())  # set up secondary y axis plot

    # semi-manually set y-axis extrema to align 0-value with primary y-axis
    ax_neg_pct = abs(ax.get_ylim()[0]) / (abs(ax.get_ylim()[0]) + ax.get_ylim()[1])
    ax_pos_pct = 1 - ax_neg_pct
    # ax2_upper_y = round_up_down(trade_pct.max(), 50, 'up')
    ax2_upper_y =  round_up_down(trade_pct.max(), 100, 'up')

    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    # ax2.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(25))

    # this is just for spacing the 0 in the right place
    ax2.set_ylim(top=(ax2_upper_y), bottom=-(ax2_upper_y*ax_neg_pct) / ax_pos_pct)

    # remove negative y ticks on secondary axis (trade as % of total prod)
    ticks = [tick for tick in plt.gca().get_yticks() if tick >= 0]
    ax2.set_yticks(ticks)

    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False,
                right=True, labelright=True)

    # yticks = ax2.yaxis.get_major_ticks()
    # for t in yticks:
    #     print(t.label2.get_text())
    #     if '-' in t.label2.get_text():
    #         t.label2.set_visible(False)
    #         t.tick1line.set_visible(False)
    #         t.tick2line.set_visible(False)
    ax2.set_ylabel('Gross trade, as percentage of total production (%)', labelpad=5, fontsize=14)
    ax2.set_facecolor('none')

    for _, spine in ax2.spines.items():
        spine.set_visible(False)

    trade_pct.plot(kind='line', style='D', markersize=8, color='#99a63f', mec='k',
        label='Gross trade, as % of total production [right axis]', grid=False, ax=ax2, alpha=0.8, fontsize=13)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    order = [1, 2, 0, 3]  # manually arrange legend entries
    handles = handles + handles2
    labels = labels +labels2
    plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=13, frameon=True, facecolor='w', borderpad=1, loc=4, framealpha=1)

    if export_figures:
        keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure,'Fig_eltrade_' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure,'Fig_eltrade_' + keeper + '.png'), bbox_inches='tight')

    plt.show()


def trade_heatmap(trades):
    """ Extra figure; plot trades as heatmap """
    plt.figure(figsize=(15, 12))
    sns.heatmap(trades.replace(0, np.nan), square=True, linecolor='silver', linewidths=0.5, cmap='inferno_r')