
# coding: utf-8

# # Mapping of production and consumption mixes in Europe and their effect on carbon footprint of electric vehicles

# This code plots the following:
#     -  CF of production vs consumption mixes of each country with visual indication of relative contribution to total European electricity production] [Figure_1]
#     -  imports, exports and net trade [extra figure]
#     -  trade matrix heatmap [extra figure]
#     -  histogram of LDV fleet and ICEV:BEV intensity ratio [Figure 5]
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
#     - road_eqs_carage.xls
# %%

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import country_converter as coco

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import AutoMinorLocator, FixedLocator

import geopandas as gpd
import seaborn as sns

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

# %%
fp = os.path.curdir
fp_data = os.path.join(fp, 'data')


def visualize(experiment, export_figures=True, include_TD_losses=True):
    fp_figure, CFEL, results, ICEV_total_impacts, mapping_data = setup(experiment)
    plot_all(experiment, fp_figure, CFEL, results, ICEV_total_impacts, mapping_data, include_TD_losses, export_figures)


def setup(experiment):
    fp_output = os.path.join(fp, 'output')
    fp_results = os.path.join(fp, 'results')
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
    # results = pickle.load(open(fp_pkl, 'rb'))

    fp_pkl = os.path.join(fp_output,
                          experiment + ' country-specific indirect_BEV_impacts_consumption.pkl')
    BEV_impactsc = pickle.load(open(fp_pkl, 'rb'))

    fp_pkl = os.path.join(fp_output,
                          experiment + ' country-specific indirect_ICEV_impacts.pkl')
    ICEV_total_impacts = pickle.load(open(fp_pkl, 'rb'))

    mapping_data = map_prep(CFEL, results, BEV_impactsc)

    return fp_experiment, CFEL, results, ICEV_total_impacts, mapping_data


def plot_all(exp, fp_figure, CFEL, results, ICEV_total_impacts, mapping_data, include_TD_losses=True, export_figures=False):
    plot_fig1(exp, fp_figure, CFEL, include_TD_losses, export_figures)
    plot_fig2(exp, fp_figure, mapping_data, export_figures)
    plot_fig3(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures)

    try:
        plot_fig4(exp, fp_figure, results, export_figures, orientation='both')
    except Exception as e:
        print(e)

    plot_fig5(exp, fp_figure, results, export_figures)

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

# %% Plot Figure 1

def plot_fig1(exp, fp_figure, CFEL, include_TD_losses, export_figures):
    # Prepare variables for Figure 1
    CFEL_sorted = CFEL.sort_values(by='Production mix intensity')

    # determine net importers and exporters for using colormap; positive if net importer, negative if net exporter
    net_trade = (CFEL_sorted.iloc[:, 1] - CFEL_sorted.iloc[:, 0])

    # for plotting with transmission losses
    net_trade_TL = (CFEL_sorted.iloc[:, 1] - CFEL_sorted.iloc[:, 0]).values #/ 1000

    print("number of importers: " + str(sum(net_trade > 0)))
    print("number of exporters: " + str(sum(net_trade < 0)))


    # Finally, plot Figure 1
    mark_color = 'RdBu_r'
    sns.set_style('whitegrid')

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

    fig1_generator(exp, fp_figure, CFEL_sorted, plot_lim, export_figures,
                   'linear', size_ind, marker_size, mark_color, net_trade_TL, xlim=500, ylim=500)

# %% Figure 1 generator (carbon footprint of electricity production mix vs consumption mix)


def fig1_generator(exp, fp_figure, CFEL_sorted, plot_maxlim, export_figures, axis_type, size_ind, marker_size, marker_clr, net_trade=None, xlim=None, ylim=None):
    """ Generates figure for Figure 1; carbon footprint of production vs consumption
    mixes. Can be plotted on linear, log or semilog axes
    """
    fig, ax = plt.subplots(1, figsize=(16, 12))

    # Control for input axis types linear, semilog, logs
    if axis_type == "log":
        ax.set_yscale(axis_type)
    elif axis_type == "semilog":
        axis_type = "log"
    ax.set_xscale(axis_type)

    ### Finally, plot data
    norm = MidpointNormalize(midpoint=0, vmax=net_trade.max(), vmin=net_trade.min())
    plot = ax.scatter(CFEL_sorted.iloc[:, 2], CFEL_sorted.iloc[:, 3],
                      s=marker_size, alpha=0.5, norm=norm, c=net_trade, cmap=marker_clr, label='_nolegend_')
    ax.scatter(CFEL_sorted.iloc[:, 2], CFEL_sorted.iloc[:, 3],
               s=2, c='k', alpha=0.9, edgecolor='k', label='_nolegend_')  # Include midpoint in figure

    # Hack to have darker marker edges
    ax.scatter(CFEL_sorted.iloc[:, 2], CFEL_sorted.iloc[:, 3],
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
                                    frameon=False, loc=9, bbox_to_anchor=(0.178, 0.990),
                                    bbox_transform=ax.transAxes, prop=dict(size=17, weight='medium'))
    elif size_ind == 'production':
        legend_title = AnchoredText('% of total European production',
                                    frameon=False, loc=9, bbox_to_anchor=(0.174, 0.990),
                                    bbox_transform=ax.transAxes, prop=dict(size=17, weight='medium'))

    add_year = AnchoredText('2019', frameon=False, loc=9, bbox_to_anchor=(0.18, 0.961),
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
                    fc=[0.0196078431372549, 0.18823529411764706, 0.3803921568627451, 1.0],
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

    # 10%, 20%, 50% shading and x=y line
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], color="grey", alpha=0.6)
    plt.fill_between([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1] * 1.1],
                     [0, ax.get_xlim()[1] * 0.9], color="grey", alpha=0.13)
    plt.fill_between([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1] * 1.2],
                     [0, ax.get_xlim()[1] * 0.8], color="grey", alpha=0.1)
    plt.fill_between([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1] * 1.5],
                     [0, ax.get_xlim()[1] * 0.5], color='grey', alpha=0.07)
    ax.annotate(r'$\pm$ 10%', xy=(0, 0), xytext=(0.881, 0.952), xycoords='axes fraction', fontsize=13, rotation=43)
    ax.annotate(r'$\pm$ 20%', xy=(0, 0), xytext=(0.808, 0.952), xycoords='axes fraction', fontsize=13, rotation=43)
    ax.annotate(r'$\pm$ 50%', xy=(0, 0), xytext=(0.64, 0.952), xycoords='axes fraction', fontsize=13, rotation=43)
    ax.annotate(r'x = y', xy=(0, 0), xytext=(0.954, 0.962), xycoords='axes fraction', fontsize=13, rotation=45)

    plt.xlabel("Carbon footprint of production mix \n (g CO$_2$ kWh$^{-1}$)", fontsize=20, labelpad=14)
    plt.ylabel("Carbon footprint of consumption mix \n (g CO$_2$ kWh$^{-1}$)", fontsize=20, labelpad=14)

    # Make colorbar
    ax_cbar = fig.add_axes([0.914, 0.2, 0.02, 0.78])  # place colorbar on own Ax
    ax_cbar.margins(tight=True)

    # Calculate minimum and maximum labels for colorbar (rounded to nearest 5 within the scale)
    cbar_min = 5 * round(net_trade.min() / 5)
    cbar_max = 5 * round((net_trade.max() // 1) / 5)
    if cbar_max > net_trade.max():  # if maximum is off the legend, round down
        cbar_max = cbar_max - 5

    # Begin plotting colorbar
    cbar = plt.colorbar(plot, cax=ax_cbar, ticks=[cbar_min, 0, cbar_max], drawedges=False)
    # cbar = plt.colorbar(main_plot.get_cmap(), norm=norm, cax=ax_cbar, drawedges=False)
    cbar.set_label('Net electricity traded, TWh yr$^{-1}$', fontsize=18, rotation=90, labelpad=8)
    cbar.set_alpha(1)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(5)
    cbar.outline.set_color('k')
    cbar.draw_all()

    # Manually tack on semi-transparent rectangle on colorbar to match alpha of plot; workaround for weird rendering of colorbar with alpha
    r1 = Rectangle((0, 0), 30, 500, fc='w', ec='k', alpha=0.3)
    ax_cbar.add_patch(r1)

    # Make and format inset figure
    ax2 = fig.add_subplot(339)  # for inlay in bottom right position inset subplot
    ax2.axis([400, xlim, 400, ylim])

    # Set formatting for zoomed inlay figure

    # markersize = 40, s = 40*2, pi/4*s = marker area
    # Linear factor between main fig and inset: x1/x2 = z*(l1/l2) --> 1100/150 = 33.75/10.85*x --> x=2.357
    marker_size_ratio = (ax.get_xlim()[1] / (ax2.get_xlim()[1] - ax2.get_xlim()[0]))

    ax2.scatter(CFEL_sorted.iloc[:, 2], CFEL_sorted.iloc[:, 3],
                s=marker_size * marker_size_ratio * (np.pi / 4), alpha=0.5,
                norm=norm, c=net_trade, cmap=marker_clr, edgecolor='k', label='_nolegend_')
    ax2.scatter(CFEL_sorted.iloc[:, 2], CFEL_sorted.iloc[:, 3],
                s=marker_size * marker_size_ratio * (np.pi / 4), alpha=0.9,
                norm=norm, c="None", edgecolor='k', linewidths=0.7, label='_nolegend_')  # Hack for darker edge colours
    ax2.scatter(CFEL_sorted.iloc[:, 2], CFEL_sorted.iloc[:, 3],
                s=2, c='k', alpha=0.9, edgecolor='k', label='_nolegend_')  # Include midpoint in figure
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.tick_params(which="major", labelsize=12)
    ax2.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], color="grey", alpha=0.6)
    plt.fill_between([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1] * 1.1],
                     [0, ax2.get_xlim()[1] * 0.9], color="grey", alpha=0.13)
    plt.fill_between([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1] * 1.2],
                     [0, ax2.get_xlim()[1] * 0.8], color="grey", alpha=0.1)
    plt.fill_between([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1] * 1.5],
                     [0, ax2.get_xlim()[1] * 0.5], color='grey', alpha=0.07)

    ### Add country text labels
    for (country, country_data) in CFEL_sorted.iterrows():
        # Inset figure labelling
        if country_data["Production mix intensity"] >= (ax2.get_xlim()[0]*0.9) and country_data["Production mix intensity"] <= (ax2.get_xlim()[1]*1.1) and country_data["Consumption mix intensity"] >= (ax2.get_ylim()[0]*0.9) and country_data["Consumption mix intensity"] <= (ax2.get_ylim()[1]*1.1): #Inset labels
            if country in ['DE', 'IT', 'NL']:
                pass  # ignore the largest circles in the inset, as they don't need labelling
            else:
                ax2.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                             xytext=(np.sqrt(marker_size[country]*marker_size_ratio*(np.pi/4))/2+6,-5),
                             textcoords=("offset points"), size=15,
                             path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)])

        # Left corner of main figure
        if country_data["Production mix intensity"] <= 100 and country_data["Consumption mix intensity"] <= 100:
            # Adjust for countries in bottom left
            if country == "SE":
                ax.annotate(country, xy=(country_data["Production mix intensity"],
                                         country_data["Consumption mix intensity"]),
                            xytext=(np.sqrt(marker_size[country]) / 2 - 2, -2), textcoords=("offset points"),
                            path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)
            elif country == "NO":
                ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                            xytext=(np.sqrt(marker_size[country]) / 2 - 2, -11),
                            textcoords=("offset points"),
                            path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)
            elif country == 'FR':
                ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                            xytext=(np.sqrt(marker_size[country]) / 2 - 2, -38), textcoords=("offset points"),
                            path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)

        # Rest of figure; avoid overlapping labels
        else:
            ax.annotate(country, xy=(country_data["Production mix intensity"], country_data["Consumption mix intensity"]),
                        xytext=(-9.5, -12 - np.sqrt(marker_size[country]) / 2), textcoords=("offset points"),
                        path_effects=[pe.withStroke(linewidth=4, foreground="w", alpha=0.8)], size=15)

    plt.tight_layout()
    plt.show()

    if export_figures:
        keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure, 'Fig_1 - ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, 'Fig_1 - ' + keeper + '.png'), bbox_inches='tight')

    plt.show()


# %% Figure 5 ((histogram with fleet size vs ICEV:BEV ratio))

def plot_fig5(exp, fp_figure, results, export_figures):
    # ## Trial: Make figure comparing car fleet and energy consumption
    # Load car fleet data from Eurostat
    fleet_fp = os.path.join(fp_data, 'road_eqs_carage.xls')
    car_fleet = pd.read_excel(fleet_fp, skiprows=[0,1,2,3,4,5,6,7,8], header=[0], index_col=0, nrows=34)

    car_fleet.replace(':', value=np.nan, inplace=True)
    car_fleet.index = coco.CountryConverter().convert(names=car_fleet.index.tolist(), to="ISO3")

    # Prepare the data
    y_axis = 1 / results['RATIO BEV:ICEV - Segment C - Consumption mix']
    x_axis = car_fleet['2016']

    x_ind = list(x_axis.index)
    x_axis.index = pd.Index(coco.CountryConverter().convert(x_ind, to="ISO2"))

    histogram = pd.concat([y_axis, x_axis], axis=1)
    histogram.columns = ['y', 'x']
    histogram.dropna(how='any', axis=0, inplace=True)
    histogram.sort_values(by='y', ascending=False, axis=0, inplace=True)

    bin_x = np.cumsum(histogram['x'].tolist())
    bin_x = np.insert(bin_x, 0, [0]) / 1e6  # Change units of vehicles to millions

    # Plot Figure 5
    txtsize = 14
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16, 12))
    ax = plt.subplot(111)

    widths = bin_x[1:] - bin_x[:-1]
    bin_centre = bin_x[:-1] + (widths / 2)
    histo_fig = plt.bar(bin_centre, histogram['y'], width=widths, alpha=1, edgecolor='darkslategrey', color='k', zorder=20)  # xkcd:light grey

    # Fill shading for ICEV:BEV ratio of 1 and 1.25, respectively
    ax.fill_between([0, 270], [0, 0], [1, 1], facecolor='darkred', alpha=0.3, hatch='...', edgecolor='firebrick', zorder=1)
    ax.fill_between([0, 270], [1, 1], [1.25, 1.25], facecolor='grey', alpha=0.3, hatch='...', edgecolor='w', zorder=1)

    pos_dict = {'BG': (10, 21), 'CH': (-8, 3), 'CZ': (-10, 3),
                'DK': (-3, 3), 'EE': (8, 21), 'FI': (-8, 3),
                'GR': (-5, 3), 'HU': (15, 30), 'IE': (8, 20),
                'LT': (11, 20), 'LV': (16, 15), 'MK': (11, 20),
                'NO': (-8, 3), 'SE': (-8, 3), 'SI': (18, 30), 'SK': (11, 25)}

    no_arrow = ['NO', 'SE', 'FI', 'CH', 'CZ', 'GR', 'DK']
    for i, value in enumerate(histogram.index.tolist()):
        if value in list(pos_dict.keys()):
            if value in no_arrow:
                ax.annotate(value, xy=(bin_x[i+1], histogram['y'].loc[value]), xytext=pos_dict[value], textcoords='offset points', size=txtsize, ha='left')
            else:
                ax.annotate(value, xy=(bin_x[i+1], histogram['y'].loc[value]), xytext=pos_dict[value], textcoords='offset points', size=txtsize,
                            arrowprops=dict(color='k', arrowstyle='->', lw='0.5', relpos=(0, 0.5), shrinkA=1, shrinkB=0),
                            bbox=dict(pad=0, facecolor='none', ec='none'), va='center', ha='left')
        # elif value in ['NO', 'SE', 'FI', 'CH']:
        #     ax.annotate(value, xy=(bin_x[i+1], histogram['y'].loc[value]), xytext=(-8, 3), textcoords='offset points', size=txtsize, ha='left')
        else:
            ax.annotate(value, xy=(bin_x[i+1], histogram['y'].loc[value]), xytext=(-15, 3), textcoords='offset points', size=txtsize, ha='left')

    # Customize histogram design
    plt.xlabel('Cumulative LDV fleet, millions', fontsize=16, labelpad=11)
    plt.ylabel('ICEV:BEV lifecycle carbon intensity ratio, C-segment with consumption mix', fontsize=16, labelpad=11)

    plt.xlim(0, 270)

    plt.tick_params(which='minor', direction='out', length=4.0)
    plt.tick_params(which='major', direction='out', length=6.0, labelsize=16)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Highlight top 5 countries for greatest policy impact
    area = histogram['x'] / 1e6 * histogram['y']
    top_x = area.nlargest(6)
    threshold = top_x[-1]

    for patch in histo_fig:
        patch_area = patch.get_width() * patch.get_height()
        if patch_area >= threshold:
            patch.set_facecolor('xkcd:cadet blue')

    if export_figures:
        keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure, 'Fig_5 - ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, 'Fig_5 - ' + keeper + '.png'), bbox_inches='tight')

    plt.show()

# %% Figure 4 (differences for domestic production)
def plot_fig4(exp, fp_figure, results, export_figures, orientation='both'):
    A_diff = (results['BEV footprint, EUR production - Segment A - Consumption mix'] -
              results['BEV footprint - Segment A - Consumption mix']) / results['BEV footprint - Segment A - Consumption mix']
    F_diff = (results['BEV footprint, EUR production - Segment F - Consumption mix'] -
              results['BEV footprint - Segment F - Consumption mix']) / results['BEV footprint - Segment F - Consumption mix']
    fig_data = pd.DataFrame([A_diff, F_diff], index=['A segment', 'F segment'])
    fig_data = fig_data.T

    sorted_fig_data = fig_data.iloc[::-1].sort_values(by='F segment')
    fig4_cmap = colors.ListedColormap(['xkcd:cadet blue', 'k'])

    if orientation == 'horizontal' or 'both':
        # Horizontal variation of Figure 4
        ax = sorted_fig_data.iloc[::-1].plot(kind='barh', figsize=(16, 12), position=0.5, stacked=True,
                                             cmap=fig4_cmap, width=0.95)
        ax.set_xticklabels(['{:.0%}'.format(x) for x in ax.get_xticks()])
        ax.set_xlabel('Difference in BEV carbon intensity by shifting to domestic battery production', fontsize=16, labelpad=12)

        ax.yaxis.tick_right()
        ax.tick_params(axis='y', direction='out', color='k')
        ax.tick_params(which='major', length=4.0, labelsize=16, pad=10)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.grid(b=True, which='minor', axis='x', linestyle='-', color='whitesmoke', alpha=0.9)
        ax.grid(b=True, which='major', axis='x', linewidth=2.5)
        ax.legend(['A segment (26.6 kWh battery)', 'F segment (89.8 kWh battery)'], loc=3, facecolor='white', edgecolor='k', fontsize=18, frameon=True, borderpad=1)


        if export_figures:
            keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
            plt.savefig(os.path.join(fp_figure, 'Fig_4_horizontal ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(fp_figure, 'Fig_4_horizontal ' + keeper + '.png'), bbox_inches='tight')

        plt.show()

    if orientation == 'vertical' or 'both':
        # Vertical variation of Figure 4
        ax = sorted_fig_data.plot(kind='bar', figsize=(16, 12), position=0.5, stacked=True,
                                  cmap=fig4_cmap, alpha=0.7, rot=50, width=0.95)
        ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
        ax.tick_params(axis='x', direction='out', color='k')
        ax.legend(facecolor='white', edgecolor='k', fontsize='large', frameon=True)
        ax.set_ylabel('Difference in BEV carbon intensity by shifting to domestic battery production', labelpad=12)


        if export_figures:
            keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
            plt.savefig(os.path.join(fp_figure, 'Fig_4 ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
            plt.savefig(os.path.join(fp_figure, 'Fig_4 ' + keeper), bbox_inches='tight')
    else:
        raise Exception('Invalid orientation for Figure 4')

    plt.show()

# %% map_prep

# # Begin mapping
def map_prep(CFEL, results, BEV_impactsc):
    # Load map shapefile; file from Natural Earth
    fp_map = os.path.join(fp_data, 'maps', 'ne_10m_admin_0_countries.shp')
    country_shapes = gpd.read_file(fp_map)
    europe_shapes = country_shapes[(country_shapes.CONTINENT == "Europe")]  # restrict to Europe
    europe_shapes.loc[:, 'ISO_A2'] = coco.CountryConverter().convert(names=list(europe_shapes['ADM0_A3'].values), to='ISO2')  # convert to ISO A2 format for joining with data

    # Join results to Europe shapefile and drop irrelevant countries (i.e., those without results)
    li = results.columns.tolist()
    # segment_list = results.columns.get_level_values(1).tolist()

    # Create nicer header names for calculated results
    col = pd.Index([e[0] + " - Segment " + e[1] + " - " + e[2] for e in li])
    results.columns = col
    # col_names = results.columns
    # last_label = results.columns[-1]

    # Combined electricity footprint and vehicle footprint results
    all_data = pd.concat([CFEL, results, BEV_impactsc.T], axis=1)

    # Add to mapping shapes
    mapping_data = europe_shapes.join(all_data, on="ISO_A2")

    return mapping_data

# %%  Helper functions for plotting

# function to round to nearest given multiple
def round_up_down(number, multiple, direction):
    if direction == "up":
        return int(number + (multiple - (number % multiple)))
    elif direction == "down":
        return int(number - (number % multiple))
    else:
        print("Incorrect direction")

# truncate colormap to avoid intense black/white areas
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# function for annotating maps

fp_label_pos = os.path.join(fp_data, 'label_pos.csv')
label_pos = pd.read_csv(fp_label_pos, index_col=[0, 1], skiprows=0, header=0)  # read in coordinates for label/annotation positions
lined_countries = ['PT', 'BE', 'NL', 'DK', 'SI', 'GR', 'ME', 'MK', 'EE', 'LV', 'BA', 'CH']  # countries using leader lines

def annotate_map(ax, countries, mapping_data, values, cmap_range, threshold=0.8, round_dig=1, fontsize=7.5):
    for loc, label in zip(countries, values):
        if round_dig == 0:
            label = int(label)
        if (label / cmap_range) >= threshold:
            if label_pos.loc[loc].index.isin(lined_countries):
                lc = 'w'
                color = 'k'
            else:
                color = 'w'
        else:
            lc = 'k'
            color = 'k'

        x = label_pos.loc[loc].x
        y = label_pos.loc[loc].y
        x_cent = mapping_data.loc[loc].geometry.centroid.x
        y_cent = mapping_data.loc[loc].geometry.centroid.y
        if label_pos.loc[loc].index.isin(['DK']): # modified start point for pointer line
            ax.annotate(round(label, round_dig),
                        xy=(x_cent - 0.75, y_cent),
                        xytext=(x, y),
                        textcoords='data', size=fontsize, va='center',
                        bbox=dict(pad=0.03, facecolor="none", edgecolor="none"),
                        arrowprops=dict(color=color, arrowstyle='-', lw='0.45', shrinkA=1.5, shrinkB=0, relpos=(1, 0.5)))
        elif label_pos.loc[loc].index.isin(['LV', 'EE']):
            ax.annotate(round(label, round_dig),
                        xy=(x_cent+1.3, y_cent),
                        xytext=(x, y),
                        textcoords='data', size=fontsize, va='center', ha='center',
                        bbox=dict(pad=0.05, facecolor="none",edgecolor='none'),
                        arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=1.5, shrinkB=0))
        elif label_pos.loc[loc].index.isin(['BE']):
            ax.annotate(round(label, round_dig),
                        xy=(x_cent, y_cent),
                        xytext=(x, y),
                        textcoords='data', size=fontsize, va='center', ha='center',
                        bbox=dict(pad=0.02, facecolor="none", edgecolor='none'),
                        arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=-0.2, shrinkB=0, relpos=(0.55, 0)))
        elif label_pos.loc[loc].index.isin(['MK','NL','PT','SK','LV','EE']):
            ax.annotate(round(label, round_dig),
                        xy=(x_cent, y_cent),
                        xytext=(x, y),
                        textcoords='data', size=fontsize, va='center', ha='center',
                        bbox=dict(pad=0.05, facecolor="none", edgecolor='none'),
                        arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=0.8, shrinkB=0))
        elif label_pos.loc[loc].index.isin(['GR']): # modified start point for pointer line
            ax.annotate(round(label, round_dig),
                        xy=(x_cent-1.5, y_cent+0.9),
                        xytext=(x, y),
                        textcoords='data', size=fontsize, va='bottom', ha='center',
                        bbox=dict(pad=0.05, facecolor="none", edgecolor="none"),
                        arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=1.5, shrinkB=0))
        elif label_pos.loc[loc].index.isin(['LT','CZ','IE','NL','HU','AU','BG','RS']): # countries requiring smaller annotations
            ax.annotate(round(label, round_dig),
                        xy=(x, y),
                        textcoords='data', size=fontsize-0.5, color=color, ha='center')
        elif label_pos.loc[loc].index.isin(['SI', 'ME', 'CH', 'BA']): # countries requiring smaller annotations
            ax.annotate(round(label, round_dig),
                        xy=(x_cent, y_cent),
                        xytext=(x, y),
                        textcoords='data', size=fontsize-0.25, color=color, va='center', ha='center',
                        bbox=dict(boxstyle='round, pad=0.1', pad=0.05, facecolor="w", edgecolor="none", alpha=0.75),
                        arrowprops=dict(color=lc, arrowstyle='-', lw='0.5', shrinkA=0.5, shrinkB=0))
        else:
            ax.annotate(round(label,round_dig), xy=(x, y),
                        textcoords='data', size=fontsize, color=color, ha='center')

# %% Figure 2


def plot_fig2(exp, fp_figure, mapping_data, export_figures):
    # # Make Figure 2 as single figure with subplots
    sns.set_style('dark')
    vmin = 50
    vmax = 375
    threshold = 0.40 # threshold for switching annotation colours

    # cmap = plt.get_cmap('cmr.savanna_r', 7)
    cmap = plt.get_cmap('viridis_r',6)

    cmap_col = [cmap(i) for i in np.linspace(0, 1, 6)]  # retrieve colormap colors
    cmap = cmap_col
    # cmap = [cmap_col[i] for i in np.arange(0, len(cmap_col)) if not i%2]#cmap_col[0:3]+[cmap_col[4]]+[cmap_col[6]]#[cmap_col[0]]+[cmap_col[2]]+cmap_col[3:5]+[cmap_col[6]]

    # cmap_BEV = plt.get_cmap('viridis_r',9)

    # Make manual boundaries for cmap
    # range of negative values approximately 1/3 of that of positive values;
    # cut-off colormap for visual 'equality'

    boundaries = [i for i in np.arange(vmin, vmax, 50)]  # define boundaries of colormap transitions
    cmap_BEV, norm = colors.from_levels_and_colors(boundaries, colors=[cmap[0]]+ cmap + [cmap_col[-1]], extend='both')

    max_fp = max(mapping_data['BEV footprint - Segment A - Production mix'].max(),
                 mapping_data['BEV footprint - Segment A - Consumption mix'].max(),
                 mapping_data['BEV footprint - Segment F - Production mix'].max(),
                 mapping_data['BEV footprint - Segment F - Consumption mix'].max())


    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, squeeze=True, gridspec_kw={'wspace': 0.03, 'hspace': 0.03}, figsize=(11.5, 7), dpi=600)
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.01, hspace=0.01)

    # Plot maps
    mapping_data[mapping_data['BEV footprint - Segment A - Production mix'].isna()].plot(ax=ax[0, 0], color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
    mapping_data[mapping_data['BEV footprint - Segment A - Production mix'].notna()].plot(ax=ax[0, 0], column='BEV footprint - Segment A - Production mix', cmap=cmap_BEV, vmax=vmax, vmin=vmin, norm=norm, edgecolor='k', linewidth=0.3)

    mapping_data[mapping_data['BEV footprint - Segment A - Consumption mix'].isna()].plot(ax=ax[1, 0], color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
    mapping_data[mapping_data['BEV footprint - Segment A - Consumption mix'].notna()].plot(ax=ax[1, 0], column='BEV footprint - Segment A - Consumption mix', cmap=cmap_BEV, vmax=vmax, vmin=vmin, norm=norm, edgecolor='k', linewidth=0.3)

    mapping_data[mapping_data['BEV footprint - Segment F - Production mix'].isna()].plot(ax=ax[0, 1], color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
    mapping_data[mapping_data['BEV footprint - Segment F - Production mix'].notna()].plot(ax=ax[0, 1], column='BEV footprint - Segment F - Production mix', cmap=cmap_BEV, vmax=vmax, vmin=vmin, norm=norm, edgecolor='k', linewidth=0.3)

    mapping_data[mapping_data['BEV footprint - Segment F - Consumption mix'].isna()].plot(ax=ax[1, 1], color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
    mapping_data[mapping_data['BEV footprint - Segment F - Consumption mix'].notna()].plot(ax=ax[1, 1], column='BEV footprint - Segment F - Consumption mix', cmap=cmap_BEV, vmax=vmax, vmin=vmin, norm=norm, edgecolor='k', linewidth=0.3)

    # Annotate with values
    annotate_map(ax[0, 0], mapping_data[mapping_data['BEV footprint - Segment A - Production mix'].notna()].index.to_list(), mapping_data,
                 mapping_data[mapping_data['BEV footprint - Segment A - Production mix'].notna()]
                 ['BEV footprint - Segment A - Production mix'].values, max_fp, threshold=threshold, round_dig=0)
    annotate_map(ax[1, 0], mapping_data[mapping_data['BEV footprint - Segment A - Consumption mix'].notna()].index.to_list(), mapping_data,
                 mapping_data[mapping_data['BEV footprint - Segment A - Consumption mix'].notna()]
                 ['BEV footprint - Segment A - Consumption mix'].values, max_fp, threshold=threshold, round_dig=0)
    annotate_map(ax[0, 1], mapping_data[mapping_data['BEV footprint - Segment F - Production mix'].notna()].index.to_list(), mapping_data,
                 mapping_data[mapping_data['BEV footprint - Segment F - Production mix'].notna()]
                 ['BEV footprint - Segment F - Production mix'].values, max_fp, threshold=threshold, round_dig=0)
    annotate_map(ax[1, 1], mapping_data[mapping_data['BEV footprint - Segment F - Consumption mix'].notna()].index.to_list(), mapping_data,
                 mapping_data[mapping_data['BEV footprint - Segment F - Consumption mix'].notna()]
                 ['BEV footprint - Segment F - Consumption mix'].values, max_fp, threshold=threshold, round_dig=0)

    # Label axes
    ax[0, 0].set_ylabel('Production mix', fontsize=13.5)
    ax[1, 0].set_ylabel('Consumption mix', fontsize=13.5)
    ax[0, 0].set_title('A-segment (mini)', fontsize=13.5)
    ax[0, 1].set_title('F-segment (luxury)', fontsize=13.5)

    plt.xlim((-12, 34))
    plt.ylim((32, 75))

    plt.yticks([])
    plt.xticks([])

    sns.reset_orig()
    cb = plt.cm.ScalarMappable(cmap=cmap_BEV, norm=norm)
    cb.set_array([])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.815, 0.13, 0.025, 0.75])
    cbar = fig.colorbar(cb, cax=cbar_ax, extend='both')
    cbar.set_label('Lifecycle BEV carbon intensity, \n g CO$_2$ eq/km', rotation=90, labelpad=9, fontsize=12)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    # hack to remove minor ticks on colorbar extending past min/max values (bug in matplotlib?)
    minorticks = cbar.ax.get_yticks(minor=True)
    minorticks = [i for i in minorticks if (i >= 0 and i <= 1)]
    cbar.ax.yaxis.set_ticks(minorticks, minor=True)

    cbar.ax.tick_params(labelsize=9, pad=4)


    if export_figures:
        keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure, 'Fig_2 ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, 'Fig_2 ' + keeper), bbox_inches='tight')

    plt.show()

# %% Figure 3
def plot_fig3(exp, fp_figure, mapping_data, ICEV_total_impacts, export_figures):
    # # Make multiple versions of Figure 3
    #
    # First, with A-segment ratios and the difference of A- and F-segments (delta)
    # Second, with A-segment and F-segment ratios (original Figure 3) + separate delta figure
    # Third, with absolute difference between BEV and ICEV for both panels

    sns.set_style('dark')
    mapping_data['abs_diff_A'] = ICEV_total_impacts['A'].reindex_like(mapping_data, method='pad').subtract(mapping_data['A'])
    mapping_data['abs_diff_C'] = ICEV_total_impacts['C'].reindex_like(mapping_data, method='pad').subtract(mapping_data['C'])
    mapping_data['abs_diff_D'] = ICEV_total_impacts['D'].reindex_like(mapping_data, method='pad').subtract(mapping_data['D'])
    mapping_data['abs_diff_F'] = ICEV_total_impacts['F'].reindex_like(mapping_data, method='pad').subtract(mapping_data['F'])

    abs_diff = mapping_data[mapping_data['abs_diff_A'].notna()]
    abs_diff.set_index('ADM0_A3', inplace=True)
    abs_diff = abs_diff.iloc[:, -4:]

    rmax = max(mapping_data['abs_diff_A'].max(),
               mapping_data['abs_diff_C'].max(),
               mapping_data['abs_diff_D'].max(),
               mapping_data['abs_diff_F'].max()) - 2

    rmin = min((mapping_data['abs_diff_A'].min(),
                mapping_data['abs_diff_C'].min(),
                mapping_data['abs_diff_D'].min(),
                mapping_data['abs_diff_F'].min())) + 2

    N = 7  # Number of sections from full colormap; must be odd number
    cmap_diff = plt.get_cmap('RdYlGn', N)
    cmap_col = [cmap_diff(i) for i in np.linspace(0, 1, N)]  # retrieve colormap colors

    # Make manual boundaries for cmap
    # range of negative values approximately 1/3 of that of positive values;
    # cut-off colormap for visual 'equality'
    cutoff = int(int(N/2) - ((np.abs(rmin) - 0) / (rmax - 0)) * int(N/2))
    cmap = cmap_col[2:]  # "trim" bottom section of colormap colors
    upper_bound = [i for i in np.linspace(2.5, int(rmax), 4)]
    boundaries = [rmin, -2.5] + upper_bound  # define boundaries of colormap transitions

    cmap_colors, norm = colors.from_levels_and_colors(boundaries, colors=[cmap[0]] + cmap + [cmap[-1]], extend='both')

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, squeeze=True,
                           gridspec_kw={'wspace': 0.03, 'hspace': 0.03}, figsize=(11.5, 7), dpi=600)
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.01, hspace=0.01)

    # Plot maps; start with countries not included, then countries with values
    mapping_data[mapping_data['abs_diff_A'].isna()].plot(ax=ax[0, 0], color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
    mapping_data[mapping_data['abs_diff_A'].notna()].plot(ax=ax[0, 0], column='abs_diff_A', cmap=cmap_colors, edgecolor='k', linewidth=0.3, norm=norm, vmax=rmax, vmin=rmin)

    mapping_data[mapping_data['abs_diff_C'].isna()].plot(ax=ax[0, 1], color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
    mapping_data[mapping_data['abs_diff_C'].notna()].plot(ax=ax[0, 1], column='abs_diff_C', cmap=cmap_colors, edgecolor='k', linewidth=0.3, norm=norm, vmax=rmax, vmin=rmin)

    mapping_data[mapping_data['abs_diff_D'].isna()].plot(ax=ax[1, 0], color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
    mapping_data[mapping_data['abs_diff_D'].notna()].plot(ax=ax[1, 0], column='abs_diff_D', cmap=cmap_colors, edgecolor='k', linewidth=0.3, norm=norm, vmax=rmax, vmin=rmin)

    mapping_data[mapping_data['abs_diff_F'].isna()].plot(ax=ax[1, 1], color='lightgrey', edgecolor='darkgrey', linewidth=0.3)
    mapping_data[mapping_data['abs_diff_F'].notna()].plot(ax=ax[1, 1], column='abs_diff_F', cmap=cmap_colors, edgecolor='k', linewidth=0.3, norm=norm, vmax=rmax, vmin=rmin)

    # Annotate map with values
    threshold = boundaries[-2] / rmax  # threshold value to determine annotation text color

    annotate_map(ax[0, 0],
                 mapping_data[mapping_data['abs_diff_A'].notna()].index.to_list(), mapping_data,
                 mapping_data[mapping_data['abs_diff_A'].notna()].abs_diff_A.values,
                 max(rmax, np.abs(rmin)), threshold=threshold)
    annotate_map(ax[0, 1],
                 mapping_data[mapping_data['abs_diff_C'].notna()].index.to_list(), mapping_data,
                 mapping_data[mapping_data['abs_diff_C'].notna()].abs_diff_C.values,
                 max(rmax, np.abs(rmin)), threshold=threshold)
    annotate_map(ax[1, 0],
                 mapping_data[mapping_data['abs_diff_D'].notna()].index.to_list(), mapping_data,
                 mapping_data[mapping_data['abs_diff_D'].notna()].abs_diff_D.values,
                 max(rmax, np.abs(rmin)), threshold=threshold)
    annotate_map(ax[1, 1],
                 mapping_data[mapping_data['abs_diff_F'].notna()].index.to_list(), mapping_data,
                 mapping_data[mapping_data['abs_diff_F'].notna()].abs_diff_F.values,
                 max(rmax, np.abs(rmin)), threshold=threshold)

    # Label panels
    captions = ['(a) A-segment (mini)', '(b) C-segment (medium)',
                '(c) D-segment (large)', '(d) F-segment (luxury)']

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

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.815, 0.13, 0.025, 0.75])
    cbar = fig.colorbar(cb, cax=cbar_ax, extend='both', spacing='proportional', ticks=[-5, 0, 5, 10, 15, 20, 25, 30])
    cbar.set_label('Lifecycle $CO_2$ mitigated through electrification, \n t $CO_2$-eq/vehicle', rotation=90, labelpad=9, fontsize=12)
    minorticks = np.arange(int(cbar.vmin), int(cbar.vmax) + 1, 1)
    minorticks = (minorticks - cbar.vmin) / (cbar.vmax - cbar.vmin)  # normalize minor ticks to [0,1] scale
    cbar.ax.yaxis.set_minor_locator(FixedLocator(minorticks))
    cbar.ax.tick_params(labelsize=9, pad=4)

    if export_figures:
        keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure, 'Fig 3- abs_diff ' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure, 'Fig 3- abs_diff ' + keeper), bbox_inches='tight')

    plt.show()

#%% Optional figures for plotting

def plot_el_trade(exp, total_imported, total_exported, net_trade, export_figures):
    """ plot import, exports and net trade for each country """

    plot_trades = pd.DataFrame([total_imported, -total_exported, total_imported - total_exported, net_trade])
    plot_trades.index = ['Imports', 'Exports', 'Net trade', 'old net trade']
    plot_trades = plot_trades.T

    sns.set_style('darkgrid')

    ax = plot_trades.iloc[:, 0:2].plot(kind='bar', color=['xkcd:cadet blue','k'],
                                       stacked=True, width=1, rot=45, figsize=(20,8),
                                       grid=True, use_index=True)

    ax2 = plot_trades.iloc[:, 2].plot(kind='line', style='.', markersize=15, color='r', rot=45, fontsize=15, legend=True, grid=True)
    plt.ylabel('Electricity traded, TWh', fontsize=14)
    plt.legend(fontsize=14, frameon=True, facecolor='w', borderpad=1)

    if export_figures:
        keeper = exp + " run {:%d-%m-%y, %H_%M}".format(datetime.now())
        plt.savefig(os.path.join(fp_figure,'Fig_eltrade' + keeper + '.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(fp_figure,'Fig_eltrade' + keeper + '.png'), bbox_inches='tight')

    plt.show()


def trade_heatmap(trades):
    """ Extra figure; plot trades as heatmap """
    plt.figure(figsize=(15, 12))
    sns.heatmap(trades.replace(0, np.nan), square=True, linecolor='silver', linewidths=0.5, cmap='inferno_r')


# %%

# All figures plotted at once

#def make_figures(map_data, col_plot, legend_title, min_value, max_value, lbl_div_size, cmap, title, midpoint="None", num_clr_divs="None"):
#    if cmap=='RdBu_r':
#        cmap = cm.get_cmap(cmap,11)
#        newcmp = cmap
#        newcmp.set_bad('white')
#        map_data[col_plot] = map_data[col_plot].apply(lambda x: np.ma.masked_inside(x, 0.9,1.1))
#    else:
#        newcmp=cm.get_cmap(cmap,11)
#
## Calculate midpoint for non-normalized scales
#    if midpoint=="None":
#        midpoint = min_value + (max_value-min_value)/2 #midpoint is actually halfway through the range
#    if num_clr_divs == "None":
#        num_clr_divs = (max_value-min_value)/lbl_div_size
#
##Plot data
#    ax = map_data.plot(column=col_plot, figsize=(16,12),cmap=newcmp, norm=MidpointNormalize(midpoint=midpoint, vmax=max_value, vmin=min_value),edgecolor='k')
#    ax.set_title(title, fontsize=20, pad=20)
#    ax.grid(False)
#    fig = ax.get_figure()
#    ax.set_yticklabels([])
#    ax.set_xticklabels([])
#
## Colorbar/legend setup
#    ax = plt.cm.ScalarMappable(cmap=newcmp, norm=MidpointNormalize(midpoint=midpoint, vmax=max_value, vmin=min_value))
#    ax._A = []
#    cbar = fig.colorbar(ax, extend='both')
#    cbar.ax.set_ylabel(legend_title, rotation=90, labelpad=10)
#
#    plt.xlim((-12,34))
#    plt.ylim((32,75))
#
#    if export_figures == True:
#        keeper = " run {:%d-%m-%y, %H_%M}".format(datetime.now())
#        if ":" in col_plot:
#            col_plot = col_plot.replace(":", "-")
#
#        plt.savefig(os.path.join(fp_figure,col_plot+keeper+'.pdf'), format='pdf', bbox_inches='tight')
#        plt.savefig(os.path.join(fp_figure,col_plot+keeper), bbox_inches='tight')


# %%

# Print map-a-palooza
#
#"""RdYlGn_r"""""
#max_BEV = round_up_down(results.values.max(), 10, "up")
#min_BEV = round_up_down(results[results.columns[0:7]].values.min(),10,"down")
#max_ratio = round_up_down(results[results.columns[8:15]].values.max()*10,5,"up")/10
#min_ratio = round_up_down(results[results.columns[8:15]].values.min()*10,5,"down")/10
#min_prod = round_up_down(BEV_prod_EU.values.min(),2,"down")
#max_prod = round_up_down(BEV_prod_EU.values.max(),2,"up")
#
#for i, name in enumerate(col_names):
#    if "RATIO" in name:
#        #cmap = colors.LinearSegmentedColormap.from_list("", ["green","white","#FFFF00","orange","red","#8B0000","black"],15)
#        make_figures(mapping_data,name,"Ratio, BEV-ICEV",min_ratio,max_ratio,0.2,'RdBu_r',"Ratio carbon intensity, BEV:ICEV, Segment "+segment_list[i],midpoint=1, num_clr_divs=11)
#        #keeper = " run {:%d-%m-%y, %H_%M}".format(datetime.now())
#        #if export_data==True:
#        #    plt.savefig(name+keeper, bbox_inches='tight')
#      #  make_figures(min_ratio,max_ratio,0.5,cmap, "Ratio carbon intensity, BEV:ICEV",midpoint=1,num_clr_divs=15)#"RdYlGn_r","Ratio carbon intensity, BEV:ICEV",1)
#    elif "Ratio" in name:
#        make_figures(mapping_data,name,"Ratio, BEV:ICEV, domestic production",min_ratio,max_ratio,0.2,"RdBu_r","Ratio carbon intensity, BEV:ICEV, domestic production of segment "+segment_list[i]+" batteries",midpoint=1, num_clr_divs=11)
#        #keeper = " run {:%d-%m-%y, %H_%M}".format(datetime.now())
#        #if export_data==True:
#         #   plt.savefig(name+keeper, bbox_inches='tight')
#    elif "EUR production impacts" in name:
#        make_figures(mapping_data,name,"Battery production impacts \n (t CO$_2$-eq/battery)",min_prod,max_prod,2,"viridis","domestic production of segment "+segment_list[i]+" batteries",num_clr_divs=11)
#    elif "BEV footprint" in name:
#        if "EUR production" in name:
#            make_figures(mapping_data,name,"BEV footprint \n (g CO$_2$-eq/km)", min_BEV,max_BEV,50,cmap_BEV,"BEV lifecycle carbon intensity, domestic production of segment "+segment_list[i]+" batteries",num_clr_divs=11) #tab20b
#        else:
#            make_figures(mapping_data,name,"BEV footprint \n (g CO$_2$-eq/km)", min_BEV,max_BEV,50,cmap_BEV,"BEV lifecycle carbon intensity, Segment "+segment_list[i],num_clr_divs=11) #tab20b
#    elif "Use phase" in name:
#        make_figures(mapping_data,name,"Ratio use phase only, BEV:ICEV",min_ratio,max_ratio,0.5,"RdBu_r","Use phase ratio carbon intensity, Segment "+segment_list[i],midpoint=1, num_clr_divs=11)
#
#    #plt.xlim((-12,32))
#    #plt.ylim((32,72))


# %%

#
#for i, name in enumerate(col_names):
#    if "BEV footprint" in name:
#        if "EUR production" in name:
#            make_figures(mapping_data,name,"BEV footprint \n (g CO2-eq/km)", min_BEV,max_BEV,50,"tab20b","BEV lifecycle carbon intensity, domestic production of segment "+segment_list[i]+" batteries",num_clr_divs=15) #tab20b
