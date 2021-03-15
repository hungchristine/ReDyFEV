# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:59:28 2021

@author: chrishun
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import country_converter as coco
import re
import os
import pickle



#%% Import electricity data from Eurostat - to be used as 2020 baseline
def load_eurostat():
    """ Load electricity mixes, regionalized LCA/hybrid LCA factors from BEV footprints """
    data_fp = os.path.join(os.path.curdir, 'data')
    #r'C:\Users\chrishun\Box Sync\000 Projects IndEcol\90088200 EVD4EUR\X00 EurEVFootprints\Submission files\code\data'
    comb_fp = os.path.join(data_fp, 'NRG_IND_PEHCF__custom_6752251615553482189.xlsx')  # from Eurostat (2019 data)
    renew_fp = os.path.join(data_fp, 'NRG_IND_PEHNF__custom_6773811615580925050.xlsx')  # from Eurostat (2019 data)
    gross_net_fp = os.path.join(data_fp, 'NRG_IND_PEH__custom_6752301615553529103.xlsx')

    imp_fp = os.path.join(data_fp, 'NRG_TI_EH__custom_6791521615643363319.xlsx')
    exp_fp = os.path.join(data_fp, 'NRG_TE_EH__custom_6791721615643510206.xlsx')
    # tec_int_fp = os.path.join(data_fp, 'tec_intensities.csv')  # hybridized, regionalized LCA factors for electricity generation

    comb_df = pd.read_excel(comb_fp, sheet_name='Sheet 1', header=0, index_col=[0], skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], skipfooter=3, na_values=':')  # 2018 production mix, from Eurostat
    renew_df = pd.read_excel(renew_fp, sheet_name='Sheet 1', header=0, index_col=[0], skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], skipfooter=6, na_values=':')

    gross_net_df = pd.read_excel(gross_net_fp, sheet_name='Sheet 1', header=0, index_col=[0], skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], skipfooter=3, na_values=':')

    # trades_df = pd.read_csv(trades_fp, index_col=[0], na_values='-')  # 2019 production, in TWh

    #%% Assemble trade matrix
    # NB: Eurostat publishes both import and export tables; however, these are not internally consistent (i.e., do not balance),
    # and have different geographical resolution.
    # Solution: Take mean of values from both tables where these exist, otherwise take value where present in one of the two
    # tables.

    eurostat_import = pd.read_excel(imp_fp, sheet_name='Sheet 1', header=0, index_col=[0], skiprows=[0,1,2,3,4,5,6,7,8], skipfooter = 3, usecols="A:LR", na_values=":")
    eurostat_export = pd.read_excel(exp_fp, sheet_name='Sheet 1', header=0, index_col=[0], skiprows=[0,1,2,3,4,5,6,7,8], skipfooter = 5, usecols="A:LR", na_values=":")

    eurostat_import.drop(columns=eurostat_import.filter(like='Unnamed').columns, inplace=True)
    eurostat_export.drop(columns=eurostat_export.filter(like='Unnamed').columns, inplace=True)

    # replace extraneous countries
    eurostat_import.replace(0, np.nan, inplace=True)
    eurostat_export.replace(0, np.nan, inplace=True)

    eurostat_import = eurostat_import.T  # make matrix in producer-receiver format

    # import and export data do not seem to agree; take mean values
    eurostat_trade = pd.concat([eurostat_import.stack(), eurostat_export.stack()], axis=1).mean(axis=1).unstack()
    trade_df = eurostat_trade / 1000
    trade_df.drop(index=['Total', 'Not specified'], columns=['Total', 'Not specified'], inplace=True)   # TODO: redistribute these across other partners; not major (values are low)

    replace_country_names(trade_df, 'both')

    # Andorra and Lithuania are trade-only states; remove these as trade amounts are rather small, and no ecoinvent processes for these
    trade_df.drop(index=['AD'], columns=['AD'], inplace=True)

    #%% perform calculations on hydropower (for Eurostat data)
    hydro_df = renew_df.filter(like='ydro')
    hydro_df.replace(np.nan, 0, inplace=True)
    hydro_df['Hydro Water Reservoir'] = (hydro_df['Pure hydro power'] - hydro_df['Run-of-river hydro power']) + (hydro_df['Mixed hydro power'] - hydro_df['Mixed hydro power - pumping'])
    hydro_df['Hydro Pumped Storage'] = hydro_df['Mixed hydro power - pumping'] + hydro_df['Pumped hydro power']
    hydro_df['Hydro Run-of-river and poundage'] = hydro_df['Run-of-river hydro power']

    hydro_df.drop(columns=['Hydro', 'Pure hydro power', 'Run-of-river hydro power', 'Mixed hydro power', 'Mixed hydro power - pumping', 'Pumped hydro power'], inplace=True)
    renew_df.drop(columns=['Hydro', 'Pure hydro power', 'Run-of-river hydro power', 'Mixed hydro power', 'Mixed hydro power - pumping', 'Pumped hydro power'], inplace=True)

    # remove note columns from Eurostat data
    renew_df.drop(columns=renew_df.filter(like='Unnamed').columns, inplace=True)
    comb_df.drop(columns=comb_df.filter(like='Unnamed').columns, inplace=True)
    gross_net_df.drop(columns=gross_net_df.filter(like='Unnamed').columns, inplace=True)

    # calculate losses (difference between gross and net production)
    gross_net_df['losses'] = gross_net_df['Gross electricity production'] - gross_net_df['Net electricity production']

    # remove unnecessary aggregated categories from renewables table
    renew_df.drop(columns=['Wind'], inplace=True)
    renew_df.drop(columns=renew_df.filter(like='photovoltaic (').columns, inplace=True)

    gen_df = renew_df.join(hydro_df).join(comb_df)

    gen_df = gen_df / 1000  # Eurostat data in GWh; convert to TWh

    eurostat_dict = {'Fossil Hard coal': ['Anthracite',
                                          'Coking coal',
                                          'Other bituminous coal',
                                          'Sub-bituminous coal'],
                     'Fossil Brown coal/Lignite': ['Lignite',
                                                   'Coke oven coke',
                                                   'Gas coke',
                                                   'Patent fuel',
                                                   'Brown coal briquettes',
                                                   'Coal tar'],
                     'Fossil Coal-derived gas': ['Coke oven gas',
                                                 'Gas works gas',
                                                 'Blast furnace gas',
                                                 'Other recovered gases'],
                     'Fossil Oil': ['Crude oil',
                                    'Natural gas liquids',
                                    'Refinery gas',
                                    'Liquefied petroleum gases',
                                    'Naphtha',
                                    'Kerosene-type jet fuel',
                                    'Other kerosene',
                                    'Gas oil and diesel oil',
                                    'Fuel oil',
                                    'Other oil products',
                                    'Petroleum coke',
                                    'Bitumen'],
                     'Fossil Oil shale': 'Oil shale and oil sands',
                     'Fossil Peat': ['Peat',
                                     'Peat products'],
                     'Fossil Gas': 'Natural gas',
                     'Nuclear': 'Nuclear fuels and other fuels n.e.c.',
                     'Solar': ['Solar thermal', 'Solar photovoltaic'],
                     # 'Solar CSP': 'Solar thermal',
                     # 'Solar PV': 'Solar photovoltaic',
                     'Biomass': ['Primary solid biofuels',
                                 'Solid biofuels',
                                 'Charcoal',
                                 'Pure biogasoline',
                                 'Blended biogasoline',
                                 'Pure biodiesels',
                                 'Blended biodiesels',
                                 'Pure bio jet kerosene',
                                 'Blended bio jet kerosene',
                                 'Other liquid biofuels',
                                 'Biogases'],
                     'Geothermal': 'Geothermal',
                     'Marine': 'Tide, wave, ocean',
                     'Waste': ['Industrial waste (non-renewable)',
                               'Renewable municipal waste',
                               'Non-renewable municipal waste'],
                     'Wind Offshore': 'Wind off shore',
                     'Wind Onshore': 'Wind on shore',
                     'Other': ['Other fuels n.e.c. - heat from chemical sources',
                               'Other fuels n.e.c.']
                     }


    #%% Replace Eurostat nomenclature with ENTSO for integration with existing code


    for new_tec, eurostat_tec in eurostat_dict.items():
        if isinstance(eurostat_tec, list):
            # sum all sub-types and remove from df
            gen_df[new_tec] = gen_df.loc(axis=1)[gen_df.columns.intersection(eurostat_tec)].sum(axis=1)
            gen_df.drop(columns=gen_df.columns.intersection(eurostat_tec), inplace=True)
        else:
            # rename technologies with one-to-one correspondance
            gen_df.rename(columns={eurostat_tec: new_tec}, inplace=True)

    gen_df.dropna(how='all', axis=1, inplace=True)

    replace_country_names(gen_df, 0)
    replace_country_names(gross_net_df, 0)
    replace_country_names(comb_df, 0)
    replace_country_names(renew_df, 0)

    check = gen_df.sum(axis=1)*1000 - gross_net_df['Gross electricity production']
    check = check.loc[check.abs() > 1]

    # Add in countries with trade relationships, but no generation data; assume 0 production
    add_countries = list(set(trade_df.index) - set(gen_df.loc[gen_df.sum(axis=1) > 0].index))#list(set(trade_df.index) - set(gen_df.index))
    for country in add_countries:
        gen_df.loc[country] = 0

    rem_countries = list(set(gen_df.index) - set(trade_df.index))  # countries with generation but no trade data
    trade_df = trade_df.reindex(trade_df.index.union(rem_countries))
    trade_df = trade_df.reindex(columns=trade_df.columns.union(rem_countries))
    trade_df.sort_index(axis=1, inplace=True)
    trade_df.sort_index(axis=0, inplace=True)
    trade_df.replace(np.nan, 0, inplace=True)

    fp = os.path.abspath(os.path.curdir)
    os.chdir(os.path.join(fp, 'output'))
    with open(r'trade_final_eurostat.pkl', 'wb') as handle:
        pickle.dump(trade_df, handle)
    with open(r'gen_final_eurostat.pkl', 'wb') as handle:
        pickle.dump(gen_df, handle)

    os.chdir(fp)

    return add_countries


# Perform label cleaning before converting to ISO A2 country codes
def replace_country_names(df, axis):
    repl_dict = {'Germany (until 1990 former territory of the FRG)': 'Germany',
             'Kosovo (under United Nations Security Council Resolution 1244/99)': 'Kosovo'}
    try:
        df.drop(index=['European Union - 28 countries (2013-2020)',
                       'European Union - 27 countries (from 2020)',
                       'Euro area - 19 countries  (from 2015)' ], inplace=True)
    except KeyError:
        print("European union not found in country labels")
    if (axis==0) or (axis=='both') or (axis=='index'):
        df.rename(index=repl_dict, inplace=True)
        df.index = pd.Index(coco.convert(df.index.tolist(), to='ISO2'))
    if (axis==1) or (axis=='both') or (axis=='columns'):
        df.rename(columns=repl_dict, inplace=True)
        df.columns = pd.Index(coco.convert(df.columns.tolist(), to='ISO2'))
#%%

# tec_int_df = pd.read_csv(tec_int_fp, index_col=[0], na_values='-')  # regionalized (hybridized) carbon intensity factors of generation (g COw-e/kWh)
# tec_int_df.rename(columns={'other': 'Other', 'other renewable': 'Other renewable'}, inplace=True)

# # Make list of ISO-A2 country codes of relevant countries
# iso_a2 = europe_shapes[europe_shapes['Consumption mix intensity'].notna()].ISO_A2
# iso_a2.rename('country', inplace=True)
