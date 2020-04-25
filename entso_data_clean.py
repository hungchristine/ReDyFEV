# -*- coding: utf-8 -*-
"""
 Reads in the raw data fetched from ENTSO-E Transparency Platform, refactors the
 data and fills in missing data
"""
import pandas as pd
import numpy as np
import pickle

import os, sys

## Load previous results
with open(r'code output/entso_export_gen.pkl', 'rb') as handle:
    trade_dict = pickle.load(handle)
with open(r'code output/entso_export_trade.pkl', 'rb') as handle:
    gen_dict = pickle.load(handle)
    
#%%
def checkEqual(iterator):
    return len(set(iterator)) <= 1

def aggregate_entso(dictionary):    
    new_dict = {}
    summed_dict = {}
    for key, value in dictionary.items():
        timestep_list = []
        try:
            if isinstance(value, pd.DataFrame) or isinstance(value,pd.Series):
                for i in range(0,value.shape[0]-1):
                    try:
                        timestep = (value.index[i+1] - value.index[i]).total_seconds()/3600 # get timestep in hours
                        timestep_list.append(timestep)
                    except IndexError:
                        print(f'IndexError {i}')
                    except:
                        print('Could not perform!')
                        
                # Make sure timesteps are all equal length before calculating
                # NB: this is made obsolete by using bentso's fullyear=True
                if checkEqual(timestep_list):
                    summed_dict[key] = (value*timestep_list[0]/1e6).sum() # To calculate electricity generated; take power per timestep and multiply by length of time period. Sum over whole year
                    new_dict[key] = value
                else:
                    print(f'warning: unequal time steps for {key}')
                    if min(timestep_list)==1:
                        new_dict[key] = value.resample('1H').interpolate()
                        summed_dict[key] = (new_dict[key]*timestep_list[0]/1e6).sum()
                        print('1 hour')
                    elif min(timestep_list)==0.25:
                        new_dict[key] = value.resample('15T').interpolate()
                        summed_dict[key] = (new_dict[key]*timestep_list[0]/1e6).sum()
                        print('15 minutes')
                    else:
                        print('very uneven timesteps')
                        #value = (value*timestep_list[0]/1000).sum()
        except:
           print(f'something went wrong in {key}')
    return summed_dict, new_dict

def build_trade_mat(trade_dict):
    mi = trade_dict.keys()
    trade_mat = pd.DataFrame(trade_dict.values(), index=mi)
    trade_mat = trade_mat.unstack()
    trade_mat.columns = trade_mat.columns.get_level_values(1) # remove extraneous level from column labels
    return trade_mat


#%%
generation = gen_dict.copy()
trade = trade_dict.copy()

summed_gen, new_gen = aggregate_entso(generation)
summed_trade, new_trade = aggregate_entso(trade)

trade_df = build_trade_mat(summed_trade)
gen_df = pd.DataFrame.from_dict(summed_gen, orient='index')

""" Read in Ireland's data manually from .csv export from Transparency Platform; 
    'country' (which Bentso fetches) and the 'bidding zone' classifications differ,
    for which the latter has higher production values (which agree better with statistical 
    production from 2018)
"""

new_ie = pd.read_csv(r'Data/gen_IE.csv')
new_ie = new_ie.set_index('MTU',drop=True,append=False).drop('Area',axis=1)
new_ie = new_ie.replace('n/e',np.nan)
new_ie = (new_ie*0.5).sum()/1e6
new_ie = new_ie.drop(index='Marine  - Actual Aggregated [MW]')
new_ie.index = gen_df.columns

gen_df.loc['IE'] = new_ie

""" Aggregate GB and GB_NIR regions """
gen_df = (gen_df.reset_index().replace({'index':{'GB-NIR':'GB'}}).groupby('index',sort=False).sum())
trade_df = (trade_df.reset_index().replace({'index':{'GB-NIR':'GB'}}).groupby('index',sort=False).sum()) # aggregate trade rows
trade_df = ((trade_df.T).reset_index().replace({'index':{'GB-NIR':'GB'}}).groupby('index',sort=False).sum()).T # aggregate trade column

""" Add Cyprus to trade df (no trade relationships) """
trade_df['CY'] = 0 
trade_df.loc['CY'] = 0

# Add in countries with trade relationships, but no generation data; assume 0 production
add_countries = list(set(trade_df.index)-set(gen_df.index))
for country in add_countries:
    gen_df.loc[country] = 0

with open(r'code output/trade_final.pkl', 'wb') as handle:
    pickle.dump(trade_df, handle)
with open(r'code output/gen_final.pkl', 'wb') as handle:
    pickle.dump(gen_df, handle)

#%%
""" Load hybridized emission factors """
#
#fp_dir = os.path.join(os.path.curdir,os.path.pardir,'Results')
#sys.path.append(fp_dir)
#fp = os.path.join(fp_dir, 'hybrid_emission_factors_final.xlsx')
#
#ef = pd.read_excel(fp, sheet_name = 'country_emission_factors', index_col=[0], header=[0])
#trade_efs = pd.read_excel(fp, sheet_name = 'trade ef_hv', index_col=[0,1,2,3], header=[0])
#
#""" Fix missing emission factors """
## Note that these disregard whether or not there is actual production
#ef['Fossil Oil shale'] = ef['Fossil Oil'] # Proxy shale oil with conventional oil
#ef['Waste'] = 0 # Allocate incineration emissions to waste treatment as per ecoinvent 
#ef['Marine'] = 0
#
## Put in proxy for missing regions and technologies
#missing_factors = pd.read_excel(fp,sheet_name = 'missing_emission_factors', index_col=[0],header=[0])
#missing_factors.dropna(how='all',axis=1, inplace=True)
#missing_factors_dict = {}
#for tec, col in missing_factors.iteritems():
#    missing_factors_dict[tec] = list(col.dropna(how='any',axis=0).values)
#
## countries with no geo-specific factors
#missing_countries = list(set(gen_df.index)-set(ef.index))
#for country in missing_countries:
#    ef = ef.append(pd.Series(name=country))
#    
#temp_gen_df = gen_df.loc[missing_countries]
#temp_gen_df = temp_gen_df.replace(0, np.nan).dropna(how='all', axis=1)
#temp_dict = {}
#for tec, col in temp_gen_df.items():
#    temp = temp_gen_df[tec] > 0
#    temp_ind = temp.index
#    temp_dict[tec] = temp_ind[temp.values].to_list()
#
#""" Merge dictionaries and keep values of common keys in list"""
#def mergeDict(dict1, dict2):
#    dict3 = {**dict1, **dict2}
#    for key, value in dict3.items():
#            if key in dict1 and key in dict2:
#                dict3[key] = (dict1[key]+ dict2[key])
#    return dict3
#
## Merge dictionaries and add values of common keys in a list
#missing_factors_dict2 = mergeDict(missing_factors_dict, temp_dict)
#missing_factors_dict2
#
## Use continental arithmetic average of same technology
#for key, countries in missing_factors_dict2.items():
#    if not key=='Other' and not key=='Other renewable': # these are dealt with below
#        ef[key].loc[countries] = ef[key].mean()
#
## Make 'other' and 'other renewable' approximations
#renewable_dict = {'renewable': ['Biomass', 'Geothermal', 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',	
#                                'Hydro Water Reservoir','Solar','Waste','Wind Onshore',	'Wind Offshore','Marine'],
#                'non-renewable': ['Fossil Gas','Fossil Hard coal','Fossil Oil','Fossil Brown coal/Lignite','Nuclear','Fossil Coal-derived gas','Fossil Oil shale','Fossil Peat']}
#
#other_tec = gen_df.loc[:,renewable_dict['non-renewable']]
#other_wf = other_tec.div(other_tec.sum(axis=1),axis=0) # determine generation shares for each country to determine 
#
#other_renew = gen_df.loc[:,renewable_dict['renewable']]
#other_renew_wf = other_renew.div(other_renew.sum(axis=1),axis=0)
#
#ef['other'] = (ef*other_wf).sum(axis=1)
#ef['other renewable'] = (ef*other_renew_wf).sum(axis=1)
#
#sort_indices(trade_df)
#sort_indices(gen_df)
#sort_indices(ef)
#
## Export all data
## .csv for use in Electricity mix calculations.py
#trade_df.to_csv('trades.csv')
#gen_df.to_csv('ENTSO_production_volumes.csv')
#ef.to_csv('final_emission_factors.csv')
#
#with pd.ExcelWriter('entsoe_export_final.xlsx') as writer:
#    trade_df.to_excel(writer, 'trade')
#    gen_df.to_excel(writer, 'generation')
#    ef.to_excel(writer,'new ef')