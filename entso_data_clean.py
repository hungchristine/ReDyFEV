# -*- coding: utf-8 -*-
"""
 Reads in the raw data fetched from ENTSO-E Transparency Platform, refactors the
 data and fills in missing data
"""
import pandas as pd
import numpy as np
import pickle

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