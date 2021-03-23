# -*- coding: utf-8 -*-
"""
 Reads in the raw data fetched from ENTSO-E Transparency Platform, refactors the
 data and fills in missing data
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timezone, timedelta
import pickle
import os
import logging

#%%
def checkEqual(iterator):
    return len(set(iterator)) <= 1

def distribute_neg_hydro(time_per):
    try:
        if time_per['Hydro Pumped Storage'] < 0:
            neg_val = time_per['Hydro Pumped Storage']
            time_per['Hydro Pumped Storage'] = 0
            # "supply" pumping power from proportional to the production mix of the
            #  time period
            time_per = time_per + (time_per / time_per.sum()) * neg_val
        return time_per
    except KeyError:
        print('No pumped hydro')
    except Exception as e:
        print('Error in correcting for pumped hydro')
        print(e)

def aggregate_entso(dictionary, start=None, end=None, start2=None):
    """ Receives dictionary of raw data queried from ENTSO-E Transparency Portal
        and start and end of sub-annual period desired (optional)
        Calculates the size in timesteps between samples (rows) and checks
        if they are identical (i.e., even timesteps throughout time series)
    """
    new_dict = {}
    summed_dict = {}
    no_trade = []

    for key, value in dictionary.items():
        timestep_list = []
        if (start is not None) and (end is not None) and (not isinstance(value, float)):
            try:
                value_tmp = value.loc[start:end]
                if value_tmp.empty and (start2 is not None):
                    print(f'{key} does not have an entry for {start}. Sampling closest hour instead')
                    value_tmp = value.loc[start2:start2 + timedelta(minutes=1)]
                # entsotime = value.index
            except KeyError:
                if isinstance(key, tuple):
                    print(f'No trade for {key} at {start}')
            except AttributeError:
                if isinstance(key, tuple):
                    no_trade.append(key)
                else:
                    print(f'No dataframe for {key}, cannot take slice!')
                raise
            except Exception as e:
                print('Ran into a problem!')
                print(e)
                raise

            value = value_tmp
#        try:
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            # we don't need timesteps if we have a single time period
            for i in range(0, value.shape[0]-1):
                try:
                    timestep = (value.index[i+1] - value.index[i]).total_seconds() / 3600  # get timestep in hours
                    timestep_list.append(timestep)
                except IndexError:
                    print(f'IndexError {i}')
                except:
                    print('Could not perform!')

            # Make sure timesteps are all equal length before calculating
            # NB: this is made obsolete by using bentso's fullyear=True
            if checkEqual(timestep_list):
                if (isinstance(value, pd.DataFrame)):
                    # ENTSO-E data from 2020 introduces MultiIndex headers
                    if type(value.columns) == pd.MultiIndex:
                        value = value.loc(axis=1)[:, 'Actual Aggregated']  # drop "Actual Consumption"
                        value.columns = value.columns.droplevel(1)
                    if (value.columns.str.contains('Hydro Pumped Storage').sum() == 1):
                        # check if value is the production matrix and if so, if the country has pumped storage
                        logging.info(f'Correcting for negative pumped hydropower in {key}')
                        value = value.apply(distribute_neg_hydro, axis=1)  # correct for negative pumped hydro
                if value.shape[0] -1 > 0:
                    summed_dict[key] = (value * timestep_list[0] / 1e6).sum() # To calculate electricity generated; take power per timestep and multiply by length of time period. Sum over whole year
                elif value.shape[0] - 1 == 0:
                    summed_dict[key] = (value / 1e6).sum()
                new_dict[key] = value
            else:
                print(f'warning: unequal time steps for {key}')
                logging.warning(f'unequal time steps for %s', key)
                if min(timestep_list) == 1:
                    logging.info(f'resampling data to 1 hour increments in {key}')
                    new_dict[key] = value.resample('1H').interpolate()
                    summed_dict[key] = (new_dict[key] * timestep_list[0] / 1e6).sum()
                    print('1 hour')
                elif min(timestep_list) == 0.25:
                    logging.info(f'resampling data to 15 minute increments in {key}')
                    new_dict[key] = value.resample('15T').interpolate()
                    summed_dict[key] = (new_dict[key] * timestep_list[0] / 1e6).sum()
                    print('15 minutes')
                else:
                    print('very uneven timesteps')
                    #value = (value*timestep_list[0]/1000).sum()
#        except Exception as e:
#           print(f'something went wrong in {key}')
#           print(e)
    return summed_dict, new_dict #, entsotime


def build_trade_mat(trade_dict):
    mi = trade_dict.keys()
    trade_mat = pd.DataFrame(trade_dict.values(), index=mi)
    trade_mat = trade_mat.unstack()
    trade_mat.columns = trade_mat.columns.get_level_values(1)  # remove extraneous level from column labels
    return trade_mat

def clean_entso(year=None, start=None, end=None, country=None):
    fp = os.path.abspath(os.path.curdir)
    fp_output = os.path.join(os.path.curdir, 'output')
    os.chdir(fp_output)

    ## Load previous results
    with open(r'entso_export_trade_' + str(year) + '.pkl', 'rb') as handle:
        trade_dict = pickle.load(handle)
    with open(r'entso_export_gen_' + str(year) + '.pkl', 'rb') as handle:
        gen_dict = pickle.load(handle)

    generation = gen_dict.copy()
    trade = trade_dict.copy()

#    start = datetime(2019, 1, 1, 0)  # test values for sub-annual periods
#    end = datetime(2019, 6, 30, 23)
    if country is not None:
        # find closest timestamp to query date
        start = generation[country].iloc[generation[country].index.get_loc(start, method='nearest')].name
        if start.minute != 00:
            # some countries only have hourly sampling; go to nearest hour
            if start.minute > 30:
                start2 = start + timedelta(minutes=60-start.minute)
            else:
                start2 = start - timedelta(minutes=start.minute)
        else:
            start2 = None
        end = start + timedelta(minutes=1)
        # summed_gen = generation[country].loc[entsotime]
        # summed_trade = trade[country].loc[entsotime]
    else:
        start2 = None
    summed_gen, new_gen = aggregate_entso(generation, start, end, start2)
    summed_trade, new_trade  = aggregate_entso(trade, start, end, start2)

    trade_df = build_trade_mat(summed_trade)
    gen_df = pd.DataFrame.from_dict(summed_gen, orient='index')

    """ Read in Ireland's data manually from .csv export from Transparency Platform;
        'country' (which Bentso fetches) and the 'bidding zone' classifications differ,
        for which the latter has higher production values (which agree better with statistical
        production from 2018)
    """
    # if year == 2019:
    #     fp_ie = os.path.join(fp, 'data', 'gen_IE.csv')
    #     new_ie = pd.read_csv(fp_ie)
    #     new_ie = new_ie.set_index('MTU', drop=True, append=False).drop('Area', axis=1)
    #     new_ie = new_ie.replace('n/e', np.nan)
    #     new_ie = (new_ie * 0.5).sum() / 1e6  # samples on the half hour; MWh to TWh
    #     new_ie = new_ie.drop(index='Marine  - Actual Aggregated [MW]')
    #     new_ie.index = gen_df.columns
    #     gen_df.loc['IE'] = new_ie

    #     """ Aggregate GB and GB_NIR regions """
    #     gen_df = (gen_df.reset_index().replace({'index': {'GB-NIR':'GB'}}).groupby('index', sort=False).sum())
    #     trade_df = (trade_df.reset_index().replace({'index': {'GB-NIR':'GB'}}).groupby('index', sort=False).sum())  # aggregate trade rows
    #     trade_df = ((trade_df.T).reset_index().replace({'index': {'GB-NIR':'GB'}}).groupby('index', sort=False).sum()).T  # aggregate trade column
    # elif year == 2021:
    #     new_ie = generation['IE']
    #     new_ie = new_ie.loc(axis=1)[:, 'Actual Aggregated']  # drop "Actual Consumption"
    #     new_ie.columns = new_ie.columns.droplevel(1)
    #     new_ie *= 2  # for whatever reason, results from IE are half of what they should be....
    #     new_ie = (new_ie * 0.5).sum() / 1e6  # from MWh to TWh
    #     new_ie.rename('IE', inplace=True)
    #     gen_df.drop(index='IE', inplace=True)
    #     gen_df = gen_df.append(new_ie)

    """ Add Cyprus to trade df (no trade relationships) """
    trade_df['CY'] = 0
    trade_df.loc['CY'] = 0

    # Add in countries with trade relationships, but no generation data; assume 0 production
    add_countries = list(set(trade_df.index) - set(gen_df.index))
    for count in add_countries:
        gen_df.loc[count] = 0

    with open(r'trade_final_' + str(year) + '.pkl', 'wb') as handle:
        pickle.dump(trade_df, handle)
    with open(r'gen_final_' + str(year) + '.pkl', 'wb') as handle:
        pickle.dump(gen_df, handle)

    logging.info('Completed export of ENTSO-E data')
    os.chdir(fp)

    if country is not None:
        entsotime = start  # report the ENTSO sampling period used
        return add_countries, entsotime
    else:
        return add_countries