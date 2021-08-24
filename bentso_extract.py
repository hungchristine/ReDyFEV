## coding: utf-8
"""
 Retrieve data from ENTSO-E Transparency Platform
 Export
"""

import os
import numpy as np
import pickle

import logging

from bentso import *
from bentso import constants
from bentso import iterators
from bentso import CachingDataClient


entsoe_fp = os.path.abspath(os.path.join(os.path.curdir, 'output','entsoe'))
# entsoe_fp = r'C:\Users\chrishun\Box Sync\000 Projects IndEcol\90088200 EVD4EUR\X00 EurEVFootprints\Submission files\code\data'
cl = CachingDataClient(location=entsoe_fp, verbose=True)

os.environ['ENTSOE_API_TOKEN'] =  #<insert ENTSO-E API token here>
fp = os.path.abspath(os.path.curdir)
fp_output = os.path.join(os.path.curdir, 'output', 'entsoe')

# #### Using Bentso

country_list = ['AL', 'AT', 'BA', 'BE', 'BG',
                  'CH', 'CZ', 'DE', 'DK', 'EE',
                  'ES', 'FI', 'FR', 'GR', 'HR',
                  'HU', 'IE', 'IT', 'LT', 'LU',
                  'LV', 'ME', 'MK', 'NL', 'NO',
                  'PL', 'PT', 'RO', 'RS', 'SE',
                  'SI', 'SK', 'GB', 'TR', 'GE',
                  'CY', 'HR', 'NI', 'GB-NIR',
                  'BY', 'UA', 'MD', 'RU', 'MT', 'MD']

# smaller group for testing
# country_list = ['AT']#,'FR', 'NO', 'AT']  # ,'DK','IT']

"""Fetch production mixes from Transparency Platform
   saves as dict of Dataframes.

   NOTE 1: that GB only fetches
   one of two bidding areas; both GB and GB-NIR required
   for full GB geographical resolution

    NOTE 2: IE fetches the country-wide data from the Platform,
    however, this is incomplete relative to the bidding area
    region for IE; however, bentso does not provide a means
    to fetch the bidding area, so this is read in manually from
    a csv exported directly from the Platform.
 """

def fetch_generation(year=2019, full_year=True):
    gen_dict = {}
    for country in country_list:
        try:
            gen_dict[country] = cl.get_generation(country, year, full_year=full_year)
        except Exception as e:
            print(f'Warning: no data for generation in {country}!')
            print(e)
            logging.warning(f'No generation data for {country}')
            logging.warning(e)
    with open(r'entso_export_gen_' + str(year) + '.pkl', 'wb') as handle:
        pickle.dump(gen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return gen_dict


def fetch_trade(year=2019, full_year=True):
    """ Fetch trade relationships from Transparency Platform
     Searches all possible country pairs from country_list;
     non-existent trade relationships save as 'error'
    """
    trade_dict = {}
    no_trade_rel = []
    for exp_country in country_list:
        for imp_country in country_list:
            try:
                trade_dict[(exp_country, imp_country)] = cl.get_trade(
                        exp_country, imp_country, year, full_year=full_year)
            except:
                print(f'Warning: {imp_country}-{exp_country} trade relationship does not exist!')
                logging.warning(f'Warning: {imp_country}-{exp_country} trade relationship does not exist!')
                trade_dict[(exp_country, imp_country)] = np.nan
                no_trade_rel.append((exp_country, imp_country))
    with open(r'entso_export_trade_' + str(year) + '.pkl', 'wb') as handle:
        pickle.dump(trade_dict, handle)
    return trade_dict


def bentso_query(year=2019, full_year=True):
    os.chdir(fp_output)
    logging.info('Querying generation data')
    gen_dict = fetch_generation(year, full_year)
    logging.info('Querying trade data')
    trade_dict = fetch_trade(year, full_year)

    os.chdir(fp)

    return gen_dict
