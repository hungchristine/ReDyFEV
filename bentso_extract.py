## coding: utf-8
"""
 Retrieve data from ENTSO-E Transparency Platform
 Export
"""

import appdirs
import hashlib
import os
import pandas as pd
import numpy as np
import pickle

from bentso import *
from bentso import constants
from bentso import iterators
from bentso import CachingDataClient


cl = CachingDataClient(location=r'C:\Users\chrishun\Box Sync\000 Projects IndEcol\90088200 EVD4E UR\X00 EurEVFootprints\Data\bentso')
#os.environ['ENTSOE_API_TOKEN'] =  #<insert ENTSO-E API token here>
fp = os.path.abspath(os.path.curdir)
fp_output = os.path.join(os.path.curdir, 'output')
os.chdir(fp_output)

# #### Using Bentso

country_list = ['AL', 'AT', 'BA', 'BE', 'BG',
                'CH', 'CZ', 'DE', 'DK', 'EE',
                'ES', 'FI', 'FR', 'GR', 'HR',
                'HU', 'IE', 'IT', 'LT', 'LU',
                'LV', 'ME', 'MK', 'NL', 'NO',
                'PL', 'PT', 'RO', 'RS', 'SE',
                'SI', 'SK', 'GB', 'TR', 'GE',
                'CY', 'HR', 'NI', 'GB-NIR',
                'BY','UA','MD','RU','MT', 'MD']

# smaller group for testing
#country_list = ['DE','FR']#,'DK','IT']

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
def fetch_generation():
    gen_dict={}
    for country in country_list:
        try:
            gen_dict[country] = cl.get_generation(country, 2019, full_year=True)
        except:
            print(f'Warning: no data for generation in {country}!')
    with open(r'entso_export_gen.pkl', 'wb') as handle:
        pickle.dump(gen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return gen_dict


""" Fetch trade relationships from Transparency Platform
 Searches all possible country pairs from country_list;
 non-existent trade relationships save as 'error'
"""

def fetch_trade():
    trade_dict={}
    no_trade_rel = []
    for exp_country in country_list:
        for imp_country in country_list:
            try:
                trade_dict[(exp_country, imp_country)] = cl.get_trade(
                        exp_country, imp_country,2019, full_year=True)
            except:
                print(f'Warning: {imp_country}-{exp_country} trade relationship does not exist!')
                trade_dict[(exp_country, imp_country)] = np.nan
                no_trade_rel.append((exp_country, imp_country))
    with open(r'entso_export_trade.pkl', 'wb') as handle:
        pickle.dump(trade_dict, handle)
    return trade_dict

gen_dict = fetch_generation()
trade_dict = fetch_trade()

os.chdir(fp)