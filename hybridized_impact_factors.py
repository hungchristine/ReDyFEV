# The original code requires pylcaio, from Agez, found at:
# https://github.com/MaximeAgez/pylcaio
import pandas as pd
import numpy as np
import os, sys
import pickle
import logging

fp_results = os.path.join(os.path.curdir, 'results')
fp_output = os.path.join(os.path.curdir, 'output')

def hybrid_emission_factors(trade_only, use_entso, year=''):
    # append pylcaio folder with hybridized LCI processes
    sys.path.append(r'C:\Users\chrishun\Box Sync\000 Projects IndEcol\90088200 EVD4EUR\X00 EurEVFootprints\Data\hybridized LCA factors\pylcaio-master\pylcaio-master\src')
    import pylcaio
    Analysis = pylcaio.Analysis('ecoinvent3.5','exiobase3', method_double_counting='STAM', capitals=False)

    # For making human-readable indices; consists of all ecoinvent process metadata
    labels = Analysis.PRO_f

    # Make DataFrame for whole ecoinvent requirements matrix with human-readable indices
    Aff_labels = (labels.join(Analysis.A_ff,how='inner'))
    Aff_labels = (Aff_labels).set_index(keys=['activityName', 'geography', 'productName'], append=True)
    Aff_labels = Aff_labels.iloc[:,19:] # remove extraeneous metadata
    Aff_labels.columns = Aff_labels.index

    # Calculate life cycle impacts
    I = pd.DataFrame(np.eye(len(Analysis.A_ff)), Analysis.A_ff.index, Analysis.A_ff.columns)
    X = pd.DataFrame(np.linalg.solve(I-Analysis.A_ff, I), Analysis.A_ff.index, I.columns) # pro x pro
    D = Analysis.C_f.fillna(0).dot(Analysis.F_f).dot(X) # pro x imp

    D = D.T
    D_labels = labels.join(D, how='inner')
    D_labels.set_index(keys=['activityName', 'geography', 'productName'], append=True, inplace=True)
    D_labels = D_labels.iloc[:, 19:]
    D_labels = D_labels.T

    # Other possible impact methods
    imp_list = ['EDIP; environmental impact; global warming, GWP 100a; kg CO2-Eq', 'ReCiPe Midpoint (H); climate change; GWP100; kg CO2-Eq', 'ReCiPe Midpoint (H) V1.13; climate change; GWP100; kg CO2-Eq', 'IPCC 2013; climate change; GTP 100a; kg CO2-Eq']
    D_labels_comp = D_labels.loc[imp_list]
    D_labels = D_labels.loc['EDIP; environmental impact; global warming, GWP 100a; kg CO2-Eq']

    # the footprint of all hybridized processes from ecoinvent, with their original footprint
    fp_hybrid = os.path.join(os.path.curdir, 'data', 'Results_full_database_STAM_2011.xlsx')
    hybrid_results = pd.read_excel(fp_hybrid, 'GWP_only_hyb', usecols='A:E,G', index_col=[0,1,2,3,4])

    # ## List of ENTSO-E countries
    country_list = ['AL','AT','BA','BE','BG','CH','CZ','DE','DK','EE',
                    'ES','FI','FR','GR','HR','HU','IE','IT','LT','LU',
                    'LV','ME','MK','NL','NO','PL','PT','RO','RS','SE',
                    'SI','SK','GB','TR','GE','CY','HR','NI','UK']

    # Find all electricity processes
    set(labels.loc[labels['activityName'].str.contains('electricity')]['activityName'])

    # Isolate high voltage (production) mixes and low voltage mixes (which includes solar PV)
    # note that pumped storage hydropower in CH gets excluded as it has 0 contribution to the production mix (for whatever reason)
    market_mixes = labels.loc[labels['activityName'].str.contains('electricity, high voltage, production mix')] # production mixes, do not include solar or waste (or Swiss pumped hydro)
    waste_mixes = labels.loc[labels['activityName'].str.contains('electricity, from municipal waste incineration to generic market for electricity, medium voltage')]  # production mixes, do not include solar
    solar_mixes = labels.loc[labels['activityName'].str.contains('electricity, low voltage')]  # use low voltage mixes to get shares of solar photovoltaic

    # ### Filter for high-voltage production mixes
    # First, retrieve electricity mixes, and reduce un-needed processes
    df = Aff_labels[market_mixes.index]
    df_waste = Aff_labels.loc[waste_mixes.index]
    df_solar = Aff_labels[solar_mixes.index]

    df = df.loc[:, df.columns.isin(country_list, level=2)]

    # manual workaround for keeping CH pumped hydro after removing unneeded rows
    # (since this ecoinvent process apparently does not contribute to any electricity mixes in ecoinvent)
    ind =('69ccf019-081e-42e3-b7f5-879dc7dde86a_66c93e71-f32b-4591-901c-55395db5c132', 'electricity production, hydro, pumped storage', 'CH', 'electricity, high voltage')  # Activityid for Swiss pumped hydro
    col = df.columns.get_loc(('54ed269b-e329-469c-902e-a8991ee7d24a_66c93e71-f32b-4591-901c-55395db5c132', 'electricity, high voltage, production mix', 'CH', 'electricity, high voltage'))
    index = df.index.get_loc(ind)
    df.iloc[index, col] = 1e-7

    df = df.replace(0, np.nan).dropna(how='all', axis=0)

    # ### Filter for medium-voltage mixes (to find waste incineration shares)
    df_waste = df_waste.loc[:, df_waste.columns.isin(country_list, level=2)]
    df_waste = df_waste.replace(to_replace=0, value=np.nan).dropna(how='all', axis=0)

    # check unique waste incineration activities
    set(df_waste.index.get_level_values('activityName'))

    # ### Filter for low-voltage mixes (to find solar photovoltaic shares)
    df_solar = df_solar.loc[:, df_solar.columns.isin(country_list, level=2)]
    df_solar = df_solar.replace(to_replace=0, value=np.nan).dropna(how='all', axis=0)

    # ### Develop correspondence between ecoinvent and ENTSO-E/Eurostat technology categories

    # Correspondence from ENTSO-E technology categories to keywords for searching in ecoinvent processes
    # Note that the 'fossil oil shale', 'other' and 'other renewable' do not have ecoinvent equivalents;
    # these are dealt with later
    tec_dict = {'Biomass': ['biogas', 'wood chips'],
                'Fossil Brown coal/Lignite': 'lignite',
                'Fossil Coal-derived gas': 'coal gas',
                'Fossil Gas': 'natural gas',
                'Fossil Hard coal': 'hard coal',
                'Fossil Oil': ' oil',
                'Fossil Oil shale': '--',  #' oil?'
                'Fossil Peat': 'peat',
                'Geothermal': 'geothermal',
                'Hydro Pumped Storage': 'pumped storage',
                'Hydro Run-of-river and poundage': 'run-of-river',
                'Hydro Water Reservoir': 'reservoir',
                'Nuclear': 'nuclear',
                'Other': '---',
                'Other renewable': '----',
                'Solar': ['solar thermal', 'solar tower', 'photovoltaic'],
                'Waste': 'waste incineration',
                'Wind Offshore': 'offshore',
                'Wind Onshore': 'onshore'
    }

    # reverse correspondence from ecoinvent search keywords to ENTSO-E categories
    temp_tecdict = { 'biogas':'Biomass',
                    'wood chips':'Biomass',
                'lignite': 'Fossil Brown coal/Lignite' ,
                'coal gas': 'Fossil Coal-derived gas',
                'natural gas': 'Fossil Gas' ,
                'hard coal': 'Fossil Hard coal' ,
                ' oil': 'Fossil Oil',
                '--' : 'Fossil Oil shale',
                'peat': 'Fossil Peat' ,
                'geothermal':'Geothermal',
                'pumped storage': 'Hydro Pumped Storage',
                'run-of-river': 'Hydro Run-of-river and poundage',
                'reservoir': 'Hydro Water Reservoir',
                'nuclear' : 'Nuclear',
                '---':'Other' ,
                '----': 'Other renewable',
                'solar thermal': 'Solar',
                'solar tower': 'Solar',
                'photovoltaic': 'Solar',
                'waste incineration': 'Waste', #  'electricity, from municipal waste incineration to generic market for electricity, medium voltage',
                'offshore': 'Wind Offshore',
                'onshore':'Wind Onshore'
    }

    # flatten tec list
    tec_list = list(tec_dict.values())
    for item in tec_list:
        if type(item)==list:
            for subitem in item:
                tec_list.append(subitem)
            tec_list.remove(item)

    # Get list of solar PV processes in ecoinvent
    d = list(set(df_solar.index.get_level_values(level=1)))
    pv = [entry for entry in d if 'electricity production' in entry] # all low-voltage production technologies

    # Fetch waste incineration process names
    waste = ['electricity, from municipal waste incineration to generic market for electricity, medium voltage']

    # Make list of electricity generation technologies (both high voltage and low-voltage, i.e., solar PV)
    ecoinvent_tecs = list(set(df.index.get_level_values(level=1))) + pv + waste

    # Quality assurance; make sure all ecoinvent technologies are covered by the keyword search, and keyword matches are correct
    # also, check number of matches for each keyword
    matches = []
    match_keys = []
    num_matches = dict((keyword,0) for keyword in tec_list)

    for tec in ecoinvent_tecs:
        for keyword in tec_list:
            if keyword in tec and tec not in matches:
#                print(f'{keyword} is in {tec}')
                matches.append(tec)
                match_keys.append(temp_tecdict[keyword])
                num_matches[keyword] += 1

    display(num_matches)

    # Find ecoinvent tecs that were not matched
    if len(ecoinvent_tecs)-len(matches) > 0:
        print('ecoinvent process(es) not matched:')
        display(set(ecoinvent_tecs).difference(set(matches)))

    # Correspondence of ENTSO-E categories and specific ecoinvent processes
    from collections import defaultdict

    ei_tec_dict = defaultdict(list) # keys as entso-e categories, values are corresponding ei activities
    rev_ei_tec_dict = defaultdict(list) # vice-versa; keys as ei, values are entso-e categories
    for i, j in zip(match_keys, matches):
        ei_tec_dict[i].append(j)
        rev_ei_tec_dict[j].append(i)

    # Build correspondence table for Supplementary Information
    tec_index = pd.Index(list(ei_tec_dict.keys()))
    tec_values = list(ei_tec_dict.values())
    ei_tec_table = pd.DataFrame(tec_values, index=tec_index).stack()
    ei_tec_table.index = ei_tec_table.index.droplevel(level=1)
    ei_tec_table.sort_index(axis=0, inplace=True)


    # ### Calculate shares of ecoinvent processes
    # For multi-process technology categories, e.g., solar, biomass...

    # combine high-, medium- and low-voltage electricity mixes for full tech resolution
    f = pd.concat([df, df_waste, df_solar], axis=1)

    rev_ei_tec_dict = {k: v for k, v in rev_ei_tec_dict.items() if not not v}

    temp_list = f.index.get_level_values('activityName')
    entso_list = []
    f_check = []

    # For every activity in ecoinvent,if they are related to electricity production, add to entso_list
    # If not, drop from the matrix; they are irrelevant
    for entry in temp_list:
        try:
            entso_list.append(rev_ei_tec_dict[entry])
        except:
            f_check.append(entry)

    f.drop(index=set(f_check), level=1, axis=0, inplace=True)

    # inputs to electricity mixes not included in the calculations of emission factors
    set(f_check)

    entso_list = [item for sublist in entso_list for item in sublist] #flatten list

    entso_index = pd.Index(tuple(entso_list), name='entso_e')
    f.set_index(entso_index, append=True, inplace=True)


    # ### Get shares of each ecoinvent process for each ENTSO-E technology category

    f_gpby = f.groupby(level='entso_e')

    # Get shares of each ecoinvent process in each ENTSO-E technology category;
    # for calculating weighted average for emission factor
    ei_process_shares = f.divide(f_gpby.sum(), level=4, axis=1)

    hybrid_temp = hybrid_results.reset_index(level=[1,2,3,4], drop=True)
    g_temp = ei_process_shares.reset_index(level=[1,2,3,4], drop=True)


    # ### Calculate weighted average of hybridized emissions factor for each technology
    ef = ei_process_shares.mul(hybrid_temp.iloc[:, 0], axis=0, level=0)

    ef_aggregated = ef.groupby(level='entso_e').sum()

    ef_countries = ef_aggregated.sum(axis=1, level='geography')
    ef_countries.replace(0, np.nan, inplace=True)
    ef_countries.dropna(axis=1, how='all', inplace=True)
    ef_countries.sort_index(inplace=True)

    ef_countries = ef_countries.T
    ef_countries.sort_index(axis=0, inplace=True)

    if use_entso:
        # entso_fp = os.path.join(fp_output, 'ENTSO_production_volumes.csv')
        # entso_e_production = pd.read_csv(entso_fp, header=0, index_col=[0])
        entso_fp = os.path.join(fp_output, 'gen_final_' + str(year) + '.pkl')
    else:
        entso_fp = os.path.join(fp_output, 'gen_final_eurostat.pkl')

    with open(entso_fp, 'rb') as handle:
        entso_e_production = pickle.load(handle)

    entso_e_production.replace(0, np.nan, inplace=True)
    entso_mask = entso_e_production.isna().sort_index()
    ef_mask = ef_countries.sort_index().isna()

    # remove these technologies as they have no equivalents in ecoinvent
    remove_tecs_list = ['Fossil Oil shale', 'Other', 'Other renewable', 'Marine']
    for tec in remove_tecs_list:
        try:
            entso_mask.drop(columns=tec, inplace=True)
        except KeyError:
            print(f"Could not remove {tec}; not in ENTSO-E/Eurostat production matrix")

    entso_mask.index.rename('geography', inplace=True)
    entso_mask.columns.rename('entso_e', inplace=True)

    # Countries for which we have production data, but no hybridized emission factors
    no_ef_countries = set(entso_e_production.index) - set(ef_countries.index)
    print('Countries with production data, but no hybridized emission factors:')
    print(no_ef_countries)
    logging.warning(f'{no_ef_countries} have production data from ENTSO_E but no hybridized emission factors')

    list(set(ef_countries.columns) - set(entso_e_production.index))


    check_df = entso_mask.eq(ef_mask, axis=0)

    # Find countries missing from ecoinvent
    missing_countries = []
    for country, row in check_df.iterrows():
        if not row.any():
            missing_countries.append(country)
    # Optionally, drop them from mask
    check_df.drop(index = missing_countries, inplace=True)

    # Find the technology-country pairs that are missing hybridized emission factors
    countries_missing_ef = {}
    for tec, col in check_df.iteritems():
        temp = check_df.index[check_df[tec] == False].tolist()
        print(f'For {tec}, regionalized, hybridized emission factors are missing for {temp}. Using technological average from available regions instead.')
        temp2 = []
        for country in temp:
            if ef_mask.loc[country, tec]:
                temp2.append(country)
        countries_missing_ef[tec] = temp2#.append(country)
    print(countries_missing_ef)

    no_ef = pd.DataFrame({k:pd.Series(v[:13]) for k, v in countries_missing_ef.items()})


    # ## Calculate LC emissions for production mix for countries with missing production data

#    trade_only = ['AL','BY','HR','LU','MD','MT','RU','TR','UA']
    low_volt = labels.loc[labels['activityName'] == 'market for electricity, low voltage']
    trade_only_mixes_lv = low_volt.loc[low_volt['geography'].isin(trade_only)].index
    lv_mix_ef = D_labels.loc[trade_only_mixes_lv]  # low voltage mixes, not used

    high_volt = labels.loc[labels['activityName'] == 'market for electricity, high voltage']
    trade_only_mixes_hv = high_volt.loc[high_volt['geography'].isin(trade_only)].index
    hv_mix_ef = D_labels.loc[trade_only_mixes_hv]
    missing_mixes = list(set(trade_only) - set(hv_mix_ef.index.get_level_values('geography')))
    for country in missing_mixes:
        if country == 'AD':
            ad = D_labels.loc[high_volt.loc[high_volt['geography']=='FR'].index]
            hv_mix_ef = hv_mix_ef.append(ad)
            hv_mix_ef.index = hv_mix_ef.index.set_levels(hv_mix_ef.index.levels[2].str.replace('FR', 'AD'), level=2)

    lv_mix_ef_comp = D_labels_comp[trade_only_mixes_lv]  # low voltage mixes, not used
    hv_mix_ef_comp = D_labels_comp[trade_only_mixes_hv]

    trade_mixes_comp = lv_mix_ef_comp.join(hv_mix_ef_comp)


    labels.loc[labels['activityName'] == 'electricity, from municipal waste incineration to generic market for electricity, medium voltage']

    # ## Export results
    # Convert emission factors to g CO2-eq/kWh
    ef_countries = ef_countries * 1000
    lv_mix_ef = lv_mix_ef * 1000  # low voltage mixes, not used
    hv_mix_ef = hv_mix_ef * 1000
    trade_mixes_comp = trade_mixes_comp * 1000

    # Calculate mean technology footprints as quick check
    ef_aggregated.replace(0, np.nan).mean(axis=1)

    fp_results = os.path.join(os.path.curdir,'results')

    with pd.ExcelWriter(os.path.join(fp_results, 'hybrid_emission_factors_final.xlsx')) as writer:
        ef_countries.to_excel(writer, sheet_name='country_emission_factors')
        no_ef.to_excel(writer, sheet_name='missing_emission_factors')
        lv_mix_ef.to_excel(writer, sheet_name='ecoinvent ef')
        hv_mix_ef.to_excel(writer, sheet_name='ecoinvent ef_hv')
        trade_mixes_comp.to_excel(writer, sheet_name='trade_mixes_from_ei')
        ei_tec_table.to_excel(writer, sheet_name='correspondence table')
        ef_aggregated.to_excel(writer, sheet_name='mean tech fps')

    # Export to .csv for use in BEV_footprints_calculations
    hv_mix_ef.to_csv(os.path.join(fp_output, 'ecoinvent_ef_hv.csv'))  # used in BEV_footprints_calculations
    ef_countries.to_csv(os.path.join(fp_output, 'country_emission_factors.csv'))  # used in clean_impact_factors
    no_ef.to_csv(os.path.join(fp_output, 'missing_emission_factors.csv'))  # used in clean_impact_factors
#    lv_mix_ef.to_csv(os.path.join(fp_results, 'ecoinvent_ef_lv.csv'))

    return ef_countries, no_ef

def clean_impact_factors(year, trade_only, use_entso):
    if use_entso:
        gen_pickle = 'gen_final_' + str(year) + '.pkl'
        trade_pickle = 'trade_final_' + str(year) + '.pkl'
    else:
        gen_pickle = 'gen_final_eurostat.pkl'
        trade_pickle = 'trade_final_eurostat.pkl'
    with open(os.path.join(fp_output, gen_pickle), 'rb') as handle:
        gen_df = pickle.load(handle)

    gen_df.replace(0, np.nan, inplace=True)

    with open(os.path.join(fp_output, trade_pickle), 'rb') as handle:
        trade_df = pickle.load(handle)

    try:
        print('Calculating hybridized emission factors from pyLCAIO')
        # gen_df not modified in hybrid_emission_factors, just used to determine relevant labels
        if use_entso:
            ef, missing_factors = hybrid_emission_factors(trade_only, use_entso, year)
        else:
            ef, missing_factors = hybrid_emission_factors(trade_only, use_entso)
    except Exception as e:
        print('Calculating from pyLCAIO failed. Importing previously calculated emission factors instead')
        print(e)
        # import ready-calculated emission factors if no access to pylcaio object
        ef = pd.read_csv(os.path.join(fp_output, 'country_emission_factors.csv'), index_col=[0])
        missing_factors = pd.read_csv(os.path.join(fp_output, 'missing_emission_factors.csv'), index_col=[0])

    """ Fix missing emission factors """
    # Note that these disregard whether or not there is actual production
    ef['Fossil Oil shale'] = ef['Fossil Oil'] # Proxy shale oil with conventional oil
    ef['Waste'] = 0  # Allocate incineration emissions to waste treatment as per ecoinvent
    ef['Marine'] = 0

    # Put in proxy for missing regions and technologies
    missing_factors.dropna(how='all', axis=1, inplace=True)
    missing_factors_dict = {}
    for tec, col in missing_factors.iteritems():
        missing_factors_dict[tec] = list(col.dropna(how='any', axis=0).values)

    # countries with no geo-specific factors
    missing_countries = list(set(gen_df.index) - set(ef.index))
    for country in missing_countries:
        ef = ef.append(pd.Series(name=country))

    temp_gen_df = gen_df.loc[missing_countries]
    temp_gen_df = temp_gen_df.replace(0, np.nan).dropna(how='all', axis=1)
    temp_dict = {}
    for tec, col in temp_gen_df.items():
        temp = temp_gen_df[tec] > 0
        temp_ind = temp.index
        temp_dict[tec] = temp_ind[temp.values].to_list()

    """ Merge dictionaries and keep values of common keys in list"""
    def mergeDict(dict1, dict2):
        dict3 = {**dict1, **dict2}
        for key, value in dict3.items():
                if key in dict1 and key in dict2:
                    dict3[key] = (dict1[key] + dict2[key])
        return dict3

    # Merge dictionaries and add values of common keys in a list
    missing_factors_dict2 = mergeDict(missing_factors_dict, temp_dict)
    missing_factors_dict2

    # Use continental arithmetic average of same technology
    for key, countries in missing_factors_dict2.items():
        if not key=='Other' and not key=='Other renewable': # these are dealt with below
            ef[key].loc[countries] = ef[key].mean()

    # Make 'other' and 'other renewable' approximations
    renewable_dict = {'renewable': ['Biomass', 'Geothermal', 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
                                    'Hydro Water Reservoir','Solar','Waste','Wind Onshore', 'Wind Offshore','Marine'],
                    'non-renewable': ['Fossil Gas','Fossil Hard coal','Fossil Oil','Fossil Brown coal/Lignite','Nuclear','Fossil Coal-derived gas','Fossil Oil shale','Fossil Peat']}

    other_tec = gen_df.loc[:,renewable_dict['non-renewable']]
    other_wf = other_tec.div(other_tec.sum(axis=1), axis=0) # determine generation shares for each country

    other_renew = gen_df.loc[:,renewable_dict['renewable']]
    other_renew_wf = other_renew.div(other_renew.sum(axis=1), axis=0)

    ef['Other'] = (ef*other_wf).sum(axis=1)
    ef['Other renewable'] = (ef*other_renew_wf).sum(axis=1)


    # Sort the indices of production, rtade and emission factor matrices before exxport
    def sort_indices(df):
        df.sort_index(axis=0, inplace=True)
        df.sort_index(axis=1, inplace=True)

    sort_indices(trade_df)
    sort_indices(gen_df)
    sort_indices(ef)

    # Export all data
    # .csv for use in BEV_footprints_calculations.py
    if use_entso:
        trade_df.to_csv(os.path.join(fp_output,'trades_' + str(year) + '.csv'))
        gen_df.to_csv(os.path.join(fp_output,'ENTSO_production_volumes_' + str(year) + '.csv'))
        ef.to_csv(os.path.join(fp_output,'final_emission_factors_' + str(year) + '.csv'))
        excel_filename = 'entsoe_export_final_' + str(year) + '.xlsx'
    else:
        trade_df.to_csv(os.path.join(fp_output,'trades_eurostat.csv'))
        gen_df.to_csv(os.path.join(fp_output,'production_volumes_eurostat.csv'))
        ef.to_csv(os.path.join(fp_output,'final_emission_factors_eurostat.csv'))
        excel_filename = 'eurostat_export_final.xlsx'

    with pd.ExcelWriter(os.path.join(fp_results, excel_filename)) as writer:
        trade_df.to_excel(writer, 'trade')
        gen_df.to_excel(writer, 'generation')
        ef.to_excel(writer,'new ef')
