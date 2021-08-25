# coding: utf-8

"""Mapping of production and consumption mixes in Europe and their effect on
the carbon footprint of electric vehicles

 This code performs the following:
 -  Import data from ENTSO-E (production quantities, trades relationships)
 -  Calculates the production and consumption electricity mixes for European countries
 -  Calculates the carbon footprint (CF) for the above electricity mixes](#CF_el)
 -  Calculates the production, use-phase and end-of-life emissions for battery electric vehicles (BEVs) under
    the following assumptions:](#BEV_calcs)
     -  Production in Korea (with electricity intensity 684 g CO2-eq/kWh)
     -  Use phase uses country-specific production and consumption mix
     -  End-of-life emissions static for all countries

 Requires the following files for input:
  - ENTSO_production_volumes.csv (from hybridized_impact_factors.py)
  - final_emission_factors.csv (from hybridized_impact_factors.py)
  - trades.csv (from hybridized_impact_factors.py)
  - trade_ef_hv.csv (from hybridized_impact_factors.py)
  - API_EG.ELC.LOSS.ZS_DS2_en_csv_v2_673578.csv (transmission losses, from OECD)
  - car_specifications.xlsx
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import country_converter as coco

import logging

#%% Main function
def run_calcs(run_id, year, no_ef_countries, export_data=True, include_TD_losses=True, BEV_lifetime=180000, ICEV_lifetime=180000, flowtrace_el=True, allocation=True, production_el_intensity=679, incl_ei=False, energy_sens=False):
    """Run all electricity mix and vehicle calculations and exports results."""
    # Korean el-mix 679 g CO2/kWh, from ecoinvent

    fp = os.path.curdir
    production, trades, trade_ef, country_total_prod_disagg, country_total_cons_disagg, g_raw, C = load_prep_el_data(fp, year)
    codecheck_file, elmixes, trade_only, country_el, CFEL, CFCI = el_calcs(flowtrace_el, run_id, fp, C, production, country_total_prod_disagg, country_total_cons_disagg, g_raw, trades, trade_ef, include_TD_losses, incl_ei, export_data)  # Leontief electricity calculations
    results_toSI, ICEV_total_impacts, ICEV_prodEOL_impacts, ICEV_op_int = BEV_calcs(fp, country_el, production, elmixes, BEV_lifetime, ICEV_lifetime, production_el_intensity, CFCI, allocation, energy_sens)

    SI_fp = export_SI(run_id, results_toSI, production, trades, C, CFEL, no_ef_countries)
    pickle_results(run_id, results_toSI, CFEL, ICEV_total_impacts, codecheck_file, export_data)

    return results_toSI['BEV footprint'].xs('Consumption mix', level=1, axis=1), ICEV_prodEOL_impacts, ICEV_op_int, SI_fp


#%% Load and format data for calculations

def load_prep_el_data(fp, year):
    """Load electricity data and emissions factors."""
    fp_output = os.path.join(fp, 'output')

    # Output from bentso.py
    filepath_production = os.path.join(fp_output, 'entsoe', 'ENTSO_production_volumes_'+ str(year) +'.csv')
    filepath_intensities = os.path.join(fp_output, 'final_emission_factors_'+ str(year) +'.csv')
    filepath_trades = os.path.join(fp_output, 'entsoe', 'trades_'+ str(year) +'.csv')

    filepath_tradeonly_ef = os.path.join(fp_output, 'ecoinvent_ef_hv.csv')

    # read in production mixes (annual average)
    production = pd.read_csv(filepath_production, index_col=0)
    production.rename_axis(index='', inplace=True)

    # matrix of total imports/exports of electricity between regions; aka Z matrix
    trades = pd.read_csv(filepath_trades, index_col=0)
    trades.fillna(0, inplace=True)  # replace np.nan with 0 for matrix math, below

    # manually remove Cyprus for now
    production.drop(index='CY', inplace=True)
    trades = trades.drop(columns='CY').drop(index='CY')

    imports = trades.sum(axis=0)
    exports = trades.sum(axis=1)

    """ Make into sum of production and production + import - export"""
    country_total_prod_disagg = production.sum(axis=1)
    country_total_cons_disagg = country_total_prod_disagg + imports - exports

    waste = (production['Waste'] / production.sum(axis=1))
    waste_min = waste[waste > 0].min()
    waste_max = waste.max()

    g_raw = production.sum(axis=1)  # Vector of total electricity production (regionalized)

    """ Read power plant CO2 intensities [tech averages] """
    # average technology CO2 intensities (i.e., non-regionalized)
    all_C = pd.read_csv(filepath_intensities, index_col=0)
    all_C.drop(index='CY', inplace=True)

    # use ecoinvent factors for these countries as a proxy to calculate consumption mixes for receiving countries
    trade_ef = pd.read_csv(filepath_tradeonly_ef, index_col=[0, 1, 2, 3], header=[0])
    trade_ef.index = trade_ef.index.droplevel([0, 1, 3])  # remove DSID, activityName and productName (leaving geography)
    trade_ef.index.rename('geo', inplace=True)
    trade_ef.columns = ['emission factor']

    # Generate regionalized tech generation matrix
    C = all_C.T
    C.sort_index(axis=1, inplace=True)
    C.sort_index(axis=0, inplace=True)

    return production, trades, trade_ef, country_total_prod_disagg, country_total_cons_disagg, g_raw, C

#%% el_calcs

def el_calcs(flowtrace_el, run_id, fp, C, production, country_total_prod_disagg, country_total_cons_disagg, g_raw, trades, trade_ef, include_TD_losses, incl_ei, export_data):
    fp_data = os.path.join(fp, 'data')

    # Make list of full-country resolution
    original_countries = list(production.index)

    # Make list of aggregated countries (affects Nordic countries + GB (UK+NI))
    # read 3-letter ISO codes
    countries = list(trades.index)

    """ Calculates national production mixes and consumption mixes using Leontief assumption """
    # Start electricity calculations (ELFP.m)
    # Calculate production and consumption mixes

    # Carbon intensity of production mix
    CFPI_no_TD = pd.DataFrame(production.multiply(C.T).sum(axis=1) / production.sum(axis=1), columns=['Production mix intensity'])  # production mix intensity without losses
    CFPI_no_TD.fillna(0, inplace=True)

     # List of countries that have trade relationships, but no production data
    trade_only = list(set(trades.index) - set(production.loc[production.sum(axis=1) > 0].index))

    # Add ecoinvent proxy emission factors for trade-only countries
    logging.info('Replacing missing production mix intensities with values from ecoinvent:')
    for country in trade_only:
        if CFPI_no_TD.loc[country, 'Production mix intensity'] == 0:
            logging.info(country)
            CFPI_no_TD.loc[country] = trade_ef.loc[country].values
    i = country_total_cons_disagg.size  # Number of European regions

    g = g_raw
    g = g.sort_index()  # total generation vector (local production for each country)

    total_imported = trades.sum(axis=0)  # sum rows for total imports
    total_exported = trades.sum(axis=1)  # sum columns for total exports
    y = total_imported + g - total_exported  # total final demand (consumption) of electricity

    q = g + total_imported  # vector of total consumption
    q.replace(np.nan, 0, inplace=True)

    if flowtrace_el:
        # For flow tracing approach: make Leontief production functions (normalize columns of A)
        # normalized trade matrix quadrant
        Atmx = pd.DataFrame(np.matmul(trades, np.linalg.pinv(np.diag(q))))

        # normalized production matrix quadrant
        Agen = pd.DataFrame(np.diag(g) * np.linalg.pinv(np.diag(q)), index=countries, columns=countries)  # coefficient matrix, generation

        # "Trade" Leontief inverse
        # Total imports from region i to j per unit demand on j
        Ltmx = pd.DataFrame(np.linalg.pinv(np.identity(i) - Atmx), trades.columns, trades.index)

        # Production in country i for trade to country j
        # Total generation in i (rows) per unit demand j
        Lgen = pd.DataFrame(np.matmul(Agen, Ltmx), index=Agen.index, columns=Ltmx.columns)

        y_diag = pd.DataFrame(np.diag(y), index=countries, columns=countries)

        # total imports for given demand
        Xtmx = pd.DataFrame(np.matmul(np.linalg.pinv(np.identity(i) - Atmx), y_diag))

        # Total generation to satisfy demand (consumption)
        Xgen = np.matmul(np.matmul(Agen, Ltmx), y_diag)
        Xgen.sum(axis=0)
        Xgen_df = pd.DataFrame(Xgen, index=Agen.index, columns=y_diag.columns)

        # ### Check electricity generated matches demand
        totgen = Xgen.sum(axis=0)
        r_gendem = totgen / y  # All countries should be 1


        #%% Generation techonlogy matrix

        # TC is a country-by-generation technology matrix - normalized to share of total domestic generation, i.e., normalized generation/production mix
        # technology generation, kWh/ kWh domestic generated electricity
        TC = pd.DataFrame(np.matmul(np.linalg.pinv(np.diag(g)), production), index=g.index, columns=production.columns)
        TCsum = TC.sum(axis=1)  # Quality assurance - each country should sum to 1

        # Calculate technology generation mix in GWh based on production in each region
        TGP = pd.DataFrame(np.matmul(TC.transpose(), np.diag(g)), index=TC.columns, columns=g.index)  #.== production

        # Carbon intensity of consumption mix
        CFCI_no_TD = pd.DataFrame(np.matmul(CFPI_no_TD.T.values, Lgen), columns=CFPI_no_TD.index).T

    else:
        # Use grid-average assumption for trade
        prod_emiss = production.multiply(C.T).sum(axis=1)
        trade_emiss = (pd.DataFrame(np.diag(CFPI_no_TD.iloc(axis=1)[0]), index=CFPI_no_TD.index, columns=CFPI_no_TD.index)).dot(trades)
        CFCI_no_TD = pd.DataFrame((prod_emiss + trade_emiss.sum(axis=0) - trade_emiss.sum(axis=1)) / y)

    CFCI_no_TD.columns = ['Consumption mix intensity']

    # use ecoinvent for missing countries
    if incl_ei:
        CFCI_no_TD.update(trade_ef.rename(columns={'emission factor':'Consumption mix intensity'}))


    #%% Calculate losses
    # Transpose added after removing country aggregation as data pre-treatment
    if include_TD_losses:
        # Calculate technology characterization factors including transmission and distribution losses
        # First, read transmission and distribution losses, downloaded from World Bank economic indicators (most recent values from 2014)
        if isinstance(include_TD_losses, float):
            TD_losses = include_TD_losses  # apply constant transmission and distribution losses to all countries
        elif isinstance(include_TD_losses, bool):
            losses_fp = os.path.join(fp_data, 'API_EG.ELC.LOSS.ZS_DS2_en_csv_v2_673578.csv')
            try:
                TD_losses = pd.read_csv(losses_fp, skiprows=[0,1,2,3], usecols=[1, 58], index_col=0)
                TD_losses = TD_losses.iloc[:, -7:].dropna(how='all', axis=1)
                TD_losses = TD_losses.apply(lambda x: x / 100 + 1)  # convert losses to a multiplicative factor

                # ## Calculate total national carbon emissions from el  - production and consumption mixes
                TD_losses.index = coco.convert(names=TD_losses.index.tolist(), to='ISO2', not_found=None)
                TD_losses = TD_losses.loc[countries]
                TD_losses = pd.Series(TD_losses.iloc[:, 0])
            except:
                print("Warning! Transmission and distribution losses input files not found!")
                TD_losses = pd.Series(np.zeros(len(production.index)), index=production.index)
        else:
            print('invalid entry for losses')

        # Caclulate carbon intensity of production and consumption mixes including losses
        CFPI_TD_losses = CFPI_no_TD.multiply(TD_losses, axis=0).dropna(how='any', axis=0)  # apply transmission and distribution losses to production mix intensity
        CFCI_TD_losses = CFCI_no_TD.multiply(TD_losses, axis=0).dropna(how='any', axis=0)

        if len(CFCI_TD_losses) < len(CFPI_TD_losses):
            CFCI_TD_losses = CFCI_no_TD.multiply(TD_losses, axis=0)

        CFPI = CFPI_TD_losses
        CFCI = CFCI_TD_losses
    else:
        CFPI = CFPI_no_TD
        CFCI = CFCI_no_TD

    elmixes = (CFPI.copy()).join(CFCI.copy()).T

    #%%

    # Aggregate multi-nodes to single countries using weighted average of production/consumption as appropriate
    country_total_prod_disagg.columns = ["Total production (TWh)"]
    country_total_prod_disagg.index = original_countries
    country_total_cons_disagg.columns = ["Total consumption (TWh)"]
    country_total_cons_disagg.index = original_countries

    country_el = pd.concat([country_total_prod_disagg, country_total_cons_disagg], axis=1)
    country_el.columns = ['Total production (TWh)', 'Total consumption (TWh)']

    CFEL_mixes = elmixes.T
    CFEL = pd.concat([country_el, CFEL_mixes], axis=1)

    imports = trades.sum(axis=0)
    exports = trades.sum(axis=1)

    CFEL['Trade percentage, gross'] = (imports + exports) / CFEL['Total production (TWh)']
    CFEL['Import percentage'] = imports / CFEL['Total production (TWh)']
    CFEL['Export percentage'] = exports / CFEL['Total production (TWh)']
    CFEL['imports'] = imports
    CFEL['exports'] = exports

    #Calculate total carbon footprint intensity ratio production vs consumption
    rCP = CFCI['Consumption mix intensity'].divide(CFPI['Production mix intensity'])
    rCP.columns = ["ratio consumption:production mix"]

    # Export intermediate variables from calculations for troubleshooting
    if export_data:
        keeper = run_id + "{:%d-%m-%y, %H_%M}".format(datetime.now())
        fp_results = os.path.join(fp, 'results')
        codecheck_file = os.path.join(os.path.abspath(fp_results), 'code_check_' + keeper + '.xlsx')

        writer = pd.ExcelWriter(codecheck_file)

        g.to_excel(writer, "g")
        q.to_excel(writer, "q")
        y.to_excel(writer, 'y')
        if flowtrace_el:
            Atmx.to_excel(writer, "Atmx")
            Agen.to_excel(writer, "Agen")
            Ltmx.to_excel(writer, "LTmx")
            Lgen.to_excel(writer, "Lgen")
            Xtmx.to_excel(writer, "Xtmx")
            TGP.to_excel(writer, "TGP")
        CFPI.T.to_excel(writer, "CFPI")
        CFCI.T.to_excel(writer, "CFCI")
        rCP.to_excel(writer, "rCP")
        C.T.to_excel(writer, "C")
        writer.save()

    return codecheck_file, elmixes, trade_only, country_el, CFEL, CFCI

#%%

def BEV_calcs(fp, country_el, production, elmixes, BEV_lifetime, ICEV_lifetime, production_el_intensity, CFCI, allocation=True, energy_sens=False):
    """Calculate BEV lifecycle emissions."""

    # First, setup calculations
    # read in data
    fp_data = os.path.join(fp, 'data')
    vehicle_fp = os.path.join(fp_data, 'car_specifications.xlsx')
    cars = pd.read_excel(vehicle_fp, sheet_name='veh_emiss', index_col=[0, 1, 2], usecols='A:G')
    cars = cars.sort_index()
    vehicle_CO2 = ["BEV", "ICEV"]

    if energy_sens:
        # if performing the experiment for battery energy demand in manufacturing,
        # update with new energy values
        alt_energy = pd.read_excel(vehicle_fp, sheet_name='alt_energy', index_col=[0,1,2], usecols='A:H')  # column A is scenario name
        if isinstance(energy_sens, str):
            cars.update(alt_energy.loc[energy_sens])

    # Impacts from electricity demand in cell production
    battery_prod_el = production_el_intensity / 1e6 * cars.loc["BEV", "Production el, battery"]  # in t CO2/vehicle
    batt_prod_impacts = cars.loc["BEV", "Production, RObattery"].add(battery_prod_el, fill_value=0).sum(axis=0)

    if allocation:
        alloc_share = BEV_lifetime / ((cars.loc["BEV", "Max EFC", "cycles"] * (cars.loc["BEV", "Batt size", "kWh"]*.9) * 1000) / cars.loc["BEV", "Use phase", "Wh/km"])
    else:
        alloc_share = 1

    alloc_batt_prod_impacts = alloc_share * batt_prod_impacts

    # Total vehicle production impacts - sum of battery emissions + rest of vehicle
    BEV_prod_impacts = cars.loc["BEV", "Production, ROV"] + alloc_batt_prod_impacts

    # Modify for battery production in Europe
    # batt_prod_EU = pd.DataFrame(np.matmul(CFCI.values / 1e6, cars.loc["BEV", "Production el, battery"].values), index=CFCI.index, columns=cars.columns)
    batt_prod_EU = pd.DataFrame(np.matmul((elmixes.T['Consumption mix intensity'].values / 1e6).reshape(-1, 1),
                                          cars.loc["BEV", "Production el, battery"].values),
                                          index=elmixes.columns, columns=cars.columns)

    # Total battery production impacts in Europe
    batt_prod_EU = batt_prod_EU + cars.loc["BEV", "Production, RObattery", "t CO2"]
    alloc_batt_prod_EU = alloc_share * batt_prod_EU

    BEV_prod_EU = pd.DataFrame(index=elmixes.columns, columns=["A", "C", "JC", "JE"])
    BEV_prod_EU = alloc_batt_prod_EU + cars.loc["BEV", "Production, ROV", "t CO2"]
    BEV_prod_EU.columns = pd.MultiIndex.from_product([["EUR production impacts BEV"], BEV_prod_EU.columns, ["Consumption mix"]], names=["", "Segment", "Elmix"])

    # Calculate use phase emissions
    segs = cars.loc['BEV', 'Use phase', 'Wh/km']
    mi = pd.MultiIndex.from_product([list(elmixes.index), list(segs.index)])
    segs = segs.reindex(mi, level=1)
    segs = pd.DataFrame(segs)
    segs.columns = ['a']
    segs = segs.reindex(elmixes.columns, axis=1, method='bfill')

    elmixes_for_calc = elmixes.reindex(mi, level=0, axis=0)

    BEV_use = (segs.multiply(elmixes_for_calc / 1000)).T

    # Add production and EOL intensity for BEVs
    BEV_other = BEV_prod_impacts + cars.loc["BEV", "EOL", "t CO2"].values
    BEV_other_intensity = BEV_other / BEV_lifetime * 1e6  # in g CO2-eq
    BEV_other_intensity.index = ["g CO2/km"]

    # Calculate full lifecycle intensity using production and consumption mixes
    BEVp = pd.DataFrame(BEV_use["Production mix intensity"] + BEV_other_intensity.loc["g CO2/km"])
    BEVc = pd.DataFrame(BEV_use["Consumption mix intensity"] + BEV_other_intensity.loc["g CO2/km"])

    # BEV impacts with production and consumption mixes
    # Check which technology lifetime to use as baseline for use phase
    # (use shortest lifetime between the two for comparison to avoid vehicle replacement)

    # if BEV_lifetime <= ICEV_lifetime:
    #     lifetime = BEV_lifetime
    # elif BEV_lifetime > ICEV_lifetime:
    #     # in the case for BEV lifetimes longer than ICEV lifetimes
    #     lifetime = ICEV_lifetime

    # Calculate total absolute lifecycle emissions in t CO2e
    BEV_impactsp = (BEV_use['Production mix intensity'].T * BEV_lifetime / 1e6).add(BEV_other.loc['t CO2'], axis=0)
    BEV_impactsc = (BEV_use['Consumption mix intensity'].T * BEV_lifetime / 1e6).add(BEV_other.loc['t CO2'], axis=0)

    BEV_impacts = pd.concat([BEV_impactsp.T, BEV_impactsc.T], axis=1, keys=['Production mix', 'Consumption mix'])
    BEV_impacts = BEV_impacts.swaplevel(axis=1, i=0, j=1)
    BEV_impacts.sort_index(level=0, axis=1, inplace=True)

    # Calculate share of production phase in total BEV lifecycle emissions
    BEV_prod_sharesp = BEV_prod_impacts.values / (BEV_impactsp.T)
    BEV_prod_sharesc = BEV_prod_impacts.values / (BEV_impactsc.T)

    BEV_prod_sharesc.columns = pd.MultiIndex.from_product([BEV_prod_sharesc.columns, ['Consumption mix']])

    # Calculate share of use phase in total BEV lifecycle emissions
    BEV_use_sharesp = (BEV_use['Production mix intensity'] * BEV_lifetime / 1e6) / (BEV_impactsp.T)
    BEV_use_sharesc = (BEV_use['Consumption mix intensity'] * BEV_lifetime / 1e6) / (BEV_impactsc.T)

    # Calculate BEV footprints with EUR (domestic) battery production
    BEV_fp_EU = (BEV_prod_EU.add(cars.loc["BEV", "EOL", "t CO2"], level=1, axis=1) / BEV_lifetime * 1e6)

    # Currently, EU production assumes consumption mix, so only examine using consumption mix for both manufacturing and use phase for consistency
    EU_BEVc = BEV_fp_EU.add(BEV_use['Consumption mix intensity'].reindex(BEV_fp_EU.columns, axis=1, level=1), axis=1)
    EU_BEVc.rename(columns={"EUR production impacts BEV": "BEV footprint, EUR production"}, inplace=True)

    # Calculate EU production:Asian production footprint ratios
    fp_ratio = EU_BEVc.divide(BEVc, level=1)

    # Calculate total lifecycle emissions for ICEVs
    ICEV_prodEOL_impacts = cars.loc['ICEV', 'Production', 't CO2'].add(cars.loc['ICEV', 'EOL', 't CO2'], axis=0)

    ICEV_total_impacts = ICEV_prodEOL_impacts.add(cars.loc['ICEV', 'Use phase', 'g CO2/km'] * ICEV_lifetime / 1e6, axis=0)
    ICEV_prod_EOL_fp = ICEV_prodEOL_impacts * 1e6 / ICEV_lifetime

    ICEV_lc_footprint = ICEV_prod_EOL_fp + cars.loc['ICEV', 'Use phase', 'g CO2/km']
    ICEV_total_impacts = pd.DataFrame(ICEV_total_impacts).T  # force to dataframe

    ICEV_lc_shares = cars.loc['ICEV'] / cars.loc['ICEV'].sum(axis=0)


    #%%    # Calculate BEV:ICEV ratios

    ratio_use_prod = BEV_use["Production mix intensity"] / cars.loc["ICEV", "Use phase", "t CO2"]

    # Ratios comparing use phase only
    ratio_use_prod = pd.DataFrame(BEV_use["Production mix intensity"] / cars.loc["ICEV", "Use phase", "t CO2"])
    ratio_use_cons = pd.DataFrame(BEV_use["Consumption mix intensity"] / cars.loc["ICEV", "Use phase", "t CO2"])

    ratio_use_cons = ratio_use_cons.rename_axis("Segment", axis=1)
    ratio_use_cons = pd.concat([ratio_use_cons], keys=["RATIO: use phase"], axis=1)
    ratio_use_cons = pd.concat([ratio_use_cons], keys=["Consumption mix"], names=["Elmix"], axis=1)
    ratio_use_cons = ratio_use_cons.reorder_levels([1,2,0], axis=1)

    # Ratios with lifecycle impacts
    ratiop = pd.DataFrame(BEVp / (ICEV_lc_footprint))
    ratioc = pd.DataFrame(BEVc / (ICEV_lc_footprint))

    # Ratios with EU production
    ratioc_EU_prod = (EU_BEVc["BEV footprint, EUR production"].stack() / (ICEV_lc_footprint)).unstack()
    ratioc_EU_prod = pd.concat([ratioc_EU_prod], keys=["Ratio BEV:ICEV, European BEV production"], axis=1)

    # Extra calculations

    BEV_total_use = BEV_use * BEV_lifetime / 1e6  # absolute lifetime operation emissions

    # Assemble total results table

    # CFEL  - the CO2 footprint of electricity in different European countries based on either i) production or ii) consumption perspective.
    # BEV – the gCO2/km for electric vehicles for i) all EU countries ii) all segments iii) production and consumption el mix.
    # RATIO  - the ratio of the gCO2/km for BEVs vs ICEs for i) all EU countries ii) all segments iii) production and consumption el mix.

    fp = pd.concat({"Production mix": BEVp}, axis=1, names=["Elmix", "Segment"])
    fp = pd.concat([fp, pd.concat({"Consumption mix": BEVc}, axis=1, names=["Elmix", "Segment"])], axis=1)
    fp = fp.swaplevel(axis=1).sort_index(axis=1, level="Segment", sort_remaining=False)
    fp = pd.concat({"BEV footprint": fp}, axis=1)

    RATIOS = pd.concat({"Production mix": ratiop}, axis=1, names=["Elmix", "Segment"])
    RATIOS = pd.concat([RATIOS, pd.concat({"Consumption mix": ratioc}, axis=1, names=["Elmix", "Segment"])], axis=1)
    RATIOS = RATIOS.swaplevel(axis=1).sort_index(axis=1, level="Segment", sort_remaining=False)
    RATIOS = pd.concat({"RATIO BEV:ICEV": RATIOS}, axis=1)

    results = fp.join(pd.concat({'BEV impacts': BEV_impacts}, axis=1, names=['', 'Elmix', 'Segment']))
    results = results.join(RATIOS)
    results = results.join(ratio_use_cons)
    results = results.join(BEV_prod_EU)
    results = results.join(EU_BEVc)
    results = results.join(ratioc_EU_prod)
    results = results.join(pd.concat({"Production as share of total footprint":BEV_prod_sharesc}, axis=1))

    results_toSI = results.copy()

    return results_toSI, ICEV_total_impacts, ICEV_prodEOL_impacts, cars.loc['ICEV', 'Use phase', 'g CO2/km']


#%% Export functions
def export_SI(run_id, results_toSI, production, trades, C, CFEL, no_ef_countries):
    """Format dataframes for export to SI tables."""

    drop_countries = production.loc[production.sum(axis=1)==0].index

    CFEL_toSI = CFEL[['Production mix intensity', 'Consumption mix intensity']]
    CFEL_toSI = CFEL_toSI.round(0)
    CFEL_toSI.loc[drop_countries, 'Consumption mix intensity'] = '-'  # remove ecoinvent-based countries
    CFEL_toSI.sort_index(inplace=True)

    country_intensities = C.round(0).fillna(value='-').T
    country_intensities.drop(index=drop_countries, inplace=True)

    production.drop(index=drop_countries, inplace=True)  # drop countries with no production data
    production = production.round(2).replace(0, np.nan).fillna(value='-')

    trades_forSI = trades.round(2).replace(0, np.nan).fillna(value='-')
    trades_forSI = pd.concat([trades_forSI], keys=['Exporting countries'])
    trades_forSI = pd.concat([trades_forSI], keys=['Importing countries'], axis=1)
    trades_forSI.index = trades_forSI.index.rename([None, None])

    trade_pct = CFEL['Trade percentage, gross']
    trade_pcti = CFEL['Import percentage']
    trade_pcte = CFEL['Export percentage']
    trade_pct_toSI = pd.concat([trade_pct, trade_pcti, trade_pcte], axis=1)
    trade_pct_toSI.replace(np.inf, np.nan, inplace=True)
    trade_pct_toSI.dropna(how='all', inplace=True)

    # Export results for SI tables
    keeper = run_id + " {:%d-%m-%y, %H_%M}".format(datetime.now())
    fp_results = os.path.join(os.path.curdir, 'results')

    excel_dict = {'Table S1 - El footprints': 'Data from Figure 2 in manuscript. Calculated national production and consumption mix hybridized lifecycle carbon intensities in g CO2-eq/kWh. Shaded values indicate countries with no production data; consumption mixes are therefore not calculated, and production intensity is obtained from ecoinvent 3.5.',
                  'Table S2 - intensity matrix':'Regionalized lifecycle carbon intensities of electricity generation technologies in g CO2-eq/kWh. Shaded cells indicate proxy data used (see Methods)',
                  'Table S3 - Production mix': ' Electricity generation mixes (2020), in TWh',
                  'Table S4 - trades': 'Trades between studied countries (2020), in TWh/year. Countries denoted in red italics are trade-only and do not have production data from ENTSO-E; ecoinvent values for production mix used for these.',
                  'Table S5 - trade shares':'Total (gross) electricity traded, total imports of electricity and total exports of electricity relative to domestic production. Used in colorbar, Figure 2',
                  'Table S6 - BEV fp':'Data from Figure 3 in manuscript. BEV carbon intensities in (g CO2-eq/km) for consumption electricity mix',
                  'Table S7 - Prod share of fp':'Data from Figure 4 in manuscript. Contribution of vehicle production emissions to total carbon footprint',
                  'Table S8 - Abs BEV impacts':'Data used in Figure 5 in manuscript. Regionalized total lifecycle emissions from BEV in t CO2-eq, with 180 000 km lifetime, using consumption mixes',
                  'Table S9 - Ratio':'Ratio of BEV:ICEV carbon footprints using consumption electricity mix',
                  'Table S10 - EUR prod imp':'Production impacts of BEVs with domestic battery production using consumption electricity mix in t CO2-eq',
                  'Table S11 - EUR prod fp':'Data from Figure 7 in manuscript. Lifecycle BEV footprint with domestic battery production using consumption electricity mix, in g CO2-eq/km, and % change in lifecycle BEV impact from batteries with Asian production.',
                  }

    results_filepath = os.path.join(fp_results, 'SI_results ' + keeper + '.xlsx')

    results_toSI.drop(index=drop_countries, inplace=True) # remove ecoinvent-based countries
    # select parts of results_toSI DataFrame for each table
    table6 = results_toSI.loc(axis=1)['BEV footprint', :, 'Consumption mix'].round(0)
    table7 = results_toSI.loc(axis=1)['Production as share of total footprint', :,'Consumption mix']
    table8 = results_toSI.loc(axis=1)['BEV impacts', :, 'Consumption mix'].round(1)
    table9 = results_toSI.loc(axis=1)['RATIO BEV:ICEV', :, 'Consumption mix'].round(2)
    table10 = results_toSI.loc(axis=1)['EUR production impacts BEV', :, 'Consumption mix'].round(1)
    table11 = results_toSI.loc(axis=1)['BEV footprint, EUR production', :, 'Consumption mix'].round(0)

    # append data for building Figure 7 in mansucript to Table 11
    A_diff = (results_toSI['BEV footprint, EUR production', 'A', 'Consumption mix'] -
              results_toSI['BEV footprint', 'A', 'Consumption mix']) / results_toSI['BEV footprint', 'A', 'Consumption mix']
    F_diff = (results_toSI['BEV footprint, EUR production', 'JE', 'Consumption mix'] -
              results_toSI['BEV footprint', 'JE', 'Consumption mix']) / results_toSI['BEV footprint', 'JE', 'Consumption mix']
    diff_cols = pd.MultiIndex.from_product([['% change from Asian production'], ['A', 'JE'],['']])
    df = pd.DataFrame([A_diff, F_diff], index=diff_cols).T
    table11 = pd.concat([table11, df], axis=1)

    fig_data = pd.DataFrame([A_diff, F_diff], index=['A segment', 'F segment'])
    # reorder technologies in Tables S2 and S3 to place "other" categories at end
    tec_order = ['Biomass',
                 'Fossil Brown coal/Lignite',
                 'Fossil Coal-derived gas',
                 'Fossil Gas',
                 'Fossil Hard coal',
                 'Fossil Oil',
                 'Fossil Oil shale',
                 'Fossil Peat',
                 'Geothermal',
                 'Hydro Pumped Storage',
                 'Hydro Run-of-river and poundage',
                 'Hydro Water Reservoir',
                 'Marine',
                 'Nuclear',
                 'Solar',
                 'Waste',
                 'Wind Offshore',
                 'Wind Onshore',
                 'Other',
                 'Other renewable'
                 ]

    country_intensities = country_intensities.reindex(labels=tec_order, axis=1)
    country_intensities.index = country_intensities.index.rename(None)
    production = production.reindex(labels=tec_order, axis=1)

    # Build dictionary of cells to be shaded for Table S2. Keys are columns,
    # items are countries (may be a list)
    shade_dict =  {key: val for key, val in no_ef_countries.items() if len(val)}

    # Write to Excel
    writer = pd.ExcelWriter(results_filepath)

    # Establish cell formatting styles
    header = writer.book.add_format()
    header.set_font_color('#0070C0')
    header.set_bold(True)

    pct_format = writer.book.add_format({'num_format': '#%'})

    shade_cell = writer.book.add_format()
    shade_cell.set_pattern(1)
    shade_cell.set_bg_color('#C0C0C0')

    rot = writer.book.add_format()
    rot.set_rotation(90)
    rot.set_bold(True)
    rot.set_align('center')
    rot.set_align('vcenter')

    wrap = writer.book.add_format()
    wrap.set_text_wrap(True)

    df_header = writer.book.add_format()
    df_header.set_text_wrap(True)
    df_header.set_align('center')
    df_header.set_bold(True)
    df_header.set_border(1)

    ital_red = writer.book.add_format()
    ital_red.set_font_color('#C00000')
    ital_red.set_italic(True)

    # write data to Excel
    CFEL_toSI.to_excel(writer, 'Table S1 - El footprints', startrow=2)
    country_intensities.to_excel(writer, 'Table S2 - intensity matrix',startrow=2)
    production.to_excel(writer, 'Table S3 - Production mix', startrow=2)
    trades_forSI.to_excel(writer, 'Table S4 - trades', startrow=2)
    trade_pct_toSI.to_excel(writer, 'Table S5 - trade shares', startrow=2)
    table6.droplevel(level=[2], axis=1).to_excel(writer, 'Table S6 - BEV fp', startrow=2)
    table7.droplevel(level=[2], axis=1).to_excel(writer, 'Table S7 - Prod share of fp',  startrow=2)
    table8.droplevel(level=[2], axis=1).to_excel(writer, 'Table S8 - Abs BEV impacts', startrow=2)
    table9.droplevel(level=[2], axis=1).to_excel(writer, 'Table S9 - Ratio', startrow=2)
    table10.droplevel(level=[2], axis=1).to_excel(writer, 'Table S10 - EUR prod imp', startrow=2)
    table11.droplevel(level=[2], axis=1).to_excel(writer, 'Table S11 - EUR prod fp', startrow=2)


    for sheet, caption in excel_dict.items():
        worksheet = writer.sheets[sheet]
        worksheet.write_string(0, 0, sheet.split(' -')[0], header)
        worksheet.merge_range('C1:L1', caption, wrap)
        worksheet.set_row(0, 60)  # adjust row height for wrapped caption

        if sheet.find('share') >= 0:
            worksheet.set_column('B:E', None, pct_format)

        if sheet.find('Table S1 ') >= 0:
            # shade countries with production mixes from ecoinvent
            worksheet.set_column('B:C', 12.5)
            for col_num, text_header in enumerate(CFEL_toSI.columns.values):
                worksheet.write(2, 1+col_num, text_header, df_header)

            worksheet.conditional_format('B2:B50',
                                         {'type':'formula',
                                          'criteria': '=$C2="-"',
                                          'format': shade_cell
                                          })
            worksheet.set_row(2, 30)

        if sheet.find('Table S2') >= 0:
            worksheet.set_column('B:U', 11)
            for col_num, text_header in enumerate(country_intensities.columns.values):
                worksheet.write(2, 1+col_num, text_header, df_header)
            worksheet.set_row(2, 60)  # adjust row height to accomodate wrapped text

            # shade entire columns
            worksheet.conditional_format('B4:U35',
                                         {'type':'formula',
                                         'criteria':'=OR(B$3="Fossil Oil shale", B$3="Other", B$3="Other renewable")',
                                         'format': shade_cell})
            # shade specific cells
            for column, row in shade_dict.items():
                worksheet.conditional_format('B4:U35',
                                             {'type':'formula',
                                             'criteria':build_conditional(column, row),
                                             'format': shade_cell})
        if sheet.find('Table S3') >= 0:
            worksheet.set_column('B:U', 11)
            for col_num, text_header in enumerate(production.columns.values):
                worksheet.write(2, 1+col_num, text_header, df_header)
            worksheet.set_row(2, 60)  # adjust row height to accomodate wrapped text

        if sheet.find('Table S4') >= 0:
            worksheet.write_string('A6', 'Exporting countries', rot)
            worksheet.set_column('A:A', 3)
            worksheet.set_column('C:AO', 6)
            for country in drop_countries:
                # highlight countries that only have trading data in ENTSO
                worksheet.conditional_format('C4:AO4',
                                             {'type':'formula',
                                              'criteria': f'=COUNTIF(C$4,"{country}")',
                                              'format': ital_red})
                worksheet.conditional_format('B5:B45',
                                             {'type':'formula',
                                              'criteria': f'=COUNTIF($B5,"{country}")',
                                              'format': ital_red})

        if sheet.find('Table S11') >= 0:
            worksheet.set_column('F:G', None, pct_format)
            for col_num, text_header in enumerate(table11.columns.get_level_values(0)):
                worksheet.write(2, 1+col_num, text_header, df_header)
            worksheet.set_row(2, 30)  # adjust row height to accomodate wrapped text

    writer.save()

    return results_filepath

def build_conditional(column, row):
    """Build string for conditional formatting formula for exporting to Excel."""
    substring = ''
    if isinstance(row, list):
        for country in row:
            substring = substring + f'$A4="{country}",'
        substring = 'OR(' + substring[:-1] + ')' # remove last comma, add closing parenthesis
        formula_string = f'=AND(B$3="{column}",' + substring + ')'
    else:
        formula_string = f'=AND(B$3="{column}",$A4="{row}")'
    return formula_string


def pickle_results(run_id, results_toSI, CFEL, ICEV_total_impacts, codecheck_file, export_data):
    """Pickle results for reloading later and export data to Excel."""
    # Export for figures
    fp = os.path.abspath(os.path.curdir)
    fp_output = os.path.join(os.path.curdir, 'output')
    os.chdir(fp_output)

    # convert to percent form for figures
    results_toSI.loc(axis=1)['Production as share of total footprint', :,'Consumption mix'] *= 100

    keeper = run_id + ' country-specific indirect'
    results_toSI.to_pickle(keeper + '_BEV.pkl')
    CFEL.to_pickle(keeper + '_el.pkl')
    ICEV_total_impacts.to_pickle(keeper + '_ICEV_impacts.pkl')

    if export_data:
        with pd.ExcelWriter(codecheck_file, engine="openpyxl", mode='a') as writer:
            CFEL.to_excel(writer, 'CFEL - electricity')
            results_toSI.to_excel(writer, 'car footprints')
            writer.save()
    os.chdir(fp)