
# coding: utf-8

"""Mapping of production and consumption mixes in Europe and their effect on 
the carbon footprint of electric vehicles

 This code performs the following:
 -  Import data from ENTSO-E (production quantities, trades relationships)
 -  Calculates the production and consumption electricity mixes for European countries
 -  Calculates the carbon footprint (CF) for the above electricity mixes](#CF_el)
 -  Calculates the production, use-phase and end-of-life emissions for battery electric vehicles (BEVs) under the following assumptions:](#BEV_calcs)
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
	  - road_eqs_carage.xls
"""
#%%

import os
from datetime import datetime

import numpy as np
import pandas as pd
import country_converter as coco

export_data = True
export_figures = True
include_TD_losses = True

#%%
fp = os.path.curdir
fp_data = os.path.join(fp, 'data')
fp_output = os.path.join(fp, 'code output')
fp_results = os.path.join(fp, 'results')
#fp_data = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'Data')
#fp_results = os.path.join(os.path.pardir, os.path.pardir, 'Results')
fp_figure = os.path.join(fp_results, 'figures')

# Output from bentso.py
filepath_production = os.path.join(fp_output, 'ENTSO_production_volumes.csv')
filepath_intensities = os.path.join(fp_output, 'final_emission_factors.csv')
filepath_trades = os.path.join(fp_output, 'trades.csv')
filepath_tradeonly_ef = os.path.join(fp_output, 'AL_HR_LU_TR_ef_hv.csv')


#%%

# read in production mixes
production = pd.read_csv(filepath_production, index_col=0)
production.rename_axis(index='', inplace=True)

# matrix of total imports/exports of electricity between regions; aka Z matrix
trades = pd.read_csv(filepath_trades, index_col=0)

# manually remove Cyprus for now
production.drop(index='CY', inplace=True)
trades = trades.drop(columns='CY').drop(index='CY')

imports = trades.sum(axis=0)
exports = trades.sum(axis=1)

# List of countries that have trade relationships, but no production data
trade_only = list(set(trades.index) - set(production.loc[production.sum(axis=1) > 0].index))

""" Make into sum of production and production + import - export"""
country_total_prod_disagg = production.sum(axis=1)
country_total_cons_disagg = country_total_prod_disagg + imports - exports

country_total_cons = country_total_cons_disagg.copy()
country_total_prod = country_total_prod_disagg.copy()

#%%

waste = (production['Waste']/production.sum(axis=1))
waste_min = waste[waste > 0].min()
waste_max = waste.max()


#%%

g_raw = production.sum(axis=1) # Vector of total electricity production (regionalized)

#%%
# ### Make list of countries for remaining operations

# Make list of full-country resolution
original_countries = list(production.index) # length=42

# Make list of aggregated countries (affects Nordic countries + GB (UK+NI))
# read 3-letter ISO codes
countries = list(trades.index)


#%%

# ### Read power plant CO2 intensities [tech averages]

## This cell for average technology CO2 intensities (i.e., non-regionalized)
all_C = pd.read_csv(filepath_intensities, index_col=0)
all_C.drop(index='CY', inplace=True)

# use ecoinvent factors for these countries as a proxy to calculate consumption mixes for receiving countries
trade_ef = pd.read_csv(filepath_tradeonly_ef, index_col=[0,1,2,3], header=None)
trade_ef.index = trade_ef.index.droplevel([0,1,3])
trade_ef.index.rename('geo', inplace=True)
trade_ef.columns = ['emission factor']

# ### Generate regionalized tech generation matrix
C = all_C.T
C.sort_index(axis=1, inplace=True)
C.sort_index(axis=0, inplace=True)


#%%

# # Calculate technology characterization factors including transmission and distribution losses
# First, read transmission and distribution losses, downloaded from World Bank economic indicators (most recent values from 2014)
losses_fp = os.path.join(fp_data, 'API_EG.ELC.LOSS.ZS_DS2_en_csv_v2_673578.csv')
try:
    TD_losses = pd.read_csv(losses_fp, skiprows=[0,1,2,3], usecols=[1,58], index_col=0)
    TD_losses = TD_losses.iloc[:,-7:].dropna(how='all', axis=1)
    TD_losses = TD_losses.apply(lambda x: x/100+1) # convert losses to a multiplicative factor

    # ## Calculate total national carbon emissions from el  - production and consumption mixes
    TD_losses.index = coco.convert(names=TD_losses.index.tolist(), to='ISO2', not_found=None)
    TD_losses = TD_losses.loc[countries]
    TD_losses = pd.Series(TD_losses.iloc[:,0])
except:
    display("Warning! Transmission and distribution losses input files not found!")
    TD_losses = pd.Series(np.zeros(len(production.index)), index=production.index)


#%%

# # Start electricity calculations (ELFP.m)
# ### Calculate production and consumption mixes

i = country_total_cons_disagg.size # Number of European regions

g = g_raw
g = g.sort_index() # total generation vector (local production for each country)

total_imported = trades.sum(axis=0) # sum rows for total imports
total_exported = trades.sum(axis=1) # sum columns for total exports
y = total_imported + g - total_exported # total final demand (consumption) of electricity

q = g + total_imported # vector of total consumption
q.replace(np.nan, 0, inplace=True)


#%%

# ### Make Leontief production functions (normalize columns of A)

# normalized trade matrix quadrant
Atmx = pd.DataFrame(np.matmul(trades, np.linalg.pinv(np.diag(q))))

# normalized production matrix quadrant
Agen = pd.DataFrame(np.diag(g)*np.linalg.pinv(np.diag(q)), index=countries, columns=countries)  # coefficient matrix, generation

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


#%%

# ### Check electricity generated matches demand
totgen = Xgen.sum(axis=0)
r_gendem = totgen / y  # All countries should be 1

#%%

# ### Generation techonlogy matrix

# TC is a country-by-generation technology matrix - normalized to share of total domestic generation, i.e., normalized generation/production mix
# technology generation, kWh/ kWh domestic generated electricity
TC = pd.DataFrame(np.matmul(np.linalg.pinv(np.diag(g)), production), index=g.index, columns=production.columns)
TCsum = TC.sum(axis=1)  # Quality assurance - each country should sum to 1

# Calculate technology generation mix in GWh based on production in each region
TGP = pd.DataFrame(np.matmul(TC.transpose(), np.diag(g)), index=TC.columns, columns=g.index)  #...== production


#%%

# Carbon intensity of production mix
CFPI_no_TD = pd.DataFrame(production.multiply(C.T).sum(axis=1)/production.sum(axis=1), columns=['Production mix intensity']) # production mix intensity without losses
CFPI_no_TD.fillna(0, inplace=True)

# Add ecoinvent proxy emission factors for trade-only countries
for country in trade_only:
    CFPI_no_TD.loc[country] = trade_ef.loc[country].values

CFPI_TD_losses = CFPI_no_TD.multiply(TD_losses, axis=0).dropna(how='any',axis=0)  # apply transmission and distribution losses to production mix intensity


#%%

# Carbon intensity of consumption mix
CFCI_no_TD = pd.DataFrame(np.matmul(CFPI_no_TD.T, Lgen), columns=CFPI_no_TD.index).T
CFCI_no_TD.columns = ['Consumption mix intensity']
CFCI_TD_losses = CFCI_no_TD.multiply(TD_losses, axis=0).dropna(how='any', axis=0)


#%%

# Transpose added after removing country aggregation as data pre-treatment
if include_TD_losses:
    CFPI = CFPI_TD_losses
    CFCI = CFCI_TD_losses
else:
    CFPI = CFPI_no_TD
    CFCI = CFCI_no_TD

#%%

# ## Aggregate multi-nodes to single countries using weighted average of production/consumption as appropriate

country_total_prod_disagg.columns = ["Total production (TWh)"]
country_total_prod_disagg.index = original_countries
country_total_cons_disagg.columns = ["Total consumption (TWh)"]
country_total_cons_disagg.index = original_countries

#%%

#Calculate total carbon footprint intensity ratio production vs consumption
rCP = np.divide(CFCI, CFPI)
rCP.columns = ["ratio consumption:production mix"]

#%%

# # BEV calculations

# ### BEV production emissions calculations

# setup calculations
segments=["A","C","D","F"] #1 = A segment, 2= C segment, 3 = D segment, 4 = F segment
lifetime = 180000 # vehicle lifetime in km
production_el_intensity = 684 # Korean el-mix 684 g CO2/kWh, from ecoinvent

# read in data
vehicle_fp = os.path.join(fp_data, 'car_specifications.xlsx')
cars = pd.read_excel(vehicle_fp, index_col=[0,1,2], usecols='A:G', skipfooter=28)

#%%

vehicle_CO2 = ["BEV","ICEV"]

# Impacts fcorom electricity demand in cell production
battery_prod_el = production_el_intensity/1e6*cars.loc["BEV","Production el, battery"] #in t CO2/vehicle

# Total vehicle production impacts
BEV_prod_impacts = cars.loc["BEV","Production, ROV"] + cars.loc["BEV", "Production, RObattery"].add(battery_prod_el, fill_value=0).sum(axis=0)

# Modify for battery production in Europe
batt_prod_EU = pd.DataFrame(np.matmul(CFCI.values/1e6, cars.loc["BEV","Production el, battery"].values), index=CFCI.index, columns=cars.columns)

# Total battery production impacts in Europe
chck3 = (cars.loc["BEV","Production, ROV"] + cars.loc["BEV","Production, RObattery"])

BEV_prod_EU = pd.DataFrame(index=countries, columns=["A","C","D","F"])

for segment in batt_prod_EU.columns:
    BEV_prod_EU[segment] = chck3[segment].values + batt_prod_EU[segment]
BEV_prod_EU.columns = pd.MultiIndex.from_product([["EUR production impacts BEV"], BEV_prod_EU.columns, ["Consumption mix"]], names=["","Segment","Elmix"])


#%%

# ### BEV use phase calculations
elmixes = (CFPI.copy()).join(CFCI.copy()).T

#%%

# Remove trade-only countries
remove_country_list = []

for country, row in production.sum(axis=1).items():
    if row ==0:
        try:
            elmixes.drop(columns = country, inplace=True)
        except:
            print(f'{country} not in elmixes!')

#%%

# Calculate use phase emissions
segs = cars.loc['BEV', 'Use phase','Wh/km']
mi = pd.MultiIndex.from_product([list(elmixes.index), list(segs.index)])
segs = segs.reindex(mi, level=1)
segs = pd.DataFrame(segs)
segs.columns = ['a']
segs = segs.reindex(elmixes.columns, axis=1, method='bfill')

elmixes_for_calc = elmixes.reindex(mi, level=0, axis=0)

BEV_use = (segs.multiply(elmixes_for_calc/1000)).T

#%%

# ### Add production and EOL BEV

BEV_other = BEV_prod_impacts + cars.loc["BEV","EOL","t CO2"].values
BEV_other_intensity = BEV_other/lifetime*1e6 # in g CO2-eq
BEV_other_intensity.index = ["g CO2/km"]


#%%

# BEV impacts with production and consumption mixes
BEV_impactsp = (BEV_use['Production mix intensity'].T*lifetime/1e6).add(BEV_other.loc['t CO2'],axis=0)
BEV_impactsc = (BEV_use['Consumption mix intensity'].T*lifetime/1e6).add(BEV_other.loc['t CO2'],axis=0)

BEV_impacts = pd.concat([BEV_impactsp.T,BEV_impactsc.T], axis=1, keys=['Production mix','Consumption mix'])
BEV_impacts = BEV_impacts.swaplevel(axis=1, i=0, j=1)
BEV_impacts.sort_index(level=0, axis=1, inplace=True)

BEV_prod_sharesp = BEV_prod_impacts.values/(BEV_impactsp.T)
BEV_prod_sharesc = BEV_prod_impacts.values/(BEV_impactsc.T)
BEV_use_sharesp = (BEV_use['Production mix intensity']*lifetime/1e6)/(BEV_impactsp.T)
BEV_use_sharesc = (BEV_use['Consumption mix intensity']*lifetime/1e6)/(BEV_impactsc.T)

#%%

BEVp = pd.DataFrame(BEV_use["Production mix intensity"] + BEV_other_intensity.loc["g CO2/km"])
BEVc = pd.DataFrame(BEV_use["Consumption mix intensity"] + BEV_other_intensity.loc["g CO2/km"])

#%%

# ### Calculate BEV footprints with EUR production

BEV_fp_EU = (BEV_prod_EU.add(cars.loc["BEV","EOL","t CO2"], level=1, axis=1) / lifetime*1e6)

## Currently, EU production assumes consumption mix, so only examine using consumption mix for both manufacturing and use phase for consistency
EU_BEVc = BEV_fp_EU.add(BEV_use['Consumption mix intensity'].reindex(BEV_fp_EU.columns, axis=1, level=1), axis=1)
EU_BEVc.rename(columns={"EUR production impacts BEV":"BEV footprint, EUR production"}, inplace=True)

#%%

# ### Calculate EU production:Asian production footprint ratios

fp_ratio = EU_BEVc.divide(BEVc, level=1)

#%%

# ### Calculate total lifecycle emissions for ICEVs


ICEV_total_impacts = pd.DataFrame(cars.loc["ICEV"].sum(axis=0)).transpose()
ICEV_lc_footprint = ICEV_total_impacts.loc[0]*1e6/lifetime
#ICEV_total_impacts
ICEV_lc_shares = cars.loc['ICEV']/cars.loc['ICEV'].sum(axis=0)


#%%

# ### Calculate BEV:ICEV ratios

ratio_use_prod = BEV_use["Production mix intensity"] / cars.loc["ICEV","Use phase","t CO2"]

#%%


# Ratios comparing use phase only
ratio_use_prod = pd.DataFrame(BEV_use["Production mix intensity"]/ cars.loc["ICEV","Use phase","t CO2"])
ratio_use_cons = pd.DataFrame(BEV_use["Consumption mix intensity"] / cars.loc["ICEV","Use phase","t CO2"])

ratio_use_cons = ratio_use_cons.rename_axis("Segment", axis=1)
ratio_use_cons = pd.concat([ratio_use_cons], keys=["RATIO: use phase"], axis=1)
ratio_use_cons = pd.concat([ratio_use_cons], keys=["Consumption mix"], names=["Elmix"], axis=1)
ratio_use_cons = ratio_use_cons.reorder_levels([1,2,0], axis=1)


#%%

# Ratios with lifecycle impacts
ratiop = pd.DataFrame(BEVp/(ICEV_lc_footprint))
ratioc = pd.DataFrame(BEVc/(ICEV_lc_footprint))
#ratiop


#%%

ratioc_EU_prod = (EU_BEVc["BEV footprint, EUR production"].stack()/(ICEV_lc_footprint)).unstack()
ratioc_EU_prod = pd.concat([ratioc_EU_prod], keys=["Ratio BEV:ICEV, European BEV production"], axis=1)

# [Go home](#home)


#%%

# Assemble total results table

# CFEL  - the CO2 footprint of electricity in different European countries based on either i) production or ii) consumption perspective.
# BEV – the gCO2/km for electric vehicles for i) all EU countries ii) all segments iii) production and consumption el mix.
# RATIO  - the ratio of the gCO2/km for BEVs vs ICEs for i) all EU countries ii) all segments iii) production and consumption el mix.

fp = pd.concat({"Production mix":BEVp}, axis=1, names=["Elmix","Segment"])
fp = pd.concat([fp, pd.concat({"Consumption mix":BEVc}, axis=1, names=["Elmix","Segment"])], axis=1)
fp = fp.swaplevel(axis=1).sort_index(axis=1, level="Segment", sort_remaining=False)
fp = pd.concat({"BEV footprint":fp}, axis=1)

RATIOS = pd.concat({"Production mix": ratiop}, axis=1, names=["Elmix","Segment"])
RATIOS = pd.concat([RATIOS, pd.concat({"Consumption mix":ratioc}, axis=1, names=["Elmix","Segment"])], axis=1)
RATIOS = RATIOS.swaplevel(axis=1).sort_index(axis=1, level="Segment", sort_remaining=False)
RATIOS = pd.concat({"RATIO BEV:ICEV":RATIOS}, axis=1)

results = fp.join(RATIOS)
results = results.join(ratio_use_cons)
results = results.join(BEV_prod_EU)
results = results.join(EU_BEVc)
results = results.join(ratioc_EU_prod)

results.droplevel(0, axis=1)

results_toSI = results.copy()

# [Go home](#home)


#%%

# # Start mapping

segments = [
    'A - Small',
    'C - Medium',
    'D - Large',
    'F - Luxury']

el_mix = ['Production', 'Consumption']

#%%

BEV = results
CFEL_mixes = elmixes.T

df = pd.concat([country_total_prod, country_total_cons], axis=1)
df.columns = ['Total production (TWh)', 'Total consumption (TWh)']

CFEL = pd.concat([df, CFEL_mixes], axis=1)

#%%
""" Format dataframes for export to SI tables """

country_intensities = C.round(0).fillna(value='-').T

country_intensities.drop(index=trade_only, inplace=True)
production.drop(index=trade_only, inplace=True) # drop countries with no production data
trades_forSI = trades.round(2).replace(0, np.nan).fillna(value='-')
production = production.round(2).replace(0, np.nan).fillna(value='-')

CFEL_toSI = CFPI.join(CFCI)
for country in trade_only:
    CFEL_toSI.loc[country, 'Consumption mix intensity'] = np.nan


#%%

# ## Export results for SI
keeper = " run {:%d-%m-%y, %H_%M}".format(datetime.now())
results_file = os.path.join(fp_results, 'SI_results'+keeper+'.xlsx')
writer = pd.ExcelWriter(results_file)

table5 = results_toSI.loc(axis=1)['BEV footprint',:,'Consumption mix'].round(0)
table6 = results_toSI.loc(axis=1)['RATIO BEV:ICEV',:,'Consumption mix'].round(2)
table7 = results_toSI.loc(axis=1)['EUR production impacts BEV',:,'Consumption mix'].round(1)
table8 = results_toSI.loc(axis=1)['BEV footprint, EUR production',:,'Consumption mix'].round(0)
table9 = BEV_impacts.loc(axis=1)[:,'Consumption mix'].round(1)

# Write to Excel
CFEL_toSI.round(0).to_excel(writer, 'Table 1')
country_intensities.sort_index().to_excel(writer, 'Table 2 - intensity matrix')
production.to_excel(writer, 'Table 3 - Production mix')
trades_forSI.to_excel(writer, 'Table 4 - trades')
table5.droplevel(level=[2], axis=1).to_excel(writer, 'Table 5 - BEV fp')
table6.droplevel(level=[2], axis=1).to_excel(writer, 'Table 6 - ratio')
table7.droplevel(level=[2], axis=1).to_excel(writer, 'Table 7 - EUR prod imp')
table8.droplevel(level=[2], axis=1).to_excel(writer, 'Table 8 - EUR prod fp')
table9.droplevel(level=[1], axis=1).to_excel(writer, 'Table 9 - LC impacts')

writer.save()

#%%

# Export for figures

keeper = fp_output + 'country-specific indirect'
results.to_pickle(keeper + '_BEV.pkl')
CFEL.to_pickle(keeper + '_el.pkl')
BEV_impactsc.to_pickle(keeper + '_BEV_impacts_consumption.pkl')
ICEV_total_impacts.to_pickle(keeper + '_ICEV_impacts.pkl')

#%%
# Extra calculations

BEV_total_use = BEV_use * lifetime / 1e6  #absolute lifetime operation emissions


#%%

# Export intermediate variables from calculations for troubleshooting
if export_data == True:
    keeper = " run {:%d-%m-%y, %H_%M}".format(datetime.now())
    codecheck_file = os.path.join(fp_results,'code_check_'+keeper+'.xlsx')
    writer = pd.ExcelWriter(codecheck_file)

    g.to_excel(writer, "g")
    q.to_excel(writer, "q")
    y.to_excel(writer, 'y')
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
    CFEL.to_excel(writer,'CFEL - electricity')
    results.to_excel(writer,'car footprints')
    writer.save()