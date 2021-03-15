# -*- coding: utf-8 -*-
"""
This script runs the data extraction, data cleaning, calculations and visualization
scripts used in "Regionalized footprints of battery electric vehicles in Europe"

Users can run two types of experiments concurrently; the electricity sampling period
('el_experiments'), and vehicle parameters ('BEV_experiments')
"""
import logging
from datetime import datetime
import pandas as pd
import openpyxl
import numpy as np
from enum import Enum
import bentso_extract
import entso_data_clean
import hybridized_impact_factors as hybrid
import BEV_footprints_calculations as calc
import BEV_visualization as viz

logname = 'run ' + datetime.now().strftime('%d-%m-%y, %H-%M') + '.log'
logging.basicConfig(filename=logname,
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


def calc_emission_factors(ei_countries, year=None, use_entso=0):
    logging.info('Starting calculation of hybridized emission factors')
    # hybrid.hybrid_emission_factors(ei_countries, use_entso, year)
    hybrid.clean_impact_factors(year, ei_countries, use_entso)

def calc_BEV_footprints(use_entso, BEV_experiments, ei_countries, el_exp):
    for BEV_exp, BEV_params in BEV_experiments.items():
        print('Performing experiment ' + BEV_exp)
        logging.info(f'Starting BEV experiment {BEV_exp}')
        all_exp_results[BEV_exp], ICEV_prodEOL_impacts, ICEV_op_int, SI_fp_temp = calc.run_calcs(run_id=BEV_exp+'_'+el_exp,
                                                  use_entso=use_entso,
                                                  year=el_exp,
                                                  BEV_lifetime=BEV_params['BEV_life'],
                                                  ICEV_lifetime=BEV_params['ICE_life'],
                                                  leontief_el=BEV_params['leontief'],
                                                  production_el_intensity=679,
                                                  export_data=True,
                                                  include_TD_losses=True,
                                                  incl_ei=False,
                                                  energy_sens=BEV_params['low energy scen']
                                                  )
        if BEV_exp == 'baseline':
            SI_fp = SI_fp_temp
        logging.info('Visualizing results...')
        viz.visualize(BEV_exp+'_'+el_exp, use_entso, export_figures=True, include_TD_losses=True)
        logging.info(f'Completed experiment electricity {el_exp}, BEV scenario {BEV_exp}')

#        viz.visualize('baseline', export_figures=True, include_TD_losses=True, plot_ei=False)

    # Compile BEV intensities from all experiments for sensitivity analysis
    if len(BEV_experiments) > 1:
        mi = pd.MultiIndex.from_product([all_exp_results.keys(), all_exp_results['baseline'].columns.tolist()])
        results = pd.DataFrame(index=all_exp_results['baseline'].index, columns=mi)

        for exp in all_exp_results.keys():
            results[exp] = all_exp_results[exp]

        # Compile ICEV intensities for different lifetimes
        tmp_ICEV_fp = ICEV_prodEOL_impacts.to_frame().T.reindex(results.index)
        ind = pd.MultiIndex.from_product([['ICEV 250k', 'ICEV 200k', 'ICEV 180k'], tmp_ICEV_fp.columns.tolist()])
        tmp_ICEV_fp = tmp_ICEV_fp.reindex(ind, axis=1)

        ICEV_dict = {'ICEV 250k': 250000,
                     'ICEV 200k': 200000,
                     'ICEV 180k': 180000}

        for ICE, dist in ICEV_dict.items():
            for seg in tmp_ICEV_fp[ICE].columns:
                tmp_ICEV_fp.loc[:, (ICE, seg)] = ((ICEV_prodEOL_impacts / dist * 1e6).add(ICEV_op_int, axis=0)).loc[seg]

        results = pd.concat([results, tmp_ICEV_fp], axis=1)
        results = results.loc[~results.index.isin(ei_countries)]
        viz.plot_fig5(results, export_figures=True)
        results = results.reindex(columns=['baseline', 'long_BEV_life', 'short_BEV_life',
                         'Moro_Lonza', 'long_BEV_life_ML', 'short_BEV_life_ML',
                         'baseline_energy', 'long_BEV_life_energy', 'short_BEV_life_energy',
                         'ICEV 250k','ICEV 200k','ICEV 180k'], level=0)
        results_flow = results.iloc[:, np.r_[0:12, -12:0]].copy()
        results_ML = results.iloc[:, np.r_[12:24]].copy()
        # results_energy = results.iloc[:, 24:-12].copy()

        # rename columns for human-readability5
        new_ind = pd.MultiIndex.from_product([['Baseline (180k)', 'Long BEV life (250k)', 'Short BEV life (150k)', 'ICEV 250k','ICEV 200k','ICEV 180k'], ['A','C','D','F']])

        results_flow.columns = new_ind
        results_ML.columns = new_ind[0:12]
        # results_energy.columns = new_ind

        df = (all_exp_results['baseline_energy'] - all_exp_results['baseline']) / all_exp_results['baseline']

        sens_batt_energy = pd.concat([all_exp_results['baseline_energy'].round(0), df], axis=1)
        sens_batt_energy.columns = pd.MultiIndex.from_product([['Footprint g CO2e/km', '% change from baseline'], df.columns])

        # dict of captions for each sheet
        excel_dict2 = {'Table S12': 'Sensitivity with lower battery production electricity. Footprint with lower electricity demand and % change from baseline',
                       'Table S13': 'Data from Figure 5 in manuscript. Comparison of BEV and ICEV CO2 footprint using lower electricity demand for battery production, in g CO2e/vkm.',
                       'Table S14': 'BEV footprints using grid-average approach to calculate electricity mix footprint, in g CO2e/vkm'}

        with pd.ExcelWriter(SI_fp, engine="openpyxl") as writer:
            writer.book = openpyxl.load_workbook(SI_fp)
            sens_batt_energy.to_excel(writer, 'Table S12', startrow=2)
            results_flow.round(0).to_excel(writer, 'Table S13', startrow=2)
            results_ML.round (0).to_excel(writer, 'Table S14', startrow=2)

            for sheet, caption in excel_dict2.items():
                worksheet = writer.sheets[sheet]
                worksheet['A1'] = sheet
                worksheet['C1'] = caption
                if sheet == 'Table S12':
                    for col in ['F','G','H','I']:
                        for row in range(worksheet.min_row, worksheet.max_row + 1):
                            worksheet[col + str(row)].number_format = '#0%'
            writer.book.save(SI_fp)

if __name__ == "__main__":
    # define experiments: electricity mix parameters and BEV/ICEV parameters
    el_experiments = {#'2019': [2019],
                      '2020': [2020]}
    BEV_experiments = {'baseline': {'BEV_life':180000, 'ICE_life':180000, 'leontief':True, 'low energy scen':False},
                        # 'long_BEV_life': {'BEV_life':250000, 'ICE_life':180000, 'leontief':True, 'low energy scen':False},
                        # 'short_BEV_life': {'BEV_life':150000, 'ICE_life':180000, 'leontief':True, 'low energy scen':False},
                        # 'Moro_Lonza': {'BEV_life':180000, 'ICE_life':180000, 'leontief':False, 'low energy scen':False},
                        # 'long_BEV_life_ML': {'BEV_life':250000, 'ICE_life':180000, 'leontief':False, 'low energy scen':False},
                        # 'short_BEV_life_ML': {'BEV_life':150000, 'ICE_life':180000, 'leontief':False, 'low energy scen':False},
                        # 'baseline_energy': {'BEV_life':180000, 'ICE_life':180000, 'leontief':True, 'low energy scen':True},
                        # 'long_BEV_life_energy': {'BEV_life':250000, 'ICE_life':180000, 'leontief':True, 'low energy scen':True},
                        # 'short_BEV_life_energy': {'BEV_life':150000, 'ICE_life':180000, 'leontief':True, 'low energy scen':True},
                      }

    use_entso = 1  # trigger for using ENTSO-E; else use Eurostat
    new_query = 0  # trigger for re-querying ENTSO-E database (takes time!)

    all_exp_results = {}

    # TODO: combine bentso_extract and entso_data_clean
    if type(el_experiments) is str:
        ei_countries = 'a'
    else:
        if use_entso:
            logging.info(f'Using ENTSO-E Transparency Portal for electricity mixes')
            for el_exp, params in el_experiments.items():
                logging.info(f'Starting el_experiment {el_exp}')
                if new_query:
                    bentso_extract.bentso_query(params[0])  # can alternatively specify year of analysis here
                # case for sub-annual analysis period
                if len(params) > 1:
                    logging.info(f'year: {params[0]}')
                    logging.info(f'subperiod start: {params[1]}')
                    logging.info(f'subperiod end: {params[2]}')
                    # can specify start and end of sub-annual analysis period here as datetime object
                    ei_countries = entso_data_clean.clean_entso(params[0], params[1], params[2])
                    print(ei_countries)
                # case for full-year analysis
                else:
                    logging.info(f'year: {params[0]}')
                    ei_countries = entso_data_clean.clean_entso(year=params[0])
                    print(ei_countries)
                # calc_emission_factors(ei_countries, params[0], use_entso)
                calc_BEV_footprints(use_entso, BEV_experiments, ei_countries, el_exp)
        else:
            import eurostat_extract
            logging.info('Using local Eurostat files for electricity mixes')
            el_exp='eurostat'
            ei_countries = eurostat_extract.load_eurostat()
            print(ei_countries)
            # calc_emission_factors(ei_countries)
            calc_BEV_footprints(use_entso, BEV_experiments, ei_countries, el_exp)
