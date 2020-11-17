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

logname = 'run ' + datetime.now().strftime('%d-%m-%y, %H-%M') + '.log'
logging.basicConfig(filename=logname,
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


if __name__ == "__main__":
    import bentso_extract
    import entso_data_clean
    import hybridized_impact_factors as hybrid
    import BEV_footprints_calculations as calc
    import BEV_visualization as viz

    # define electricity mix parameters and BEV/ICEV parame5ters
    el_experiments = {'baseline': [2019]}
    BEV_experiments = {
                       'baseline': [180000, 180000, True],
                       'Moro_Lonza': [180000, 180000, False],
                       'long_BEV_life': [250000, 180000, True],
                       'short_BEV_life': [150000, 180000, True],
                       'long_BEV_life_ML': [250000, 180000, False],
                       'short_BEV_life_ML': [150000, 180000, False]
                      }

    all_exp_results = {}

    # TODO: combine bentso_extract and entso_data_clean
    for el_exp, params in el_experiments.items():
        logging.info(f'Starting el_experiment {el_exp}')
        bentso_extract.bentso_query(params[0])  # can specify year of analysis here
        if len(params) > 1:
            logging.info(f'year: {params[0]}')
            logging.info(f'subperiod start: {params[1]}')
            logging.info(f'subperiod end: {params[2]}')
            ei_countries = entso_data_clean.clean_entso(params[1], params[2]) # can specify start and end of sub-annual analysis period here as datetime object
            print(ei_countries)
        else:
            logging.info(f'year: {params[0]}')
            ei_countries = entso_data_clean.clean_entso()
            print(ei_countries)
        logging.info('Starting calculation of hybridized emission factors')
        hybrid.hybrid_emission_factors(ei_countries)
        hybrid.clean_impact_factors(ei_countries)

        for BEV_exp, params in BEV_experiments.items():
            print('Performing experiment ' + BEV_exp)
            logging.info(f'Starting BEV experiment {BEV_exp}')
            all_exp_results[BEV_exp], ICEV_prodEOL_impacts, ICEV_op_int, SI_fp_temp = calc.run_stuff(run_id=BEV_exp,
                                                      BEV_lifetime=params[0],
                                                      ICEV_lifetime=params[1],
                                                      leontief_el=params[2],
                                                      production_el_intensity=679,
                                                      export_data=True,
                                                      include_TD_losses=True,
                                                      incl_ei=False
                                                      )
            if BEV_exp == 'baseline':
                SI_fp = SI_fp_temp
            logging.info('Visualizing results...')
            viz.visualize(BEV_exp, export_figures=True, include_TD_losses=True)
            logging.info(f'Completed experiment electricity {el_exp}, BEV scenario {BEV_exp}')

#        viz.visualize('baseline', export_figures=True, include_TD_losses=True, plot_ei=False)

        # Compile BEV intensities from all experiments for sensitivity analysis
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
        viz.plot_fig4(results, export_figures=True)
        results = results.reindex(columns=['baseline', 'long_BEV_life', 'short_BEV_life',
                         'Moro_Lonza', 'long_BEV_life_ML', 'short_BEV_life_ML',
                         'ICEV 250k','ICEV 200k','ICEV 180k'], level=0)
        results_flow = results.iloc[:, np.r_[0:12, -12:0]].copy()
        results_ML = results.iloc[:, 12:].copy()
        # rename columns for human-readability
        new_ind = pd.MultiIndex.from_product([['Baseline (180k)', 'Long BEV life (250k)', 'Short BEV life (150k)', 'ICEV 250k','ICEV 200k','ICEV 180k'], ['A','C','D','F']])

        results_flow.columns = new_ind
        results_ML.columns = new_ind

        with pd.ExcelWriter(SI_fp, engine="openpyxl") as writer:
            writer.book = openpyxl.load_workbook(SI_fp)
            results_flow.to_excel(writer, 'Table S11 - Sensitivity I')
            results_ML.to_excel(writer, 'Table S12 - Sensitivity II')
            writer.book.save(SI_fp)