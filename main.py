# -*- coding: utf-8 -*-
"""
This script runs the data extraction, data cleaning, calculations and visualization
scripts used in "Regionalized footprints of battery electric vehicles in Europe"

Users can run two types of experiments concurrently; the electricity sampling period
('el_experiments'), and vehicle parameters ('BEV_experiments')
"""
import logging
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment
import numpy as np

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


if __name__ == "__main__":

    # define electricity mix parameters and BEV/ICEV
    # accepted forms of el experiment:
    # -- single-element list with the full year of analysis
    # -- 2-item dict with start and stop periods in datetime objects
    # -- 3-item dict with time of assessment (datetime object), country and vehicle segment of desired footprint

    # provide start and end period for subannual analysis, or just start for a country footprint
    start = datetime(2021, 3, 14, 22, 44, tzinfo=pytz.timezone('Europe/Berlin'))  # year, month, day, hour, minute, timezone
    end = datetime(2021, 3, 14, 23, 00, tzinfo=pytz.timezone('Europe/Berlin'))

    if start.year != end.year:
        logging.error('Cannot do cross-year comparisons!')

    # definition of minimum data for each experiment type
    exp_type_dict = {'fullyear':set(['year']),
                     'subyear': set(['year', 'start', 'end']),
                     'country_fp': set(['year', 'start', 'country', 'segment'])
                     }

    el_experiments = {'2020': {'year': 2020},
                      # '2021': {'year': start.year, 'start': start, 'end': end}  # example subannual period analysis
                      # '2021fp': {'year': start.year, 'start':start, 'country':'FR', 'segment':'C'}
                      }

    BEV_experiments = {'baseline': {'BEV_life':180000, 'ICE_life':180000, 'leontief':True, 'low energy scen':False},
                        'long_BEV_life': {'BEV_life':250000, 'ICE_life':180000, 'leontief':True, 'low energy scen':False},
                        'short_BEV_life': {'BEV_life':150000, 'ICE_life':180000, 'leontief':True, 'low energy scen':False},
                        'grid_avg': {'BEV_life':180000, 'ICE_life':180000, 'leontief':False, 'low energy scen':False},
                        'long_BEV_life_ga': {'BEV_life':250000, 'ICE_life':180000, 'leontief':False, 'low energy scen':False},
                        'short_BEV_life_ga': {'BEV_life':150000, 'ICE_life':180000, 'leontief':False, 'low energy scen':False},
                        'baseline_energy': {'BEV_life':200000, 'ICE_life':180000, 'leontief':True, 'low energy scen':True},
                        'long_BEV_life_energy': {'BEV_life':250000, 'ICE_life':180000, 'leontief':True, 'low energy scen':True},
                        'short_BEV_life_energy': {'BEV_life':150000, 'ICE_life':180000, 'leontief':True, 'low energy scen':True},
                       }

    # this dict defines the groups of BEV_experiments to plot together in sensitivity analysis.
    # the list items must therefore match BEV_experiment keys
    sensitivity_groups = {'baseline':['baseline', 'long_BEV_life', 'short_BEV_life'],
                          'grid_average': ['grid_avg', 'long_BEV_life_ga', 'short_BEV_life_ga'],
                          'batt_energy': ['baseline_energy', 'long_BEV_life_energy', 'short_BEV_life_energy']}

    new_query = 0  # trigger for re-querying ENTSO-E dat abase (takes time!)
    all_exp_results = {}  # holds all results for sensitivity analysis
    segments = ['A', 'C', 'D', 'F', 'mini', 'medium', 'large', 'luxury']  # valid entries for segment
    segment_dict = {'mini': 'A', 'medium':'C', 'large':'D', 'luxury':'F'}

    # Begin running electricity experiments
    # Default is to recalculate everything
    for el_exp, params in el_experiments.items():
        logging.info(f'Starting el_experiment {el_exp}')

        if ('year' not in params.keys()) and ('start' in params.keys()):
            # manually populate the 'year' entry manually if missing
            params['year'] = params[start].year

        if set(params.keys()) == exp_type_dict['fullyear']:
            experiment_type = 'fullyear'
        elif set(params.keys()) == exp_type_dict['subyear']:
            experiment_type = 'subyear'
        elif set(params.keys()) == exp_type_dict['country_fp']:
            if params['segment'] in segments:
                experiment_type = 'country_fp'
                if params['segment'] in segment_dict.keys():
                    params['segment'] = segment_dict[params['segment']]  # convert to letter segment classification
            else:
                logging.error(f'Incorrect specification of desired vehicle segment')
        else:
            logging.error('Invalid electricity experiment input')

        # case for sub-annual analysis period (or single-country footprint)
        if (experiment_type == 'subyear') or (experiment_type == 'country_fp'):
            if new_query:
                bentso_extract.bentso_query(params['year'])  # can alternatively specify year of analysis here
            logging.info(f'year: {params["year"]}')
            if experiment_type == 'subyear':
                # for sub-annual analysis period
                logging.info(f'subperiod start: {params["start"]}')
                logging.info(f'subperiod end: {params["end"]}')
                ei_countries = entso_data_clean.clean_entso(year=params['year'],
                                                            start=params['start'],
                                                            end=params['end'])
            else:
                # For country-footprint experiment (snapshot of time and segment)
                # identical to subannual analysis; only visualization changes
                if new_query:
                    # NB: we can use full_year = True here, but query will be rounded to closest hour
                    # (and which is averaged out if sub-hourly samples are available)
                    bentso_extract.bentso_query(params['year'], full_year=False)
                logging.info(f'country footprint: {params["country"]}')
                logging.info(f'segment footprint: {params["segment"]}')
                ei_countries, timestamp = entso_data_clean.clean_entso(year=params['year'],
                                                                        start=params['start'],
                                                                        end=params['start'] + timedelta(hours=1),
                                                                        country=params['country'])
        # case for full-year analysis
        elif experiment_type == 'fullyear':
            if new_query:
                bentso_extract.bentso_query(params['year'])  # can alternatively specify year of analysis here
            logging.info(f'year: {params["year"]}')
            ei_countries = entso_data_clean.clean_entso(year=params['year'])

        # Calculate hybridized impact factors for electricity
        print(f'Using ecoinvent factors for {ei_countries} \n')
        logging.info('Starting calculation of hybridized emission factors \n')
        hybrid.hybrid_emission_factors(ei_countries, params['year'])
        hybrid.clean_impact_factors(params['year'], ei_countries)


        # Begin BEV experiments - calculation and visualization
        for BEV_exp, BEV_params in BEV_experiments.items():
            print('Performing experiment ' + BEV_exp)
            logging.info(f'Starting BEV experiment {BEV_exp}')
            all_exp_results[BEV_exp], ICEV_prodEOL_impacts, ICEV_op_int, SI_fp_temp = calc.run_calcs(run_id=BEV_exp+'_'+el_exp,
                                                      year = params['year'],
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
                SI_fp = SI_fp_temp  # only need to update the main SI file (that will is submitted with article)
            logging.info('Visualizing results...')
            if (experiment_type == 'fullyear') or (experiment_type == 'subyear'):
                viz.visualize(BEV_exp+'_'+el_exp, export_figures=True, include_TD_losses=True)
            elif experiment_type == 'country_fp':
                viz.country_footprint(BEV_exp+'_'+el_exp, params, timestamp, export_figures=True)

            logging.info(f'Completed experiment electricity {el_exp}, BEV scenario {BEV_exp}')

        # Check we have all the necessary experiments to run a sensitivity analysis
        run_sensitivity = False
        for sens, exp in sensitivity_groups.items():
            if len(set(exp) - set(exp).intersection(set(BEV_experiments.keys()))) == 0:
                run_sensitivity = True

        # Compile BEV intensities from all experiments for sensitivity analysis
        if run_sensitivity and (experiment_type != 'country_fp'):
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

            # concatenate all results and visualize sensitivity analysis
            results = pd.concat([results, tmp_ICEV_fp], axis=1)
            results = results.loc[~results.index.isin(ei_countries)]
            viz.plot_fig6(results, export_figures=True)

            results = results.reindex(columns=list(BEV_experiments.keys()) + ['ICEV 205k', 'ICEV 200k', 'ICEV 180k'], level=0)

            #-- Begin export to SI
            # separate each 'set' of BEV_experiments for exporting to SI
            results_lifetime_sensitivity = results.iloc[:, np.r_[0:12, -12:0]].copy()
            if results.shape[1] > 24:
                results_ga = results.iloc[:, np.r_[12:24]].copy()

            # rename columns for human-readability
            new_ind = pd.MultiIndex.from_product([['Baseline (180k)', 'Long BEV life (250k)', 'Short BEV life (150k)', 'ICEV 250k','ICEV 200k','ICEV 180k'], ['A','C','D','F']])

            results_lifetime_sensitivity.columns = new_ind
            results_ga.columns = new_ind[0:12]

            # make custom dataframe for low-manufacturing-energy sensitivity
            df = (all_exp_results['baseline_energy'] - all_exp_results['baseline']) / all_exp_results['baseline']

            sens_batt_energy = pd.concat([all_exp_results['baseline_energy'].round(0), df], axis=1)
            sens_batt_energy.columns = pd.MultiIndex.from_product([['Footprint g CO2e/km', '% change from baseline'], df.columns])

            # dict of captions for each sheet
            excel_dict2 = {'Table S12': 'Sensitivity with lower battery production electricity. Footprint with lower electricity demand and % change from baseline',
                           'Table S13': 'Data for Figure 7. Sensitivity with differing vehicle lifetimes for BEVs and ICEVs. Lifecycle carbon intensity, in g CO2e/vkm.',
                           'Table S14': 'BEV footprints using grid-average approach from Moro and Lonza (2018) to calculate electricity mix footprint, in g CO2e/vkm'}

            with pd.ExcelWriter(SI_fp, engine="openpyxl") as writer:
                writer.book = openpyxl.load_workbook(SI_fp)
                sens_batt_energy.to_excel(writer, 'Table S12', startrow=2)
                results_lifetime_sensitivity.round(0).to_excel(writer, 'Table S13', startrow=2)
                results_ga.round(0).to_excel(writer, 'Table S14', startrow=2)

                for sheet, caption in excel_dict2.items():
                    worksheet = writer.sheets[sheet]
                    worksheet['A1'] = sheet
                    worksheet['A1'].font = Font(color='0070C0', bold=True)
#
                    worksheet.merge_cells('C1:L1')
                    worksheet['C1'] = caption
                    worksheet['C1'].alignment = Alignment(wrap_text=True)

                    worksheet.row_dimensions[1].height = 60

                    if sheet == 'Table S12':
                        for col in ['F','G','H','I']:
                            for row in range(worksheet.min_row, worksheet.max_row + 1):
                                worksheet[col + str(row)].number_format = '#0%'

                writer.book.save(SI_fp)