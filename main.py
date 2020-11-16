# -*- coding: utf-8 -*-
"""
This script runs the data extraction, data cleaning, calculations and visualization
scripts used in "Regionalized footprints of battery electric vehicles in Europe"

Users can run two types of experiments concurrently; the electricity sampling period
('el_experiments'), and vehicle parameters ('BEV_experiments')
"""
import logging
from datetime import datetime

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

    el_experiments = {'baseline': [2019]}
    # TODO: combine bentso_extract and entso_data_clean
    for el_exp, params in el_experiments.items():
        logging.info(f'Starting el_experiment {el_exp}')
        bentso_extract.bentso_query(params[0])  # can specify year of analysis here
        if len(params) > 1:
            logging.info(f'year: {params[0]}')
            logging.info(f'subperiod start: {params[1]}')
            logging.info(f'subperiod end: {params[2]}')
            entso_data_clean.clean_entso(params[1], params[2]) # can specify start and end of sub-annual analysis period here as datetime object
        else:
            logging.info(f'year: {params[0]}')
            entso_data_clean.clean_entso()
        logging.info('Starting calculation of hybridized emission factors')
        hybrid.hybrid_emission_factors()
        hybrid.clean_impact_factors()


        BEV_experiments = {
                       'baseline': [180000, 180000, False]#,
                       'flow_tracing': [180000, 200000, True],
                       'long_BEV_life': [200000, 180000, False],
                       'short_BEV_life': [160000, 180000, False]
                      }

        for BEV_exp, params in BEV_experiments.items():
            print('Performing experiment ' + BEV_exp)
            logging.info(f'Starting BEV experiment {BEV_exp}')
            calc.run_stuff(run_id=BEV_exp,
                           BEV_lifetime=params[0],
                           ICEV_lifetime=params[1],
                           leontief_el=params[2],
                           production_el_intensity=684,
                           export_data=True,
                           include_TD_losses=True
                           )
            logging.info('Visualizing results...')
            viz.visualize(BEV_exp, export_figures=True, include_TD_losses=True)
            logging.info(f'Completed experiment electricity {el_exp}, BEV scenario {BEV_exp}')
