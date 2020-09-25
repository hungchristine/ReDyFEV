# -*- coding: utf-8 -*-
"""
This script runs the data extraction, data cleaning, calculations and visualization
scripts used in "Regionalized footprints of battery electric vehicles in Europe"

"""

if __name__ == "__main__":
    #import bentso_extract
    # import entso_data_clean
    # import hybridized_impact_factors
    import BEV_footprints_calculations as calc
    import BEV_visualization as viz

    experiments = {'baseline': [180000, 200000, True],
                   # 'no_LEONTIEF': [180000, 20000, False]
                   'long_BEV_life': [200000, 200000, True],
                   'short_BEV_life': [160000, 200000, True]}

    for exp, params in experiments.items():
        print('Performing experiment ' + exp)
        calc.run_stuff(run_id=exp,
                       BEV_lifetime=params[0],
                       ICEV_lifetime=params[1],
                       leontief_el=True,  # params[2]
                       production_el_intensity=684,
                       export_data=True,
                       include_TD_losses=True
                       )
        viz.visualize(exp, export_figures=True, include_TD_losses=True)

