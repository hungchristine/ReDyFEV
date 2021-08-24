# ReDyFEV

A Python module for calculating the ***Re***gionalized ***Dy***namic lifecycle climate ***F***ootprints of battery ***E***lectric ***V***ehicles in Europe


## Description
This package contains the code required to run the data retrieval from the ENTSO-E Transparency Platform, perform electricity mix and 
BEV footprint calculations, and visualizations in the manuscript entitled "Regionalized climate footprints of battery electric vehicles".  

## Installation
[Clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository) this repository to your local machine to install ReDyFEV.

Note that querying the [ENTSO-E Transparency Portal](https://transparency.entsoe.eu/) requires an API key, which is added to the environment in `bentso_extract.py`

## Dependencies
* [bentso](https://github.com/BONSAMURAIS/bentso) (modified, see [here](https://github.com/BONSAMURAIS/bentso/pull/8))
* [pylcaio 1.0](https://github.com/MaximeAgez/pylcaio/releases/tag/v1.0)
* [country_converter](https://github.com/konstantinstadler/country_converter)
* geopandas
* pandas
* matplotlib
* seaborn
* cmcrameri (colourmap; optional)

## Usage
Run `main.py` to perform data retrieval and all calculations and visualization. 
Users can modify a number of parameters to tailor their analysis, opt out of querying the ENTSO-E Transparency Platform (if cached data is available), or skip calculating hybridized emission factors. 

Experiment parameters are defined in `main.py`. Experiment parameters include:
* time period for analysis (query ENTSO-E)
  * in the case of a footprint analysis (single-country, "intstantenous" footprint), country and vehicle segment
* electricity consumption mix calculation approach (flow-tracing or grid average)
* lifetime in km, BEV and ICEV
* allocation of battery production emissions to second-life applications
* sensitivity analysis of electricity use:  battery production

Each specified time period constitutes an "electriicty experiment"; each combination of the remaining parameters constitutes a "BEV experiment". `main.py` runs all combinations of electricity and BEV experiments by default.

Users can also modify vehicle specifications in `\data\car_specification.xlsx`:
* carbon intensity of life cycle phases, BEV and ICEV
* assumed cycle life of batteries
* electricity intensity of battery production
* use-phase energy efficiency

### Folder structure
The files containing data from external sources for the calculations and visualization are in the `\data` folder. 
  * `car_specifications.xlsx` contains the parameters for BEV and ICEV cars to be studied, and the values for energy use in battery production for sensitivity analyses, if desired.
  * `label_pos.csv` provides the positioning of annotation labels on the country figures
  * `API_EG.ELC.LOSS.ZS_DS2_en_csv_v2_673578.csv` (not included) provides electricity transmission losses from [OECD (subscription required)](https://data.worldbank.org/indicator/EG.ELC.LOSS.ZS) - users can provide their own geographic specific values
  * maps data from [Natural Earth](https://www.naturalearthdata.com/)
  
The output files containing results from intermediate calculations and that are required to run the modules individually (i.e., not via `main.py`) are provided in the `\output` folder. 
  * cached queries results to ENTSO-E Transparency Portal API (`\output\entsoe`)
  * pickle files of each intermediate step (calculating electricity mixes and their footprint, vehicle footprints)
  * logs of missing emission factors

The `\results` folder contains Excel files of intermediate output for troubleshooting, and the SI-formatted results, and figures from the visualization. 

## License
[Apache](https://www.apache.org/licenses/LICENSE-2.0)
