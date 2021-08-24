COUNTRIES = ['AT','BE','BA','BG','HR','CY','CZ','DK','EE','FI','FR',
             'DE','GR','HU','IE','IS','IT','LV','LU','MK','ME','NL',
             'NI','NO','PL','PT','RO','RS','SK','SI','SE','CH']

RENEWABLES = {
    'Solar',
    'Wind Onshore',
    'Wind Offshore',
    'Hydro Water Reservoir',
    'Hydro Run-of-river and poundage',
    'Marine',
    'Geothermal',
    'Biomass',
}

# Not all countries actually present, even if they should be
ENTSO_COUNTRIES = set(COUNTRIES).difference({"HR", "CY", "IS", "LU", "ME", "NI"})

TRADE_PAIRS = {
    'AT': {'CH', 'CZ', 'DE', 'HU', 'IT', 'SI'},
    'BA': {'HR', 'ME', 'RS'},
    'BE': {'FR', 'LU', 'NL'},
    'BG': {'GR', 'MK', 'RO', 'RS'},
    'CH': {'AT', 'DE', 'FR', 'IT'},
    'CZ': {'AT', 'DE', 'PL', 'SK'},
    'DE': {'AT', 'CH', 'CZ', 'DK', 'FR', 'LU', 'NL', 'PL', 'SE'},
    'DK': {'DE', 'NO', 'SE'},
    'EE': {'FI', 'LV'},
    'FI': {'EE', 'NO', 'SE'},
    'FR': {'BE', 'CH', 'DE', 'IT'},
    'GR': {'BG', 'IT', 'MK'},
    'HR': {'BA', 'HU', 'RS', 'SI'},
    'HU': {'AT', 'HR', 'RO', 'RS', 'SK'},
    'IT': {'AT', 'CH', 'FR', 'GR', 'SI'},
    'LU': {'BE', 'DE'},
    'LV': {'EE'},
    'ME': {'BA', 'RS'},
    'MK': {'BG', 'GR', 'RS'},
    'NL': {'BE', 'DE', 'NO'},
    'NO': {'DK', 'FI', 'NL', 'SE'},
    'PL': {'CZ', 'DE', 'SE', 'SK'},
    'RO': {'BG', 'HU', 'RS'},
    'RS': {'BA', 'BG', 'HR', 'HU', 'ME', 'MK', 'RO'},
    'SE': {'DE', 'DK', 'FI', 'NO', 'PL'},
    'SI': {'AT', 'HR', 'IT'},
    'SK': {'CZ', 'HU', 'PL'},
}
