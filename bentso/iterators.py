from .client import CachingDataClient
from .constants import COUNTRIES
import itertools


def iterate_generation(year, location=None, verbose=False):
    """Iterate over all countries and technology types, and return annual
    generation sums as ``(technology, country, amount)``.

    All data returned are in ENTSO-E labels and ISO country codes."""
    c = CachingDataClient(key="cache-only", location=location, verbose=verbose)

    for country in COUNTRIES:
        gen = c.get_generation(country, year)
        if gen is not None:
            # Unit conversion from MWh to MJ
            gen = gen.sum() * 3600
            for technology, amount in zip(gen.index, gen):
                yield technology, country, amount


def iterate_trade(year, location=None, verbose=False):
    """Iterate over all country combinations, and return annual trade sums as
    ``(from country, to country, amount)``.

    All country labels returned are in ISO country codes."""
    c = CachingDataClient(key="cache-only", location=location, verbose=verbose)

    for from_country, to_country in itertools.combinations(COUNTRIES, 2):
        trd = c.get_trade(from_country, to_country, year)
        if trd is not None:
            # Unit conversion from MWh to MJ
            yield from_country, to_country, trd.sum() * 3600
