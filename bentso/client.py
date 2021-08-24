from .constants import RENEWABLES
from .db import get_database, File
from .filesystem import (
    DEFAULT_DATA_DIR,
    create_dir,
    sha256,
    load_token,
)
from entsoe import EntsoePandasClient
from peewee import DoesNotExist
from unittest.mock import Mock
import os
import pandas as pd


class CachingDataClient:
    def __init__(self, location=None, key=None, verbose=True):
        self._cache_only = key == "cache-only"
        self.verbose = verbose
        USER_PATH = os.environ.get('BENTSO_DATA_DIR')
        self.dir = location or USER_PATH or DEFAULT_DATA_DIR
        create_dir(self.dir)
        get_database(self.dir)
        self.data_dir = os.path.join(self.dir, "data")
        create_dir(self.data_dir)
        if self.verbose:
            print("Using data directory {}".format(os.path.abspath(self.dir)))
        if not self._cache_only:
            self.client = EntsoePandasClient(api_key=load_token())
        else:
            self.client = Mock()

    def get_trade(self, from_country, to_country, year, full_year=False):
        country_field = "{}-{}".format(from_country, to_country)
        result = self._cached_query(
            (from_country, to_country,),
            year,
            'trade',
            country_field,
            self.client.query_crossborder_flows,
        )
        if full_year:
            result = self._full_year(result, year)
        return result

    def get_consumption(self, country, year):
        return self._cached_query(
            (country,),
            year,
            'load',
            country,
            self.client.query_load,
        )

    def get_generation(self, country, year, clean=False, full_year=False):
        result = self._cached_query(
            (country,),
            year,
            'generation',
            country,
            self.client.query_generation,
        )
        if clean:
            result = self._clean_all(result)
        if full_year:
            result = self._full_year(result, year)
        return result

    def get_capacity(self, country, year):
        return self._cached_query(
            (country,),
            year,
            'capacity',
            country,
            self.client.query_installed_generation_capacity,
        )

    def get_hydro_charging(self, country, year):
        pass

    def get_day_ahead_prices(self, country, year):
        return self._cached_query(
            (country,),
            year,
            'price',
            country,
            self.client.query_day_ahead_prices,
        )

    def _cached_query(self, args, year, kind_label, country_label, method):
        year = int(year)
        start, end = self._get_start_end(year)
        try:
            obj = File.select().where(File.kind==kind_label,
                                      File.country==country_label,
                                      File.year==year).get()
            return self._load_df(obj)
        except DoesNotExist:
            if not self._cache_only:
                if self.verbose:
                    print("Querying ENTSO-E API. Please be patient...")
                start, end = self._get_start_end(year)
                df = method(*args, start=start, end=end)
                hash, name = self._store_df(
                    df,
                    "{}-{}-{}.pickle".format(kind_label, country_label, year)
                )
                File.create(
                    filename=name,
                    country=country_label,
                    year=year,
                    sha256=hash,
                    kind=kind_label,
                )
                return df
            else:
                if self.verbose:
                    print("Value not in cache; returning nothing.")

    def _full_year(self, df, year):
        def leapYear(year):
            import calendar
            return(calendar.isleap(year))

        if leapYear(year):
            num_hours = 8784
            num_halfhours = 17568
            num_quarterhours = 35136
        else:
            num_hours = 8760
            num_halfhours = 17520
            num_quarterhours = 35040

        if (df.shape[0] > num_halfhours) and (df.shape[0] < num_quarterhours):
            start, end = self._get_start_end(year)
            idx = pd.date_range(start, end, freq='15min')
            df = df.reindex(idx).fillna(df.mean())
            df = (df*0.25).resample('H').sum()
        elif (df.shape[0] > num_hours) and (df.shape[0] < num_halfhours):
            start, end = self._get_start_end(year)
            idx = pd.date_range(start, end, freq='30min')
            df = df.reindex(idx).fillna(df.mean())
            df = (df*0.5).resample('H').sum()
        elif df.shape[0] < num_hours:
            start, end = self._get_start_end(year)
            idx = pd.date_range(start, end, freq='H')
            df = df.reindex(idx).fillna(df.mean())
        return df

    def _get_start_end(self, year):
        return (
            pd.Timestamp(year=year, month=1, day=1, hour=0, tz='Europe/Brussels'),
            pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, tz='Europe/Brussels'),
        )

    def _store_df(self, df, name):
        filepath = os.path.join(self.data_dir, name)
        df.to_pickle(filepath)
        return sha256(filepath), name

    def _load_df(self, obj):
        filepath = os.path.join(self.data_dir, obj.filename)
        if sha256(filepath) != obj.sha256:
            raise OSError("Corrupted cache file: {}".format(obj.filename))
        return pd.read_pickle(filepath)

    def _clean_all(self, df):
        df = self._remove_zero_columns(df)
        df = self._remove_other_renewable(df)
        df = self._remove_other(df)
        return df

    def _remove_other_renewable(self, df, renewables=RENEWABLES):
        """Remove `Other renewables` column and rescale renewable columns"""
        renewables = set(renewables)

        if 'Other renewable' in df:
            renewable_total = sum([df[label].sum()
                                   for label in renewables
                                   if label in df])
            if not renewables.intersection(set(df.columns)):
                raise ValueError("No substitutable renewable sources found")

            oth_renew_total = df['Other renewable'].sum()
            scale = (oth_renew_total / renewable_total) + 1
            for label in renewables:
                if label in df:
                    df[label] *= scale
            return df.drop('Other renewable', axis=1)
        else:
            return df

    def _remove_other(self, df):
        """Remove `Other` column and rescale everything else"""
        if 'Other' in df:
            total = df.sum().sum()
            scale = total / (total - df['Other'].sum())
            return df.drop('Other', axis=1) * scale
        else:
            return df

    def _remove_zero_columns(self, df):
        """Drop columns with zero generation"""
        for col in df:
            if not df[col].sum():
                df = df.drop(col, axis=1)
        return df
