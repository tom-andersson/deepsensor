# %%
from typing import Union

import xarray as xr
import numpy as np
import pandas as pd
import unittest

from deepsensor.data.processor import DataProcessor
from tests.utils import gen_random_data_xr, gen_random_data_pandas


def _gen_data_xr(coords=None, dims=None, data_vars=None):
    """Gen random raw data"""
    if coords is None:
        coords = dict(
            time=pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            lat=np.linspace(20, 40, 30),
            lon=np.linspace(40, 60, 20),
        )
    da = gen_random_data_xr(coords, dims, data_vars)
    return da


def _gen_data_pandas(coords=None, dims=None, cols=None):
    """Gen random raw data"""
    if coords is None:
        coords = dict(
            time=pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            lat=np.linspace(20, 40, 10),
            lon=np.linspace(40, 60, 10),
        )
    df = gen_random_data_pandas(coords, dims, cols)
    return df


class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor

    Tests TODO:
    - Test different time frequencies
    - ...
    """

    def assert_allclose_pd(
        self, df1: Union[pd.DataFrame, pd.Series], df2: Union[pd.DataFrame, pd.Series]
    ):
        if isinstance(df1, pd.Series):
            df1 = df1.to_frame()
        if isinstance(df2, pd.Series):
            df2 = df2.to_frame()
        try:
            pd.testing.assert_frame_equal(df1, df2)
        except AssertionError:
            return False
        return True

    def assert_allclose_xr(
        self, da1: Union[xr.DataArray, xr.Dataset], da2: Union[xr.DataArray, xr.Dataset]
    ):
        try:
            xr.testing.assert_allclose(da1, da2)
        except AssertionError:
            return False
        return True

    def test_different_names_xr(self):
        """
        The time, x1 and x2 dimensions can have arbitrary names and these should be restored
        after unnormalisation.
        """
        da_raw = _gen_data_xr()
        da_raw = da_raw.rename(
            {"time": "datetime", "lat": "latitude", "lon": "longitude"}
        )

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="datetime",
            x1_name="latitude",
            x2_name="longitude",
        )
        da_norm = dp(da_raw)
        self.assertListEqual(
            ["time", "x1", "x2"], list(da_norm.dims), "Failed to rename dims."
        )

        da_unnorm = dp.unnormalise(da_norm)
        self.assertTrue(
            self.assert_allclose_xr(da_unnorm, da_raw),
            f"Original {type(da_raw).__name__} not restored.",
        )

    def test_same_names_xr(self):
        """
        Test edge case when dim names are already in standard form.
        """
        da_raw = _gen_data_xr()
        da_raw = da_raw.rename({"lat": "x1", "lon": "x2"})

        dp = DataProcessor(x1_map=(20, 40), x2_map=(40, 60))
        da_norm = dp(da_raw)
        self.assertListEqual(
            ["time", "x1", "x2"], list(da_norm.dims), "Failed to rename dims."
        )

        da_unnorm = dp.unnormalise(da_norm)
        self.assertTrue(
            self.assert_allclose_xr(da_unnorm, da_raw),
            f"Original {type(da_raw).__name__} not restored.",
        )

    def test_wrong_order_xr_ds(self):
        """Order of dimensions in xarray must be: time, x1, x2"""
        ds_raw = _gen_data_xr(dims=("time", "lat", "lon"), data_vars=["var1", "var2"])
        ds_raw = ds_raw.transpose("time", "lon", "lat")  # Transpose, changing order

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="lat",
            x2_name="lon",
        )
        with self.assertRaises(ValueError):
            dp(ds_raw)

    def test_wrong_order_xr_da(self):
        """Order of dimensions in xarray must be: time, x1, x2"""
        da_raw = _gen_data_xr()
        da_raw = da_raw.T  # Transpose, changing order

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="lat",
            x2_name="lon",
        )
        with self.assertRaises(ValueError):
            dp(da_raw)

    def test_different_names_pandas(self):
        """
        The time, x1 and x2 dimensions can have arbitrary names and these should be restored
        after unnormalisation.
        """
        df_raw = _gen_data_pandas()
        df_raw.index.names = ["datetime", "lat", "lon"]

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="datetime",
            x1_name="lat",
            x2_name="lon",
        )

        df_norm = dp(df_raw)

        self.assertListEqual(["time", "x1", "x2"], list(df_norm.index.names))

        df_unnorm = dp.unnormalise(df_norm)

        self.assertTrue(
            self.assert_allclose_pd(df_unnorm, df_raw),
            f"Original {type(df_raw).__name__} not restored.",
        )

    def test_same_names_pandas(self):
        """
        Test edge case when dim names are already in standard form.
        """
        df_raw = _gen_data_pandas()
        df_raw.index.names = ["time", "x1", "x2"]

        dp = DataProcessor(x1_map=(20, 40), x2_map=(40, 60))  # No name changes

        df_norm = dp(df_raw)

        self.assertListEqual(["time", "x1", "x2"], list(df_norm.index.names))

        df_unnorm = dp.unnormalise(df_norm)

        self.assertTrue(
            self.assert_allclose_pd(df_unnorm, df_raw),
            f"Original {type(df_raw).__name__} not restored.",
        )

    def test_wrong_order_pandas(self):
        """Order of dimensions in pandas index must be: time, x1, x2"""
        df_raw = _gen_data_pandas()
        df_raw = df_raw.swaplevel(0, 2)

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="lat",
            x2_name="lon",
        )

        with self.assertRaises(ValueError):
            dp(df_raw)

    def test_extra_indexes_preserved_pandas(self):
        """
        Other metadata indexes are allowed (only *after* the default dimension indexes of
        [time, x1, x2] or just [x1, x2]), and these should be preserved during normalisation.
        """
        coords = dict(
            time=pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            lat=np.linspace(20, 40, 30),
            lon=np.linspace(40, 60, 20),
            station=["A", "B"],
        )
        df_raw = _gen_data_pandas(coords=coords)

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="lat",
            x2_name="lon",
        )

        df_norm = dp(df_raw)
        df_unnorm = dp.unnormalise(df_norm)

        self.assertListEqual(list(df_raw.index.names), list(df_unnorm.index.names))
        self.assertTrue(
            self.assert_allclose_pd(df_unnorm, df_raw),
            f"Original {type(df_raw).__name__} not restored.",
        )

    def test_wrong_extra_indexes_pandas(self):
        """
        Other metadata indexes are allowed but if they are not *after* the default dimension
        indexes of [time, x1, x2] or just [x1, x2], then an error should be raised.
        """
        coords = dict(
            station=["A", "B"],
            time=pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            lat=np.linspace(20, 40, 30),
            lon=np.linspace(40, 60, 20),
        )
        df_raw = _gen_data_pandas(coords=coords)

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="lat",
            x2_name="lon",
        )

        with self.assertRaises(ValueError):
            dp(df_raw)

def test_norm_unnorm_preserves_coords_xr(self):
    region_size = (61,81)
    lat_lims = (30,75)
    lon_lims = (-15,45)
    
    latitude = np.linspace(*lat_lims, region_size[0], dtype=np.float32)
    longitude = np.linspace(*lon_lims, region_size[1], dtype=np.float32)
    dummy_data = np.random.normal(size=region_size)
    da_raw = xr.DataArray(dummy_data, dims=['latitude', 'longitude'], coords={'latitude':latitude, 'longitude':longitude})

    data_processor = DataProcessor(
        x1_name='latitude', x1_map=lat_lims,
        x2_name='longitude', x2_map=lon_lims
    )

    da = data_processor(da_raw) # to compute normalisation params

    da_norm = data_processor.map_coords(da_raw)
    da_unnorm = data_processor.unnormalise(da_norm)

    try:
        xr.align(da_raw, da_unnorm, join='exact')
    except ValueError:
        self.fail('Normalise-unnormalise did not preserve xarray coordinates exactly')

if __name__ == "__main__":
    unittest.main()
