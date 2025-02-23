# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from collections import defaultdict

import earthkit.data as ekd
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

LOG = logging.getLogger(__name__)

CF_NAME_SFC = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2t": "2m_temperature",
    "lsm": "land_sea_mask",
    "msl": "mean_sea_level_pressure",
    "tp": "total_precipitation_6hr",
    "z": "geopotential_at_surface",
}

CF_NAME_PL = {
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "z": "geopotential",
}


def forcing_variables_numpy(sample, forcing_variables, dates):
    """Generate variables from earthkit-data

    Args:
        date (datetime): Datetime of current time step in forecast
        params (List[str]): Parameters to calculate as constants

    Returns:
        torch.Tensor: Tensor with constants
    """
    ds = ekd.from_source(
        "forcings",
        sample,
        date=dates,
        param=forcing_variables,
    )

    return (
        ds.order_by(param=forcing_variables, valid_datetime="ascending")
        .to_numpy(dtype=np.float32)
        .reshape(len(forcing_variables), len(dates), 721, 1440)
    )


def create_training_xarray(
    *,
    fields_sfc,
    fields_pl,
    lagged,
    start_date,
    hour_steps,
    lead_time,
    forcing_variables,
    constants,
    timer,
):
    time_deltas = [
        datetime.timedelta(hours=h)
        for h in lagged + [hour for hour in range(hour_steps, lead_time + hour_steps, hour_steps)]
    ]

    all_datetimes = [start_date + time_delta for time_delta in time_deltas]

    with timer("Creating forcing variables"):
        forcing_numpy = forcing_variables_numpy(fields_sfc, forcing_variables, all_datetimes)

    with timer("Converting GRIB to xarray"):
        # Create Input dataset

        lat = fields_sfc[0].metadata("distinctLatitudes")
        lon = fields_sfc[0].metadata("distinctLongitudes")

        # SURFACE FIELDS

        fields_sfc = fields_sfc.order_by("param", "valid_datetime")
        sfc = defaultdict(list)
        given_datetimes = set()
        for field in fields_sfc:
            given_datetimes.add(field.metadata("valid_datetime"))
            sfc[field.metadata("param")].append(field)

        # PRESSURE LEVEL FIELDS

        fields_pl = fields_pl.order_by("param", "valid_datetime", "level")
        pl = defaultdict(list)
        levels = set()
        given_datetimes = set()
        for field in fields_pl:
            given_datetimes.add(field.metadata("valid_datetime"))
            pl[field.metadata("param")].append(field)
            levels.add(field.metadata("level"))

        data_vars = {}

        for param, fields in sfc.items():
            if param not in CF_NAME_SFC.keys():
                continue
            if param in ("z", "lsm"):
                data_vars[CF_NAME_SFC[param]] = (["lat", "lon"], fields[0].to_numpy())
                continue

            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(
                1,
                len(given_datetimes),
                len(lat),
                len(lon),
            )

            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, len(all_datetimes) - len(given_datetimes)),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=(np.nan,),
            )

            data_vars[CF_NAME_SFC[param]] = (["batch", "time", "lat", "lon"], data)

        for param, fields in pl.items():
            if param not in CF_NAME_PL.keys():
                continue
            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(
                1,
                len(given_datetimes),
                len(levels),
                len(lat),
                len(lon),
            )
            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, len(all_datetimes) - len(given_datetimes)),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=(np.nan,),
            )

            data_vars[CF_NAME_PL[param]] = (
                ["batch", "time", "level", "lat", "lon"],
                data,
            )

        data_vars["toa_incident_solar_radiation"] = (
            ["batch", "time", "lat", "lon"],
            forcing_numpy[0:1, :, :, :],
        )

        training_xarray = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=lon,
                lat=lat,
                time=time_deltas,
                datetime=(
                    ("batch", "time"),
                    [all_datetimes],
                ),
                level=sorted(levels),
            ),
        )

    with timer("Reindexing"):
        # And we want the grid south to north
        training_xarray = training_xarray.reindex(lat=sorted(training_xarray.lat.values))

    if constants:
        # Add geopotential_at_surface and land_sea_mask back in
        x = xr.load_dataset(constants)

        for patch in ("geopotential_at_surface", "land_sea_mask"):
            LOG.info("PATCHING %s", patch)
            training_xarray[patch] = x[patch]

    return training_xarray, time_deltas

def forcing_variables_numpy_1deg(sample, forcing_variables, dates, lat, lon, new_lat, new_lon):
    """Generate variables from earthkit-data

    Args:
        date (datetime): Datetime of current time step in forecast
        params (List[str]): Parameters to calculate as constants

    Returns:
        torch.Tensor: Tensor with constants
    """
    ds = ekd.from_source(
        "forcings",
        sample,
        date=dates,
        param=forcing_variables,
    )

    # Step 1: Order dataset
    ordered_ds = ds.order_by(param=forcing_variables, valid_datetime="ascending")

    # Step 2: Convert to NumPy array
    numpy_array = ordered_ds.to_numpy(dtype=np.float32)

    # Step 3: Reshape to original spatial grid (721, 1440)
    reshaped_array = numpy_array.reshape(len(forcing_variables), len(dates), 721, 1440)

    # Step 4: Create empty array for interpolated data
    interpolated_data = np.zeros((reshaped_array.shape[0], reshaped_array.shape[1], 181, 360), dtype=np.float32)

    # Step 5: Perform interpolation
    for i in range(len(forcing_variables)):
        for j in range(len(dates)):
            interpolated_data[i, j] = interpolate(reshaped_array[i, j], lat, lon, new_lat, new_lon)

    # Return final interpolated data
    return interpolated_data

def create_training_xarray_1deg(
    *,
    fields_sfc,
    fields_pl,
    lagged,
    start_date,
    hour_steps,
    lead_time,
    forcing_variables,
    constants,
    timer,
):
    time_deltas = [
        datetime.timedelta(hours=h)
        for h in lagged + [hour for hour in range(hour_steps, lead_time + hour_steps, hour_steps)]
    ]

    all_datetimes = [start_date + time_delta for time_delta in time_deltas]

    lat = fields_sfc[0].metadata("distinctLatitudes")
    lon = fields_sfc[0].metadata("distinctLongitudes")

    new_lat = np.arange(-90,91,1)[::-1]
    new_lon = np.arange(0,360,1)

    with timer("Creating forcing variables"):
        forcing_numpy = forcing_variables_numpy_1deg(fields_sfc, forcing_variables, all_datetimes, lat, lon, new_lat, new_lon)

    with timer("Converting GRIB to xarray"):
        # Create Input dataset

        lat = fields_sfc[0].metadata("distinctLatitudes")
        lon = fields_sfc[0].metadata("distinctLongitudes")

        # SURFACE FIELDS

        fields_sfc = fields_sfc.order_by("param", "valid_datetime")
        sfc = defaultdict(list)
        given_datetimes = set()
        for field in fields_sfc:
            given_datetimes.add(field.metadata("valid_datetime"))
            sfc[field.metadata("param")].append(field)

        # PRESSURE LEVEL FIELDS

        fields_pl = fields_pl.order_by("param", "valid_datetime", "level")
        pl = defaultdict(list)
        levels = set()
        given_datetimes = set()
        for field in fields_pl:
            given_datetimes.add(field.metadata("valid_datetime"))
            pl[field.metadata("param")].append(field)
            levels.add(field.metadata("level"))

        data_vars = {}

        for param, fields in sfc.items():
            if param not in CF_NAME_SFC.keys():
                continue
            if param in ("z", "lsm"):
                data_vars[CF_NAME_SFC[param]] = (["lat", "lon"], interpolate(fields[0].to_numpy(),lat,lon,new_lat,new_lon)) 
                continue

            data = np.stack([interpolate(field.to_numpy(dtype=np.float32),lat,lon,new_lat,new_lon) for field in fields]).reshape(
                1,
                len(given_datetimes),
                len(new_lat),
                len(new_lon),
            )

            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, len(all_datetimes) - len(given_datetimes)),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=(np.nan,),
            )

            data_vars[CF_NAME_SFC[param]] = (["batch", "time", "lat", "lon"], data)

        for param, fields in pl.items():
            if param not in CF_NAME_PL.keys():
                continue
            data = np.stack([interpolate(field.to_numpy(dtype=np.float32),lat,lon,new_lat,new_lon) for field in fields]).reshape(
                1,
                len(given_datetimes),
                len(levels),
                len(new_lat),
                len(new_lon),
            )
            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, len(all_datetimes) - len(given_datetimes)),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=(np.nan,),
            )

            data_vars[CF_NAME_PL[param]] = (
                ["batch", "time", "level", "lat", "lon"],
                data,
            )

        data_vars["toa_incident_solar_radiation"] = (
            ["batch", "time", "lat", "lon"],
            forcing_numpy[0:1, :, :, :],
        )

        training_xarray = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=new_lon,
                lat=new_lat,
                time=time_deltas,
                datetime=(
                    ("batch", "time"),
                    [all_datetimes],
                ),
                level=sorted(levels),
            ),
        )

    with timer("Reindexing"):
        # And we want the grid south to north
        training_xarray = training_xarray.reindex(lat=sorted(training_xarray.lat.values))

    if constants:
        print("CONSTANTS")
        # Add geopotential_at_surface and land_sea_mask back in
        x = xr.load_dataset(constants)

        for patch in ("geopotential_at_surface", "land_sea_mask"):
            LOG.info("PATCHING %s", patch)
            training_xarray[patch] = x[patch]

    return training_xarray, time_deltas

def interpolate(data, lat, lon, new_lat, new_lon):
    lon_extended = np.concatenate(([lon[-1] - 360], lon, [lon[0] + 360]))
    data_extended = np.concatenate((data[:, -1:], data, data[:, :1]), axis=1)
    interpolator = RegularGridInterpolator((lat, lon_extended), data_extended, bounds_error=False)
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon,new_lat)
    points = np.array([new_lat_grid.flatten(),new_lon_grid.flatten()]).T
    data_interpolated = interpolator(points).reshape(new_lat_grid.shape)
    return data_interpolated
