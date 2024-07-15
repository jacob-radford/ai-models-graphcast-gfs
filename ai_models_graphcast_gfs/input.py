# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from collections import defaultdict
from scipy.interpolate import RegularGridInterpolator
import climetlab as cml
import numpy as np
import xarray as xr

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


def forcing_variables_numpy(sample, forcing_variables, dates, onedeg):
    """Generate variables from climetlabs

    Args:
        date (datetime): Datetime of current time step in forecast
        params (List[str]): Parameters to calculate as constants

    Returns:
        torch.Tensor: Tensor with constants
    """
    ds = cml.load_source(
        "constants",
        sample,
        date=dates,
        param=forcing_variables,
    )

    if not onedeg:
        force = ds.order_by(param=forcing_variables, valid_datetime="ascending").to_numpy(dtype=np.float32).reshape(len(forcing_variables), len(dates), 721, 1440)
    else:
        force = np.zeros((len(forcing_variables),len(dates),181,360))
        for k,forcing_variable in enumerate(ds.order_by(param=forcing_variables, valid_datetime="ascending")):
            force[0,k,:,:] = interpolate(forcing_variable.to_numpy(dtype=np.float32))
        force = force.reshape(len(forcing_variables),len(dates),181,360)
    return (
        force
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
    onedeg
):
    time_deltas = [
        datetime.timedelta(hours=h)
        for h in lagged
        + [hour for hour in range(hour_steps, lead_time + hour_steps, hour_steps)]
    ]

    all_datetimes = [start_date() + time_delta for time_delta in time_deltas]

    with timer("Creating forcing variables"):
        forcing_numpy = forcing_variables_numpy(
            fields_sfc, forcing_variables, all_datetimes,onedeg
        )

    with timer("Converting GRIB to xarray"):
        # Create Input dataset

        if not onedeg:
            lat = fields_sfc[0].metadata("distinctLatitudes")
            lon = fields_sfc[0].metadata("distinctLongitudes")
        else:
            lat = np.arange(90,-91,-1)
            lon = np.arange(0,360,1)
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
            if param in ["100u","100v","sp","tcwv"]:
                continue

            if param in ("z", "lsm"):
                if not onedeg:
                    data_vars[CF_NAME_SFC[param]] = (["lat", "lon"], fields[0].to_numpy())
                else:   
                    data_vars[CF_NAME_SFC[param]] = (["lat", "lon"], interpolate(fields[0].to_numpy()))
                continue
            
            if not onedeg: 
                data = np.stack(
                    [field.to_numpy(dtype=np.float32) for field in fields]
                ).reshape(
                    1,
                    len(given_datetimes),
                    len(lat),
                    len(lon),
                )
            else:
                data = np.stack(
                    [interpolate(field.to_numpy(dtype=np.float32)) for field in fields]
                ).reshape(
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
            if param in ["r"]:
                continue
            if not onedeg:
                data = np.stack(
                    [field.to_numpy(dtype=np.float32) for field in fields]
                ).reshape(
                    1,
                    len(given_datetimes),
                    len(levels),
                    len(lat),
                    len(lon),
                )
            else:
                data = np.stack(
                    [interpolate(field.to_numpy(dtype=np.float32)) for field in fields]
                ).reshape(
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
        training_xarray = training_xarray.reindex(
            lat=sorted(training_xarray.lat.values)
        )

    if constants:
        # Add geopotential_at_surface and land_sea_mask back in
        x = xr.load_dataset(constants)

        for patch in ("geopotential_at_surface", "land_sea_mask"):
            LOG.info("PATCHING %s", patch)
            if not onedeg:
                training_xarray[patch] = x[patch]
            else:
                print(x[patch])
#                training_xarray[patch] = 

    return training_xarray, time_deltas

def interpolate(data):
    qlats = np.arange(90,-90.25,-0.25)
    qlons = np.arange(0,360,0.25)
    interpolator = RegularGridInterpolator((qlats,qlons),data)
    olats = np.arange(90,-91,-1)
    olons = np.arange(0,360,1)
    olon_grid,olat_grid = np.meshgrid(olons,olats)
    points = np.array([olat_grid.flatten(),olon_grid.flatten()]).T
    data_interpolated = interpolator(points).reshape(olat_grid.shape)
    return data_interpolated
