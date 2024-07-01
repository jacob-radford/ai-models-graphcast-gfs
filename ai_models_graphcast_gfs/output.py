# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import datetime
import xarray as xr
from netCDF4 import Dataset as DS

from .convert import GRIB_TO_CF, GRIB_TO_XARRAY_PL, GRIB_TO_XARRAY_SFC

LOG = logging.getLogger(__name__)


def save_output_xarray(
    *,
    output,
    target_variables,
    write,
    all_fields,
    ordering,
    lead_time,
    hour_steps,
    lagged
):
    LOG.info("Converting output xarray to GRIB and saving")

    yshape = len(output['lat'])
    xshape = len(output['lon'])

    output["total_precipitation_6hr"] = output.data_vars[
        "total_precipitation_6hr"
    ].cumsum(dim="time")

    all_fields = all_fields.sel(param=["u","v","w","t","z","q","10u","10v","msl","2t","tp","lsm"])
    
    all_fields = all_fields.order_by(
        valid_datetime="descending",
        param_level=ordering,
        remapping={"param_level": "{param}{levelist}"},
    )
    for time in range(lead_time // hour_steps):
        for fs in all_fields[: len(all_fields) // len(lagged)]:
            param, level = fs["shortName"], fs["level"]

            if level != 0:
                param = GRIB_TO_XARRAY_PL.get(param, param)
                if param not in target_variables:
                    continue
                values = output.isel(time=time).sel(level=level).data_vars[param].values
            else:
                param = GRIB_TO_CF.get(param, param)
                param = GRIB_TO_XARRAY_SFC.get(param, param)
                if param not in target_variables:
                    continue
                values = output.isel(time=time).data_vars[param].values

            # We want to field north=>south

            values = np.flipud(values.reshape((yshape,xshape)))

            if param == "total_precipitation_6hr":
                write(
                    values,
                    template=fs,
                    startStep=0,
                    endStep=(time + 1) * hour_steps,
                )
            else:
                write(
                    values,
                    template=fs,
                    step=(time + 1) * hour_steps,
                )

def create_variable(f, name, dimensions, data, attrs):
    if name in ['time','level']:
        dtype = 'i4'
    else:
        dtype = 'f4'
    var = f.createVariable(name, dtype, dimensions,compression='zlib',complevel=4)
    var[:] = data
    for attr_name, attr_value in attrs.items():
        var.setncattr(attr_name, attr_value)

def save_output_xarray_nc(
    input_xr,
    output,
    lead_time,
    hour_steps,
    path,
    date,
    time
):
    yshape = len(input_xr['lat'])
    xshape = len(input_xr['lon'])

    transposed = output.transpose('batch', 'time', 'level', 'lat', 'lon')
    # Initialize the output dictionary with metadata and zero-filled arrays
    out = {
        'u10': {
            'values': np.zeros((lead_time // hour_steps + 1, yshape, xshape)), 
            'name': '10 metre U wind component', 'units': 'm s-1'
        },
        'v10': {
            'values': np.zeros((lead_time // hour_steps + 1, yshape, xshape)), 
            'name': '10 metre V wind component', 'units': 'm s-1'
        },
        't2': {
            'values': np.zeros((lead_time // hour_steps + 1, yshape, xshape)), 
            'name': '2 metre temperature', 'units': 'K'
        },
        'msl': {
            'values': np.zeros((lead_time // hour_steps + 1, yshape, xshape)), 
            'name': 'Pressure reduced to MSL', 'units': 'Pa'
        },
        'apcp': {
            'values': np.zeros((lead_time // hour_steps + 1, yshape, xshape)), 
            'name': '6-hr accumulated precipitation', 'units': 'm'
        },
        't': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, yshape, xshape)), 
            'name': 'Temperature', 'units': 'K'
        },
        'u': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, yshape, xshape)), 
            'name': 'U component of wind', 'units': 'm s-1'
        },
        'v': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, yshape, xshape)), 
            'name': 'V component of wind', 'units': 'm s-1'
        },
        'z': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, yshape, xshape)), 
            'name': 'Geopotential', 'units': 'm2 s-2'
        },
        'q': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, yshape, xshape)), 
            'name': 'Specific humidity', 'units': 'g kg-1'
        },
        'w': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, yshape, xshape)), 
            'name': 'Vertical velocity', 'units': 'Pa s-1'
        }
    }

    # Map of input field names to output keys
    input_fields = {
        't': 'temperature',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
        'z': 'geopotential',
        'q': 'specific_humidity',
        'w': 'vertical_velocity',
        'u10': '10m_u_component_of_wind',
        'v10': '10m_v_component_of_wind',
        'msl': 'mean_sea_level_pressure',
        't2': '2m_temperature'
    }

    # Assign initial values
    for key, value in input_fields.items():
        if key in ['t', 'u', 'v', 'z', 'q', 'w']:
            out[key]['values'][0,:,:,:] = input_xr[value][0,1,::-1,::-1,:]
        else:
            out[key]['values'][0,:,:] = input_xr[value][0,1,::-1,:]

    # Assign zero values to 'apcp' at the initial time
    out['apcp']['values'][0, :, :] = np.zeros((yshape, xshape))

    # Assign output values
    for key, value in input_fields.items():
        if key in ['t', 'u', 'v', 'z', 'q', 'w']:
            out[key]['values'][1:,:,:,:] = transposed[value][0,:,::-1,::-1,:]
        else:
            out[key]['values'][1:,:,:] = transposed[value][0,:,::-1,:]
    # Assign values to 'apcp' from the output dataset
    apcp = transposed['total_precipitation_6hr'][0, :, ::-1, :]
    apcpdiff = apcp.diff(dim='time',label='upper')
    apcpdiff = xr.concat([apcp.isel(time=0),apcpdiff],dim='time')
    apcpdiff = apcpdiff.transpose('time','lat','lon')
    out['apcp']['values'][1:, :, :] = apcpdiff
    
    # Create the output NetCDF file
    outdir = path
    f = DS(outdir, 'w', format='NETCDF4')
    f.createDimension('time', lead_time // hour_steps + 1)
    f.createDimension('level', 13)
    f.createDimension('longitude', xshape)
    f.createDimension('latitude', yshape)

    # Prepare time values for the NetCDF file
    year = str(date)[:4]
    month = str(date)[4:6]
    day = str(date)[6:8]
    hh = str(time).zfill(4)[:2]
    initdt = datetime.datetime.strptime(f"{year}{month}{day}{hh}", "%Y%m%d%H")
    times = [int((initdt + datetime.timedelta(hours=int(i))).timestamp()) for i in 
             np.arange(0, lead_time + hour_steps, hour_steps)]
    # Create time, longitude, latitude, and level variables in the NetCDF file
    create_variable(
        f, 'time', ('time',), np.array(times), {
            'long_name': 'Date and Time', 'units': 'seconds since 1970-1-1', 
            'calendar': 'standard'
        }
    )
    create_variable(
        f, 'longitude', ('longitude',), input_xr['lon'], {
            'long_name': 'Longitude', 'units': 'degree'
        }
    )
    create_variable(
        f, 'latitude', ('latitude',), input_xr['lat'][::-1], {
            'long_name': 'Latitude', 'units': 'degree'
        }
    )
    create_variable(
        f, 'level', ('level',), np.array(
            [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        )[::-1], {'long_name': 'Isobaric surfaces', 'units': 'hPa'}
    )

    # Create variables for each meteorological parameter
    for variable in [
        'u10', 'v10', 't2', 'msl', 't', 'u', 'v', 'z', 'q', 'w', 'apcp'
    ]:
        dims = ('time', 'level', 'latitude', 'longitude') if variable in [
            'u', 'v', 'z', 't', 'q', 'w'
        ] else ('time', 'latitude', 'longitude')
        create_variable(
            f, variable, dims, out[variable]['values'], {
                'long_name': out[variable]['name'], 'units': out[variable]['units']
            }
        )

    # Add global attributes to the NetCDF file
    f.Conventions = 'CF-1.8'
    f.version = '1_2023-10-14'
    f.model_name = 'GraphCast'
    f.model_version = 'v1'
    f.initialization_model = 'GFS'
    f.initialization_time = (
        f"{initdt.year}-{str(initdt.month).zfill(2)}-{str(initdt.day).zfill(2)}"
        f"T{str(initdt.hour).zfill(2)}:00:00"
    )
    f.first_forecast_hour = '0'
    f.last_forecast_hour = '240'
    f.forecast_hour_step = '6'
    f.creation_time = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    f.close()
