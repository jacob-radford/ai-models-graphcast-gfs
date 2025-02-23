# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import eccodes
import numpy as np

from .convert import GRIB_TO_CF
from .convert import GRIB_TO_XARRAY_PL
from .convert import GRIB_TO_XARRAY_SFC

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
    lagged,
    onedeg
):
    LOG.info("Converting output xarray to GRIB and saving")

    output["total_precipitation_6hr"] = output.data_vars["total_precipitation_6hr"].cumsum(dim="time")

    all_fields = all_fields.sel(param_level=ordering, remapping={"param_level": "{param}{levelist}"})

    all_fields = all_fields.order_by(
        valid_datetime="descending",
        param_level=ordering,
        remapping={"param_level": "{param}{levelist}"},
    )

    for time in range(lead_time // hour_steps):
        for fs in all_fields[: len(all_fields) // len(lagged) + 1]:
            param, level = fs.metadata("shortName"), fs.metadata("levelist", default=None)
            if level is not None:
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
            if onedeg:
                grib_handle = fs.handle._handle
                eccodes.codes_set(grib_handle, "Ni", 360)  # Longitude points
                eccodes.codes_set(grib_handle, "Nj", 181)  # Latitude points

                # Set correct grid spacing for 1-degree resolution
                eccodes.codes_set(grib_handle, "iDirectionIncrementInDegrees", 1.0)
                eccodes.codes_set(grib_handle, "jDirectionIncrementInDegrees", 1.0)

                # Define latitude/longitude bounds
                eccodes.codes_set(grib_handle, "latitudeOfFirstGridPointInDegrees", 90)
                eccodes.codes_set(grib_handle, "longitudeOfFirstGridPointInDegrees", 0)
                eccodes.codes_set(grib_handle, "latitudeOfLastGridPointInDegrees", -90)
                eccodes.codes_set(grib_handle, "longitudeOfLastGridPointInDegrees", 359)

                values = np.flipud(values.reshape((181,360)))
            else:
                values = np.flipud(values.reshape(fs.shape))

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
