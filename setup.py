#!/usr/bin/env python
# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import io
import os

import setuptools


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(file_path, encoding="utf-8").read()


version = "0.0.6"

assert version


setuptools.setup(
    name="ai-models-graphcast-gfs",
    version=version,
    description="Run graphcast ai weather models with capabilities for GFS and GDAS initial conditions and NetCDF output",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Jacob Radford",
    author_email="jacob.t.radford@gmail.com",
    license="Apache License Version 2.0",
    url="https://github.com/jacob-radford/ai-models-graphcast-gfs",
    packages=setuptools.find_packages(),
    include_package_data=True,
    # JAX requirements are in requirements.txt
    install_requires=[
        "dm-tree",
        "dm-haiku==0.0.10",
    ],
    zip_safe=True,
    keywords="tool",
    entry_points={
        "ai_models_gfs.model": [
            "graphcast = ai_models_graphcast_gfs.model:model",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
    ],
)
