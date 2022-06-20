"""Alpacka initialization."""

import gin

from alpacka.third_party import dask


gin.config._OPERATIVE_CONFIG_LOCK = dask.SerializableLock()  # pylint: disable=protected-access
