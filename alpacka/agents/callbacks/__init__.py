"""Callbacks."""

import gin

from alpacka.agents.callbacks import graph_size_callback


# Configure callbacks in this module to ensure they're accessible via the
# alpacka.agents.callbacks.* namespace.
def configure_callback(callback_class):
    return gin.external_configurable(
        callback_class, module='alpacka.agents.callbacks'
    )


GraphSizeCallback = configure_callback(graph_size_callback.GraphSizeCallback)  # pylint: disable=invalid-name
