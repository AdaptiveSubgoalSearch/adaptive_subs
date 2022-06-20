"""Parameter schedules."""

import gin


@gin.configurable
class LinearAnnealing:
    """Linear annealing parameter schedule."""

    def __init__(self, max_value, min_value, n_epochs):
        """Initializes LinearAnnealing schedule.

        Args:
            max_value (float): Maximal (starting) parameter value.
            min_value (float): Minimal (final) parameter value.
            n_epochs (int): Across how many epochs parameter should reach from
                its starting to its final value.
        """
        self._min_value = min_value
        self._slope = - (max_value - min_value) / n_epochs
        self._intersect = max_value

    def __call__(self, epoch):
        return max(self._min_value, self._slope * epoch + self._intersect)


@gin.configurable
class RsqrtAnnealing:
    """Reciprocal square root annealing parameter schedule."""

    def __init__(self, max_value=1, scale=1):
        """Initializes RsqrtAnnealing schedule.

        Decreases from max_value at epoch=0 to 0 at epoch->inf.

        Args:
            max_value (float): Maximum (initial) value.
            scale (float): Scale in the x axis - scale 2 means the curve will
                decrease 2 times slower than at scale 1.
        """
        self._max_value = max_value
        self._scale = scale

    def __call__(self, epoch):
        return self._max_value / (1 + epoch / self._scale) ** 0.5
