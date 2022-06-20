"""Metric logging."""

import collections
import re


class LoggerWithSmoothing:
    """Abstract base class for logging with smoothing.
    This logger automatically adds smoothed version of specified metrics.
    """

    def __init__(self):
        self._previous_values = collections.defaultdict(float)
        self._smoothing_coeffs = collections.defaultdict(float)

    def _log_scalar(self, name, step, value):
        """Implements a particular logging method (e.g. to stdout)."""
        raise NotImplementedError()

    def log_scalar(self, name, step, value, smoothing=None):
        """log scalar value. The method adds smoothing.

        Args:
            name (str): Name of the metric to log.
            step (int): Step of the logger.
            value (float): Value to be logged.
            smoothing (tuple): Specification of smoothing of metrics.
                (regex, coeff), where of regex is a regex of metrics names which
                should be smoothed. coeff in (0,1) indices the desired smoothing
                parameter. Eg. ('return', 0.99) will calculate the running
                average of all metrics which contain 'return' in their name.
                The average is calculated according to
                average = 0.99*average + 0.01*new_value.
                Warm-up period is also applied.
            """
        self._log_scalar(name, step, value)
        if smoothing is not None:
            smoothing_regex, smoothing_coeff = smoothing
            if re.search(smoothing_regex, name) is not None:
                name_smoothed = name + rf'/smoothing_{smoothing_coeff}'
                prev_value = self._previous_values[name_smoothed]
                prev_smoothing_coeff = self._smoothing_coeffs[name_smoothed]

                # This implements warm-up period of length "10". That is
                # the smoothing coefficient start with 0 and is annelled to the
                # desired value.
                # This results in better estimates of soothed value
                # at the beginning, which might be useful for short experiments.
                new_smoothing_coeff = prev_smoothing_coeff * 0.9 + \
                                      smoothing_coeff * 0.1
                new_value = value * (1 - prev_smoothing_coeff) \
                            + prev_value * prev_smoothing_coeff

                self._previous_values[name_smoothed] = new_value
                self._smoothing_coeffs[name_smoothed] = new_smoothing_coeff
                self._log_scalar(name_smoothed, step, new_value)


class StdoutLogger(LoggerWithSmoothing):
    """Logs to standard output."""

    def _log_scalar(self, name, step, value):
        """Logs a scalar to stdout."""
        # Format:
        #      1 | accuracy:                   0.789
        #   1234 | loss:                      12.345
        print('{:>6} | {:32}{:>9.3f}'.format(step, name + ':', value))

    @staticmethod
    def log_property(name, value):
        # Not supported in this logger.
        pass


_loggers = [StdoutLogger()]


def register_logger(logger):
    """Adds a logger to log to."""
    _loggers.append(logger)


def log_scalar(name, step, value, smoothing=None):
    """Logs a scalar to the loggers."""
    for logger in _loggers:
        logger.log_scalar(name, step, value, smoothing)


def log_property(name, value):
    """Logs a property to the loggers."""
    for logger in _loggers:
        logger.log_property(name, value)


def log_scalar_metrics(prefix, step, metrics, smoothing=None):
    for (name, value) in metrics.items():
        log_scalar(prefix + '/' + name, step, value, smoothing=smoothing)
