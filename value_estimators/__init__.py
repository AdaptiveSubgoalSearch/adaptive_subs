import gin

from value_estimators import value_estimator
from value_estimators import value_estimator_rubik
from value_estimators.int import value_estimator_int


def configure_value_estimator(value_estimator_class):
    return gin.external_configurable(
        value_estimator_class, module='value_estimators'
    )


ValueEstimator = configure_value_estimator(value_estimator.ValueEstimator)
ValueEstimatorRubik = configure_value_estimator(value_estimator_rubik.ValueEstimatorRubik)
TrivialValueEstimatorINT = configure_value_estimator(value_estimator_int.TrivialValueEstimatorINT)
ValueEstimatorINT = configure_value_estimator(value_estimator_int.ValueEstimatorINT)
