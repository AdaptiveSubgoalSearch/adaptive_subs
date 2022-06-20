import gin

from goal_verificators import verificator_rubik, verificator_int


def configure_goal_verificator(goal_verificator_class):
    return gin.external_configurable(
        goal_verificator_class, module='goal_verificators'
    )


VerificatorINT = configure_goal_verificator(verificator_int.VerificatorInt)
VerificatorRubik = configure_goal_verificator(verificator_rubik.VerificatorRubik)
