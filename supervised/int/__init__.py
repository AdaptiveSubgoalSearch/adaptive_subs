import gin

from supervised.int.representation import infix
from supervised.int.representation import prefix
from supervised.int.representation import action_representation_mask
from supervised.int.representation import infix_value


def configure_class(int_class):
    return gin.external_configurable(
        int_class, module='supervised.INT'
    )

InfixRepresentation = configure_class(infix.InfixRepresentation)
PrefixRepresentation = configure_class(prefix.PrefixRepresentation)
ActionRepresentationMask = configure_class(action_representation_mask.ActionRepresentationMask)
InfixValueRepresentation = configure_class(infix_value.InfixValueRepresentation)