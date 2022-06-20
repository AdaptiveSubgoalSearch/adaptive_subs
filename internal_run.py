# """This is used to execute specification from inside of the code. Used by pytest and pycharm debugger"""
import gin

import jobs
from runner import run


def internal_run(spec_path):
    gin.clear_config()
    gin_bindings = []
    from mrunner_utils import mrunner_client
    specification, overrides = mrunner_client.get_configuration(spec_path)
    gin_bindings.extend(overrides)
    gin.parse_config_files_and_bindings(None, gin_bindings)
    return run()