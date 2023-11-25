from kinn.configs.template import PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE
from typing import Union
def read_ARGS(path) -> Union[PRIMNET_ARGS_TEMPLATE, FC_PRIMNET_ARGS_TEMPLATE, PCC_PRIMNET_ARGS_TEMPLATE]:
    import importlib
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ARGS = getattr(module,'ARGS')()
    return ARGS

def isPrimitive(cast_type):
    return cast_type in [int, float, str, bool]


def ismodule(class_, module_):
    return hasattr(module_, class_.__name__)    
    
def cast_bool(cast_from):
    if type(cast_from) is str:
        if cast_from == 'True':
            return True
        elif cast_from == "False":
            return False
    
    if type(cast_from) is bool:
        return cast_from
    
def cast_like(cast_from, cast_to, cast_candidate):
    cast_type = type(cast_to)
    
    if isPrimitive(cast_type):
        if cast_type is bool:
            return cast_bool(cast_from)
        else:
            return cast_type(cast_from)
    
    for cast_candidate_ in cast_candidate:
        if ismodule(cast_to ,cast_candidate_):
            return getattr(cast_candidate_, cast_from)
    
    raise LookupError(f"Can not find appropriate cast_candidate for {cast_from}")
