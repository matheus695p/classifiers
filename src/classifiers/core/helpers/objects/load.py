import importlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)


def load_object(model_params: dict, verbose: bool = False) -> Any:
    """Load object.

    Loads a object from the class given as a parameter.

    Args:
        model_params: dictionary of parameters for train
        verbose: print logs.

    Returns:
        Any python object.
    """
    model_class = model_params["class"]
    model_kwargs = model_params["kwargs"]
    if model_params["kwargs"] == None:
        model_kwargs = {}
    if verbose:
        logger.info(f"loading with {model_params}")
    python_object = _load_obj(model_class)(**model_kwargs)
    return python_object


def load_object_with_arg(model_params: dict, extra_args: Dict):
    """Load from catalog sklearn transformer.

    Loads a regressor object based on given parameters.

    Args:
        model_params: dictionary of parameters for train

    Returns:
        sklearn compatible model
    """
    model_class = model_params["class"]
    model_kwargs = model_params["kwargs"]
    extra_args.update(model_kwargs)
    if model_params["kwargs"] == None:
        model_kwargs = {}
    baseline_model = _load_obj(model_class)(**extra_args)
    return baseline_model
