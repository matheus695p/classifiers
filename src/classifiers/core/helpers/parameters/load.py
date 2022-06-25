import logging

from kedro.config import ConfigLoader

logger = logging.getLogger(__name__)

CONF_SOURCE = "conf"


def load_parameters() -> dict:
    """Get kedro parameters

    Returns:
        dict: Kedro parameters
    """
    conf_loader = ConfigLoader(CONF_SOURCE)
    params = conf_loader.get("parameters*", "parameters*/**")

    return params
