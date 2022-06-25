import logging
import random

import numpy as np
import sklearn

from classifiers.core.helpers.parameters.load import load_parameters

logger = logging.getLogger(__name__)


def get_global_seed() -> int:
    """Global seed from parameters.

    Returns:
        int: global seed.
    """
    params = load_parameters()

    return params["GLOBAL_SEED"]


def seed_file(
    seed: int, contain_message: bool = False, message="", verbose: bool = True
):
    """It seeds the random number generator of the three libraries.

    Args:
      seed (int): int
      if_message (bool): bool. Defaults to False
      message: If True, print the message.
      verbose: If True, print the message.

    Returns: None
    """
    sklearn.utils.check_random_state(seed)
    np.random.seed(seed)
    random.seed(seed)
    if verbose:
        if not contain_message:
            logger.info(
                f"Seeding sklearn, numpy and random libraries with the seed {seed}"
            )
        else:
            logger.info(f"{message}")
