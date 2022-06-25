import logging
import warnings
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from classifiers.core.helpers.objects.load import load_object

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


logger = logging.getLogger(__name__)
