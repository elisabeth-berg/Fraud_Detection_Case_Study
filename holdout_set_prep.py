"""Splits the data into holdout and training sets.
data.json not available on this github repo because of confidentiality."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

full_data = pd.read_json('data/data.json')

train_data, holdout_data = train_test_split(full_data, test_size=0.20, random_state=56)

train_data.to_json('data/train_data.json')
holdout_data.to_json('data/holdout_data.json')
