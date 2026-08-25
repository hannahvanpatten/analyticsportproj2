"""
This script computes Mean Absolute Percentage Error (MAPE) at multiple levels:
- Overall MAPE for the entire test set
- MAPE by product
- MAPE by forecast horizon (time period)

Usage:
    python calculate_mape.py

Outputs:
    - outputs/mape_summary.csv              # Overall MAPE for all three models
    - outputs/mape_by_product.csv           # MAPE broken down by product for all models
    - outputs/mape_by_horizon.csv           # MAPE broken down by time period for all models
"""

import os
import pandas as pd
import numpy as np