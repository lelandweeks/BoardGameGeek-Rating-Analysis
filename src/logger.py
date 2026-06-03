"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import os
import json
from datetime import datetime

LOG_PATH = "output/runs_log.json"

def log_run(question, model_name, features, preprocessing, feature_engineering, results):

    # load existing log if it exists
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            log = json.load(f)
    else:
        log = []

    # build the entry for this run
    entry = {
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question":            question,
        "model":               model_name,
        "features":            features,
        "preprocessing":       preprocessing,
        "feature_engineering": feature_engineering,
        "results":             results,
    }

    log.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=4)

    print(f"Run logged to {LOG_PATH}")
