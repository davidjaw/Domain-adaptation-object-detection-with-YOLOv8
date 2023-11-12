import os
from ultralytics.engine.trainer import BaseTrainer


def create_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    else:
        counter = 0
        new_path = f"{base_path}{counter}"
        # Keep incrementing the counter until the path doesn't exist
        while os.path.exists(new_path):
            counter += 1
            new_path = f"{base_path}{counter}"
        return new_path




