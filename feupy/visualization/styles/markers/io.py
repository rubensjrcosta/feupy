"""I/O utilities for marker dictionaries."""

import os
import yaml

def write_marker_dict(marker_dict, filename, overwrite=False):
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"{filename} exists.")

    with open(filename, "w") as f:
        yaml.safe_dump({"ref_markers": marker_dict}, f)


def read_marker_dict(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    with open(filename) as f:
        data = yaml.safe_load(f)

    return data.get("ref_markers", {})
