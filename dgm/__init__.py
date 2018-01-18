__all__ = ["basemodel", "tfmodel", "kerasmodel", "loader", "config" ]

import os
import logging

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(name)s: %(levelname)s] %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p")
logging.info("dgm imported")


def find_file(name):
    for prefix_dir in [".", "config", "data"]:
        file_name = prefix_dir + name
        if os.path.exists(file_name):
            logging.warning("Found config/data file:%s", file_name)
            return file_name
    logging.critical("File not found: %s", name)
    return None
