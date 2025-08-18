import logging
import sys


def setup_default_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
