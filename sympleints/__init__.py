import logging

from sympleints.helpers import canonical_order, get_center, get_map, get_timer_getter, shell_iter, Timer
from sympleints.version import version as __version__

logger = logging.getLogger("sympleints")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("sympleints.log", mode="w", delay=True)
logger.addHandler(file_handler)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)
