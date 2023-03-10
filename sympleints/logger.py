import logging
import re
import sys


def escape_ansi(line):
    """Strip ANSI escape sequences from given line.

    From https://stackoverflow.com/a/38662876."""
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


class StripFormatter(logging.Formatter):

    def format(self, record):
        record.msg = escape_ansi(record.msg)
        return super().format(record)


logger = logging.getLogger("sympleints")
logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)

file_handler = logging.FileHandler("sympleints.log", mode="w", delay=True)
file_handler.setFormatter(StripFormatter())
logger.addHandler(file_handler)


bench_logger = logging.getLogger("symplebench")
bench_logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
bench_logger.addHandler(stdout_handler)

file_handler = logging.FileHandler("symplebench.log", mode="w", delay=True)
file_handler.setFormatter(StripFormatter())
bench_logger.addHandler(file_handler)
