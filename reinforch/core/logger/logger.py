import sys

from logbook import Logger, StreamHandler, INFO, DEBUG, WARNING, ERROR, CRITICAL

StreamHandler(sys.stdout).push_application()

LEVEL = dict(
    debug=DEBUG,
    info=INFO,
    warning=WARNING,
    warn=WARNING,
    error=ERROR,
    critical=CRITICAL,
)

default_level = LEVEL['info']


def Log(name, level=default_level):
    return Logger(name, level)


logging = Log(__name__)
