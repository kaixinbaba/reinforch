import sys

from logbook import Logger, StreamHandler, INFO, DEBUG, WARNING, ERROR, CRITICAL

StreamHandler(sys.stdout).push_application()


def Log(name, level=INFO):
    return Logger(name, level)


LEVEL = dict(
    debug=DEBUG,
    info=INFO,
    warning=WARNING,
    warn=WARNING,
    error=ERROR,
    critical=CRITICAL,
)

level = LEVEL['info']

logging = Log(__name__)
