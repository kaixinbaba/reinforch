import sys

from logbook import Logger, StreamHandler, INFO, DEBUG, WARNING, ERROR

StreamHandler(sys.stdout).push_application()


def Log(name, level=INFO):
    return Logger(name, level)


logging = Log(__name__)
