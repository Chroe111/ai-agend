import builtins


def print(msg: str, debug: bool=True):
    if debug:
        builtins.print(msg)
