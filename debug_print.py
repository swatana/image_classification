
class Color:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'


def debug_print():
    import inspect
    import os
    import datetime
    original_print = print

    def debug(*args, **kwargs):
        # def debug(*args):
        dt_now = datetime.datetime.now()
        frame = inspect.currentframe().f_back
        color = Color.WHITE
        if 'color' in kwargs:
            color = getattr(Color, kwargs['color'].upper())
        # if 'blue' in args:
        #     color = Color.BLUE
        # if 'red' in args:
        #     color = Color.RED
        # if 'green' in args:
        #     color = Color.GREEN
        # if 'yellow' in args:
        #     color = Color.YELLOW
        # if color is not Color.WHITE:
            # args.pop(-1)
            # args = tuple(list(args).pop(-1))
            # args = tuple(kwargs)
        original_print("[" + str(dt_now) + " " + os.path.basename(frame.f_code.co_filename) +
                       ":" + str(frame.f_lineno) + "]", color, *args, Color.END)
        # print(Color.RED + str('%f' % diff_btc) + Color.END)
        pass
    return debug
