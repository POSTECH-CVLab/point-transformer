from time import time

class Timer:
    def __init__(self, tag, print=False):
        self.tag = tag
        self.ts = None
        self.print = print

    def tic(self):
        self.ts = time()

    def toc(self):
        if self.print:
            print("{}: {}s".format(self.tag, time() - self.ts))
        return time()
