from time import time

class Timer:
    def __init__(self, tag):
        self.tag = tag
        self.ts = None

    def tic(self):
        self.ts = time()

    def toc(self):
        print("{}: {}s".format(self.tag, time() - self.ts))
        return time()
