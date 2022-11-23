import random


class Pipes:

    def __init__(self):
        self.curX = 1300
        self.topY = -400 + random.randint(-50, 50)
        self.bottomY = self.topY + 750
