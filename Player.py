class Player:
    floor = 475
    velocity = 0

    def __init__(self):
        self.curX = 200
        self.curY = Player.floor/2
    
    def get_coords(self):
        return self.curX, self.curY
    
    def move_up(self):
        self.velocity = 8

    def hit_ceiling(self):
        return self.curY == 0

    def hit_ground(self):
        return self.curY == self.floor

    def update(self):
        self.curY -= self.velocity
        self.velocity -= 0.4
        if self.curY < 0:
            self.curY = 0
        elif self.curY > self.floor:
            self.curY = self.floor

