import numpy as np


LEFT = np.array([-1, 0])
LEFTUP = np.array([-1, 1])
UP = np.array([0, 1])
UPRIGHT = np.array([1, 1])
RIGHT = np.array([1, 0])
RIGHTDOWN = np.array([1, -1])
DOWN = np.array([0, -1])
DOWNLEFT = np.array([-1, -1])


MOVES = [
    LEFT,
    LEFTUP,
    UP,
    UPRIGHT,
    RIGHT,
    RIGHTDOWN,
    DOWN,
    DOWNLEFT,
]


class Path:
    def __init__(self, cave_init, neib, max_jump=10):
        self.cave = cave_init.copy()
        self.neib = neib
        self.x, self.y = self.get_start_point()
        self.coord_hist = [(self.x, self.y)]
        self.cave[self.x, self.y] = 2
        self.max_jump = max_jump

    def get_start_point(self):
        max_x, max_y = self.cave.shape

        x = round(max_x/2)
        y = round(max_y/2)

        while(not self.cave[x, y]):
            x -= 1
        return np.array([x, y])

    def find(self, max_steps=1000):
        self.ok_points = []
        x = self.x
        y = self.y
        for i in range(max_steps):
            try:
                x, y = self.next_step(np.array([x, y]))
                self.coord_hist.append((x, y))
                self.cave[x][y] += 1
            except NameError:
                return self.cave, self.coord_hist
        return self.cave, self.coord_hist

    def next_step(self, coord):
        x, y = coord

        for move in MOVES:
            x, y = coord + move
            if self.cave[x][y] == 1 and self.neib[x][y] < 7:
                self.ok_points.append((x, y))

        if len(self.ok_points) == 0:
            raise NameError
        for (x_next, y_next) in reversed(self.ok_points):
            if (x - x_next)**2 + (y - y_next)**2 < self.max_jump:
                self.ok_points.remove((x_next, y_next))
                return (x_next, y_next)

        # return self.ok_points.pop()

        # for move in MOVES:
        #    x, y = coord + move
        #    if cave[x][y] == 2:
        #        return coord + move

        # for move in MOVES:
        #    x, y = coord + move
        #    if cave[x][y] == 3:
        #        return coord + move

        raise NameError
