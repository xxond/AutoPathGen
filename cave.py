import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
from tqdm import tqdm

from path import Path


class Cave:
    def __init__(
            self,
            mask,
            size=400,
            prob=0.515,
            steps=100,
            death=[(0, 0), (4, 20), (5, 50), (4, 60)],
            birth=[(5, 0), (4, 40), (5, 55), (4, 90)],
            min_len=1000,
            max_steps=10000,
            max_jump=20,
            smooth_coef=[5],
            beg=5,
            delta=0.005,
            smooth_width=1,
            smooth_color='black',
            point_size=10,
            interval=250,
            debug=False):

        assert mask.shape == (size, size), 'Wrong dimention of mask'

        self.mask = mask

        self.size = size
        self.steps = steps

        self.death = death
        self.birth = birth

        self.min_len = max([min_len, smooth_coef[-1]])
        self.max_steps = max_steps
        self.max_jump = max_jump

        self.smooth_coef = smooth_coef

        self.beg = beg
        self.delta = delta
        self.smooth_width = smooth_width
        self.smooth_color = smooth_color
        self.point_size = point_size
        self.interval = interval

        if debug:
            self.map = np.random.choice(a=[True, False], size=(size, size),
                                        p=[prob, 1-prob])

            self.simulate()
        else:
            le = 0
            for _ in tqdm(range(10**5), total=False, desc='Attempt'):
                self.map = np.random.choice(a=[True, False], size=(size, size),
                                            p=[prob, 1-prob])

                self.simulate()
                le = self.find_path()
                if le > self.min_len:
                    break
            self.smoother_path()

    def simulate(self):
        self.map_hist = []
        for i in range(self.steps):
            n_map = self.neighbor_map()
            new_map = self.map.copy()

            death_th = next(x[0] for x in reversed(self.death) if x[1] <= i)
            birth_th = next(x[0] for x in reversed(self.birth) if x[1] <= i)

            dead = np.logical_not(self.map)
            new_map[np.logical_and(n_map < death_th, self.map)] = False
            new_map[np.logical_and(n_map > birth_th, dead)] = True
            new_map[self.mask] = False

            self.map = new_map
            self.map_hist.append(self.map)
        self.map.dtype = np.int8

    def neighbor_map(self):
        mask = np.zeros((self.map.shape[0] + 2, self.map.shape[1] + 2))
        mask[1:-1, 1:-1] = self.map
        conv = np.ones((3, 3))
        conv[1, 1] = 0
        return convolve2d(mask, conv, mode='valid')

    def find_path(self):
        path = Path(self.map, self.neighbor_map(), max_jump=self.max_jump)
        self.path, hist = path.find(max_steps=self.max_steps)

        counter = self.beg
        self.path_hist = []
        for x, y in hist:
            if (x, y) in self.path_hist:
                continue
            else:
                self.path_hist.append((x, y))

            self.path[x][y] = counter
            counter += self.delta

        return len(self.path_hist)

    def smoother_path(self):
        self.smooth_path = []
        for n in self.smooth_coef:
            y = moving_average(np.array(self.path_hist, dtype=np.double)[:, 1], n=n)
            x = moving_average(np.array(self.path_hist, dtype=np.double)[:, 0], n=n)
            self.smooth_path.append(np.array([x, y]).transpose())

    def animation(self, size=(6, 6)):
        def anim_step(i):
            line.set_data(self.map_hist[i])
            ax.set_title(f'{i}/{self.steps}', fontsize=10)
            ax.set_axis_off()
            return line

        fig, ax = plt.subplots(figsize=size)
        line = plt.imshow(self.map_hist[0], cmap='magma')

        return FuncAnimation(fig, anim_step,
                             frames=self.steps, interval=self.interval)

    def show(self, size=(8, 8)):
        fig, ax = plt.subplots(figsize=size)

        # ax[0].imshow(self.path, cmap='magma') #  jet
        # ax[1].imshow(self.map, cmap='magma')
        # ax[2].imshow(self.hood, cmap='magma')
        # ax[3].imshow(self.mono, cmap='gray')

        length = len(self.smooth_path)
        start = 1.4
        end = 0.01
        width = np.arange(start, end, (end-start)/length)
        paths = self.smooth_path
        # ax_counter = 0
        # axs = ax.flatten()

        points = True
        reverse = False
        delta = 50
        p_size = 7500

        self.show_final(ax, paths, width, delta=delta,
                        p_size=p_size, points=points, reverse=reverse)

        # for a in axs:
        #    a.set_axis_off()
        ax.set_axis_off()

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'img/{hash(str(self.path_hist))}.png', dpi=300)
        fig.show()

    def show_final(self, ax, paths, widths, delta=50, p_size=5000, name='-',
                   points=True, mask=True, reverse=True, title=False):
        if reverse:
            for path, w in list(zip(paths, reversed(widths))):
                ax.plot(*reversed((path).transpose()),
                        linewidth=w,
                        color=self.smooth_color)
            if points:
                ax.scatter(*reversed(paths[-1][::delta].transpose()),
                           s=self.point_size,
                           color=self.smooth_color)
        else:
            for path, w in list(zip(paths, widths)):
                ax.plot(*reversed(path.transpose()),
                        linewidth=w,
                        color=self.smooth_color)
            if points:
                ax.scatter(*reversed(paths[0][::delta].transpose()),
                           s=self.point_size,
                           color=self.smooth_color)

        if mask:
            ax.scatter([self.size/2], [self.size/2], s=p_size, color='black')
        if title:
            ax.set_title(f'D: {delta}, Ps:{points}, R:{reverse}, ' +
                         f'W:{np.around(widths, 2)}\nname: {name}', fontsize=10)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlim([0, self.size])
        ax.set_ylim([self.size, 0])

    def show_map(self, size=(4, 4)):
        _, ax = plt.subplots(figsize=size)
        ax.imshow(self.map, cmap='magma')
        plt.show()

    def show_path(self, size=(4, 4)):
        _, ax = plt.subplots(figsize=size)
        ax.imshow(self.path, cmap='magma')
        plt.show()


def moving_average(a, n=20):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    for i in range(n-1):
        ret[i] = ret[i]/(i + 1)
    ret[n - 1:] = ret[n - 1:] / n
    return ret
