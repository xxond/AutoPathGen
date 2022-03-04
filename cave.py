import joblib
import numpy as np
import plotly.graph_objects as go
from scipy.signal import convolve2d
from tqdm import tqdm

from path import Path

layout = dict(
    autosize=False,
    width=450,
    height=450,
    xaxis_visible=False,
    yaxis_visible=False,
    xaxis_range=[0,400-1],
    yaxis_range=[0,400-1],
    showlegend=False,
    margin=dict(l=0, r=0, b=0, t=0, pad=0)
)

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

    def show(self, save=False, size=400):
        fig = go.Figure()

        length = len(self.smooth_path)
        start = 1.4
        end = 0.01
        width = np.arange(start, end, (end-start)/length)
        paths = self.smooth_path

        multi_size = size/400
        self.size = size

        width *= multi_size

        points = True
        delta = 50
        p_size = (4*multi_size, 90*multi_size)

        self.show_final(fig, paths, width, delta=delta, p_size=p_size, points=points)

        save_layout = layout.copy()
        save_layout.update(dict(width=size, height=size))
        fig.update_layout(**save_layout)
        
        if not save:
            fig.update_layout(**save_layout)
            fig.show()
        else:
            fig.update_layout(**save_layout)

            hash_name = hex(abs(hash(self)))[2:]
            self.dump(hash_name)
            fig.write_image(f'img/{hash_name}.png')

    def show_final(self, fig, paths, widths, delta=50, p_size=(4, 90), points=True,
                   mask=True):
        for path, w in zip(paths, widths):
            fig.add_trace(go.Scatter(x=path[:, 1], y=path[:, 0], mode='lines',
                line=dict(color='rgb(0,0,0)', width=w)))

        if points:
            fig.add_trace(go.Scatter(x=self.smooth_path[0][::delta, 1], 
                                        y=self.smooth_path[0][::delta, 0],
                                        mode='markers',
                                        marker=dict(color='rgb(0,0,0)',
                                                    size=p_size[0])))
        if mask:
            fig.add_trace(go.Scatter(x=[200], y=[200], mode='markers',
                          marker=dict(color='rgb(0.737,0,0.176)',  size=p_size[1])))

    def show_map(self):
        fig = go.Figure(data=go.Heatmap(z=self.map, colorscale='greys'))
        fig.update_layout(**layout)
        fig.update_traces(showscale=False)
        fig.show()

    def show_path(self):
        fig = go.Figure(data=go.Heatmap(z=self.path, colorscale='greys'))
        fig.update_layout(**layout)
        fig.update_traces(showscale=False)
        fig.show()

    def dump(self, name):
        joblib.dump(self, f'caves/{name}.cave')

    def load(self, name):
        self.__dict__.update(joblib.load(f'caves/{name}.cave').__dict__)


def moving_average(a, n=20):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    for i in range(n-1):
        ret[i] = ret[i]/(i + 1)
    ret[n - 1:] = ret[n - 1:] / n
    return ret
