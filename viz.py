from scipy.stats import norm, multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go

from pylab import plot, show, axis, subplot, xlabel, ylabel, grid

# visualization tools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import IPython

def surface(data, theta_vs, theta_ws, z_label=""):
    """ make 3D surface plot of data assuming x and y axis are theta_v and theta_w """
    fig = go.Figure(data=[go.Surface(z=data, x=theta_vs, y=theta_ws)])
    fig.update_layout(
        scene=dict(
            xaxis_title="theta_v",
            yaxis_title="theta_w",
            zaxis_title=z_label,
            xaxis=dict(
                nticks=4,
                range=[0, 2 * np.pi],
            ),
            yaxis=dict(
                nticks=4,
                range=[0, 2 * np.pi],
            ),
            zaxis=dict(nticks=4, range=[-1, 1]),
        ),
        autosize=False,
        margin=dict(l=65, r=50, b=65, t=90),
    )