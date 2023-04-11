import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import sys
sys.path.append("/home/nick/Regression/src")
from deep_learning import nnet


def histogram(df, x, bins=None, color=None, title=None, font_size=None):
    fig = px.histogram(df, x=x, nbins=bins, color=color, title=title, marginal="box")
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def parity(df, predict, actual, color=None, title=None, font_size=None):
    fig = px.scatter(df, x=actual, y=predict, color=color, title=title)
    fig.add_trace(go.Scatter(x=df[actual], y=df[actual], mode="lines", showlegend=False, name="Actual"))
    fig.update_layout(font=dict(size=font_size))
    plot(fig)


data = pd.read_csv("/home/nick/Regression/test/LungCap.csv")

# histogram(data, x="LungCap", bins=9, font_size=16)

model = nnet(
    df=data, 
    outputs=["LungCap"], 
    test_frac=0.5,
    deep=False,
)

parity(
    df=model.predictions[model.outputs[0]],
    predict="Predicted",
    actual="Actual",
    font_size=16,
)

print(model.metric[model.outputs[0]])
