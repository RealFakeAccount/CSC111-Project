import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Label import Label
from plotly.graph_objs import Scatter, Figure
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def graph_figure() -> Figure():
    """create a figure object based on given variables. 
    See a3_visualization.py for examples.
    The following code is just a example and (TODO) SHOULD be modified 
    """
    fig = px.line(x=["a","b","c"], y=[1,3,2], title="sample figure")
    return fig

ele = [
    # Title
    html.H3(children="Animmend - An interactive anime recommendation system", style={"textAlign": "center"}),

    # Text Input
    html.Label("Text Input"),
    dcc.Input(id = "name", value = "Oregairu", type = "text"),

    # slider
    html.Label('Slider'),
    dcc.Slider(min = 1, max = 5, marks = [str(i) for i in range(1, 6)], value = 5), 

    # Graph
    dcc.Graph(
        id = "connection-graph",
        Figure = graph_figure()
    )

    # Anime description
    html.Label()
]
app.layout = html.Div(children=ele)



@app.callback(
    Input(component_id="name", component_property="value")
)
def update_using_input_name(value):
    '''TODO: change the graph and whatever based on the user input

    '''
    ...


if __name__ == '__main__':
    app.run_server(debug=True)