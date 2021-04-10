import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Label import Label
from plotly.graph_objs import Scatter, Figure
import plotly.express as px
import Graph

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

ele = [
    # Title
    html.H3(children="Animmend - An interactive anime recommendation system", style={"textAlign": "center"}),

    # Text Input
    html.Label("Text Input"),
    dcc.Input(id="name", value="Oregairu", type="text"),

    # slider
    html.Label('Slider of Depth'),
    dcc.Slider(id="depth", value=5, min=1, max=5, step=1, marks={i: str(i) for i in range(1, 6)}),

    # Graph
    dcc.Graph(
        id="connection-graph",
        figure=Graph().draw_graph("Oregairu", 1) 
        # figure = Figure(data=px.line(x=["a","b","c"], y=[1,3,2], title="sample figure"))
    ),

    # Anime description
    html.Label("Anime Description"),
    dcc.Markdown(id="description", children = Graph().get_anime_description("Oregairu"))
]
app.layout = html.Div(children=ele)



@app.callback(
    Input("name", "value"),
    Input("depth", "value"),
    Output("connection-graph", "figure"),
    Output("description", "value")
)
def update_graph(name, depth):
    '''change the graph and whatever based on the user input
    '''
    return Graph().draw_graph(name, depth), Graph().get_anime_description(name)


if __name__ == '__main__':
    app.run_server(debug=True)
