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

G = Graph.load_from_serialized_data("data/full_graph.json")

ele = [
    # Title
    html.H3(children="Animmend - An interactive anime recommendation system", style={"textAlign": "center"}),

    # Text Input
    html.Label("Text Input"),
    dcc.Input(id="name", value="Karakai Jouzou no (Moto) Takagi-san Special", type="text"),

    # slider
    html.Label('Slider of Depth'),
    dcc.Slider(id="depth", value=1, min=1, max=5, step=1, marks={i: str(i) for i in range(1, 6)}),

    # slider
    html.Label('Slider of Neighbour'),
    dcc.Slider(id="neighbour", value=1, min=1, max=20, step=1, marks={i: str(i) for i in range(1, 21)}),

    # Graph
    dcc.Graph(
        id="connection-graph",
        figure=G.draw_graph("Karakai Jouzou no (Moto) Takagi-san Special", 1, 1)
        # figure = Figure(data=px.line(x=["a","b","c"], y=[1,3,2], title="sample figure"))
    ),

    # # Anime description
    dcc.Markdown(id="description title", children="""
    ### Anime description:
    """),
    dcc.Markdown(id="description", children=G.get_anime_description("Karakai Jouzou no (Moto) Takagi-san Special"))
]
app.layout = html.Div(children=ele)



@app.callback(
    Output("connection-graph", 'figure'),
    Output("description", "children"),
    Input("name", "value"),
    Input("depth", "value"),
    Input("neighbour", "value")
)
def update_graph(name, depth, neighbour):
    """change the graph and whatever based on the user input
    """
    global G
    print(f"{name}, {depth}, {neighbour}")
    return G.draw_graph(name, depth, neighbour), G.get_anime_description(name)

@app.callback(
    Output("name", 'value'),
    Input('connection-graph', 'clickData')
)
def update_name(clickData):
    return clickData['points'][0]['hovertext']


if __name__ == '__main__':
    app.run_server(debug=True)
