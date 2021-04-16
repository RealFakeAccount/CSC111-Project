import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Label import Label
from plotly.graph_objs import Scatter, Figure
import plotly.express as px
import graph
from anime import Anime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Animmend"

G = graph.load_from_serialized_data("data/full_graph.json")

upvote_button_style = {'background-color': 'blue',
                      'color': 'white',
                      'width': '47.5%',
                      }

downvote_button_style = {'background-color': 'while',
                        'color': 'black',
                        'width': '47.5%',
                        }

ele = [
    # Title
    html.H1(children='Animmend - An interactive anime recommendation system', style={'textAlign': 'center'}, className="row"),
    html.Div([
        html.Div([
            # Text Input
            html.H5('Text Input'),
            dcc.Dropdown(
                id='name',
                options=[
                {'label': i, 'value': i} for i in G.get_all_anime()],
                value='40meterP: Color of Drops',
                placeholder='40meterP: Color of Drops',
            ),

            # slider
            html.H5(children='Slider of Depth'),
            dcc.Slider(id='depth', value=1, min=1, max=5, step=1, marks={i: str(i) for i in range(1, 6)}),

            # slider
            html.H5(children='Slider of Neighbour'),
            dcc.Slider(id='neighbour', value=1, min=1, max=20, step=1, marks={i: str(i) for i in range(1, 21)}),

            # Button
            html.H5('Related Button'),
            html.Button('upvote', id='upvote', n_clicks=0, style=upvote_button_style),
            html.Button('downvote', id='downvote', n_clicks=0, style=downvote_button_style),

            # Anime description
            dcc.Markdown(id='description title', children="""
            ### Anime Description:
            """),
            dcc.Markdown(id='description', style={'border': 'thin lightgrey solid', 'overflowX': 'scroll', 'height': '500px'})
        ], style={'display': 'inline-block'}, className='four columns'),
        html.Div([
            # Graph
            dcc.Graph(
            id='connection-graph',
            figure=G.draw_graph('40meterP: Color of Drops', 1, 1)
            )
        ], style={'display': 'inline-block'}, className='eight columns')
    ])  
]
app.layout = html.Div(children=ele)

# TODO 
# @app.callback(
#     Output(),
#     Input(b)
# )
# def upvode_downvote():
#     """
#     """

@app.callback(
    Output('connection-graph', 'figure'),
    Input('name', 'value'),
    Input('depth', 'value'),
    Input('neighbour', 'value')
)
def update_graph(name, depth, neighbour) -> tuple[Anime, str]:
    """change the graph and whatever based on the user input
    """
    global G, D
    print(f'{name}, {depth}, {neighbour}')
    return G.draw_graph(name, depth, neighbour)


@app.callback(
    Output('connection-graph', 'clickData'),
    Output('name', 'value'),
    Input('connection-graph', 'clickData'),
    Input('name', 'value')
)
def update_name(clickData, name):
    """change the graph based on the user click input
    """
    print(clickData)
    if clickData is not None and 'hovertext' not in clickData['points'][0]: # deal with edge
        return None, name
    if clickData is not None and 'Similarity Score' not in clickData['points'][0]['hovertext'] and clickData['points'][0]['hovertext'] != name:
        print('name_update')
        name=clickData['points'][0]['hovertext']
    else:
        print('no change')
    return None, name

@app.callback(
    Output('connection-graph', 'hoverData'),
    Output('description', 'children'),
    Input('connection-graph', 'hoverData'),
    Input('description', 'children')
)
def update_description(hoverData, description):
    """change the description based on the user hover input
    """
    print(hoverData)
    if hoverData is not None and 'hovertext' not in hoverData['points'][0]: # deal with edge
        print('Edge End')
        return None, description
    if hoverData is not None and 'Similarity Score' not in hoverData['points'][0]['hovertext']:
        description = '"'+ hoverData['points'][0]['hovertext']+ '" : '+ G.get_anime_description(hoverData['points'][0]['hovertext'])
        print('Successful Read Description')
        return None, description
    elif hoverData is not None and 'Similarity Score' in hoverData['points'][0]['hovertext']:
        return None, 'The point you hovered is Similarity Score.'
    else:
        return None, 'Wait to hover.'

if __name__ == '__main__':
    app.run_server(debug=True)
