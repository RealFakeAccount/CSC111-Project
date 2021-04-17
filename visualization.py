"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials, please consult the Course Syllabus.

Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import graph
from anime import Anime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Animmend'
feedback = {}  # {('anime1', 'anime2'): (upvote, downvote)}
core, hover = '40meterP: Color of Drops', None  # the current graph shell and current hover anime

G = graph.load_from_serialized_data('data/full_graph.json')

upvote_button_style = {'background-color': 'blue', 'color': 'white', 'width': '47.5%'}

downvote_button_style = {'background-color': 'while', 'color': 'black', 'width': '47.5%'}

ele = [
    # Title
    html.H1(children='Animmend - An interactive anime recommendation system',
            style={'textAlign': 'center'}, className='row'),
    html.Div([
        html.Div([
            # Text Input
            html.H5('Text Input'),
            dcc.Dropdown(
                id='name',
                options=[{'label': i, 'value': i} for i in G.get_all_anime()],
                value='40meterP: Color of Drops',
                placeholder='40meterP: Color of Drops',
            ),

            # slider
            html.H5(children='Slider of Depth'),
            dcc.Slider(id='depth', value=2, min=1, max=5, step=1,
                       marks={i: str(i) for i in range(1, 6)}),

            # slider
            html.H5(children='Slider of Neighbour'),
            dcc.Slider(id='neighbour', value=3, min=1, max=20, step=1,
                       marks={i: str(i) for i in range(1, 21)}),

            # Button
            html.H5('Related Button'),
            html.Button('upvote', id='upvote', n_clicks=0, style=upvote_button_style),
            html.Button('downvote', id='downvote', n_clicks=0, style=downvote_button_style),

            # Button text
            dcc.Markdown(id='button feedback', children=""),

            # Anime description
            html.Div([
                html.Div([
                    dcc.Markdown(id='description title', children="""
                    ### Anime Description:
                    """),
                    dcc.Markdown(id='description',
                                 style={'border': 'thin lightgrey solid', 'overflowX': 'scroll',
                                        'height': '500px'})
                ], className='six columns'),
                html.Div([
                    dcc.Markdown(id='description picture', children="""
                    ### Anime Picture:
                    """),
                    html.Img(id='thumbnail')
                ], className='six columns')
            ])

        ], style={'display': 'inline-block'}, className='four columns'),
        html.Div([
            # Graph
            dcc.Graph(id='connection-graph', figure=G.draw_graph('40meterP: Color of Drops', 1, 1))
        ], style={'display': 'inline-block'}, className='eight columns')
    ])
]
app.layout = html.Div(children=ele)


@app.callback(
    Output('button feedback', 'children'),
    Input('upvote', 'n_clicks'),
    Input('downvote', 'n_clicks'),
)
def upvote_downvote(upvote_times: int, downvote_times: int):
    """ this function give feedback to graph object
    """
    global hover, core
    if hover is not None:
        edge = (hover, core) if hover < core else (core, hover)
        prev = feedback.get(edge, (0, 0))
        if abs(upvote_times - prev[0]) + abs(downvote_times - prev[1]) == 0:
            return ""

        action = 'upvote' if abs(upvote_times - prev[0]) != 0 else 'downvote'
        feedback[edge] = (prev[0] + 1, prev[1]) if action == 'upvote' else (prev[0], prev[1] + 1)

        G.store_feedback(action, G.get_anime(core), G.get_anime(hover))
        G.dump_feedback_to_file('data/feedback.json')

        return f'{action} to {hover}'


@app.callback(
    Output('connection-graph', 'figure'),
    Input('name', 'value'),
    Input('depth', 'value'),
    Input('neighbour', 'value')
)
def update_graph(name, depth, neighbour) -> tuple[Anime, str]:
    """change the graph and whatever based on the user input
    """
    global G
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
    global hover, core
    if clickData is not None and 'hovertext' not in clickData['points'][0]:  # deal with edge
        return None, name
    if clickData is not None and 'Similarity Score' not in clickData['points'][0]['hovertext'] and \
            clickData['points'][0]['hovertext'] != name:
        name = clickData['points'][0]['hovertext']
    core = name
    return None, name


@app.callback(
    Output('connection-graph', 'hoverData'),
    Output('description', 'children'),
    Output('thumbnail', 'src'),
    Input('connection-graph', 'hoverData'),
    Input('description', 'children')
)
def update_description(hoverData, description):
    """change the description based on the user hover input
    """
    global hover, core
    if hoverData is None:
        return None, 'Wait to hover.', None

    if 'hovertext' not in hoverData['points'][0]:  # deal with edge
        hover = None
        return None, description, None

    anime_title = hoverData['points'][0]['hovertext']
    if 'Similarity Score' not in anime_title:
        description = '"' + anime_title + '" : ' + G.get_anime_description(anime_title)
        hover = anime_title
        thumbnail = G.get_anime_thumbnail_url(anime_title)
        return None, description, thumbnail
    elif 'Similarity Score' in anime_title:
        hover = None
        return None, 'The point you hovered is Similarity Score.', None


def run_test_server() -> None:
    """Start the test server"""
    global app
    app.run_server(debug=True)


if __name__ == '__main__':
    # app.run_server(debug=True)

    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E9999', 'E9998'],
        'max-nested-blocks': 4
    })
