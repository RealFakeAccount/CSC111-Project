"""CSC111 Winter 2021 Final Project
This file is provided for whichever TA is grading and giving us 100.
Forms of distribution of this code are allowed.
For more information on copyright for CSC111 materials, please consult the Course Syllabus.

Copyright (c) 2021 by Ching Chang, Letian Cheng, Arkaprava Choudhury, Hanrui Fan
"""
import json
import os
from typing import Optional, Any
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from anime import Anime
import graph


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Animmend'
feedback = {}  # {('anime1', 'anime2'): (upvote, downvote)}

full_graph = graph.load_from_serialized_data('data/full_graph.json')

upvote_button_style = {'background-color': 'white', 'color': 'black', 'width': '47.5%'}

downvote_button_style = {'background-color': 'while', 'color': 'black', 'width': '47.5%'}

ele = [
    # Title
    html.H1(children='Animmend - An interactive anime recommendation system',
            style={'textAlign': 'center'}, className='row'),
    html.Div([
        html.Div([
            # Text Input
            html.H5('Search Anime'),
            dcc.Dropdown(
                id='name',
                options=[{'label': i, 'value': i} for i in full_graph.get_all_anime()],
                value='40meterP: Color of Drops',
                placeholder='40meterP: Color of Drops',
                clearable=False
            ),

            # slider
            html.H5(children='Depth'),
            dcc.Slider(id='depth', value=2, min=1, max=5, step=1,
                       marks={i: str(i) for i in range(1, 6)}),

            # slider
            html.H5(children='Number of Neighbours'),
            dcc.Slider(id='neighbour', value=3, min=1, max=20, step=1,
                       marks={i: str(i) for i in range(1, 21)}),

            # Button
            html.H5('Rate this recommendation (not the anime)'),
            html.Button('good', id='upvote', n_clicks=0, style=upvote_button_style),
            html.Button('bad', id='downvote', n_clicks=0, style=downvote_button_style),

            # Button text
            dcc.Markdown(id='button feedback', children=""),

            # Anime description
            html.Div([
                html.Div([
                    html.H3(id='description picture', children='40meterP: Color of Drops'),
                    html.Img(id='thumbnail', style={'height': '100%', 'width': '100%'})
                ], className='six columns'),
                html.Div([
                    html.H3(id='description title', children='Anime Description:'),
                    dcc.Markdown(id='description',
                                 style={'border': 'thin lightgrey solid', 'overflowX': 'scroll',
                                        'height': '500px'})
                ], className='six columns'),
            ])

        ], style={'display': 'inline-block'}, className='four columns'),
        html.Div([
            # Graph
            dcc.Graph(id='connection-graph',
                      figure=full_graph.draw_graph('40meterP: Color of Drops', 1, 1))
        ], style={'display': 'inline-block'}, className='eight columns')
    ])
]
app.layout = html.Div(children=ele)


@app.callback(
    Output('button feedback', 'children'),
    Input('upvote', 'n_clicks'),
    Input('downvote', 'n_clicks'),
)
def upvote_downvote(upvote_times: int, downvote_times: int) -> str:
    """Send the feedback to graph object whenever we receive one
    """
    hover, core = read_hover('.')
    if hover is not None:
        edge = (hover, core) if hover < core else (core, hover)
        prev = feedback.get(edge, (0, 0))
        if abs(upvote_times - prev[0]) + abs(downvote_times - prev[1]) == 0:
            return ''

        action = 'upvote' if abs(upvote_times - prev[0]) != 0 else 'downvote'
        feedback[edge] = (prev[0] + 1, prev[1]) if action == 'upvote' else (prev[0], prev[1] + 1)

        full_graph.store_feedback(action, core, hover)
        full_graph.dump_feedback_to_file('data/feedback.json')

        return f'{action} to {hover}'
    else:
        return ''


@app.callback(
    Output('connection-graph', 'figure'),
    Input('name', 'value'),
    Input('depth', 'value'),
    Input('neighbour', 'value')
)
def update_graph(name: str, depth: int, neighbour: int) -> tuple[Anime, str]:
    """Update the user's input to be the centre of the graph whenever we receive a user input
    """
    global full_graph
    return full_graph.draw_graph(name, depth, neighbour)


@app.callback(
    Output('connection-graph', 'clickData'),
    Output('name', 'value'),
    Input('connection-graph', 'clickData'),
    Input('name', 'value')
)
def update_name(click_data: dict[str, Any], name: str) -> tuple[None, str]:
    """change the graph based on the user click input
    """
    hover, core = '', '40meterP: Color of Drops'
    if click_data is not None and 'hovertext' not in click_data['points'][0]:  # deal with edge
        return None, name
    if click_data is not None and 'Similarity Score' not in click_data['points'][0]['hovertext'] \
            and click_data['points'][0]['hovertext'] != name:
        name = click_data['points'][0]['hovertext']
    core = name
    write_hover('.', hover, core)
    return None, name


@app.callback(
    Output('connection-graph', 'hoverData'),
    Output('description', 'children'),
    Output('thumbnail', 'src'),
    Output('description picture', 'children'),
    Input('connection-graph', 'hoverData'),
    Input('description', 'children'),
    Input('thumbnail', 'src'),
    Input('description picture', 'children')
)
def update_description(hover_data: dict[str, Any], description: str, thumbnail: str,
                       pic_title: str) -> tuple[None, str, Optional[str], str]:
    """change the description based on the user hover input
    """
    hover, core = '', '40meterP: Color of Drops'
    if hover_data is None:
        return None, 'Waiting for hover...', None, 'Waiting for hover...'

    if 'hovertext' not in hover_data['points'][0]:  # deal with edge
        hover = None
        write_hover('.', hover, core)
        return None, description, thumbnail, pic_title

    anime_title = hover_data['points'][0]['hovertext']
    if 'Similarity Score' not in anime_title:
        description = full_graph.get_anime_description(anime_title)
        hover = anime_title
        write_hover('.', hover, core)
        thumbnail = full_graph.get_anime_thumbnail_url(anime_title)
        pic_title = anime_title
        return None, description, thumbnail, pic_title
    elif 'Similarity Score' in anime_title:
        hover = None
        write_hover('.', hover, core)
        return None, 'The point you hovered is Similarity Score.', thumbnail, pic_title
    else:
        return None, '', None, ''


def run_test_server() -> None:
    """Start the test server"""
    global app
    app.run_server(debug=True)


def write_hover(data_folder: str, hover: str, core: str) -> None:
    """Let's get things complicated by not using global variables :)
    This function save the global variable into a file:
    hover -- the user's hover node;
    core -- the current center of the graph
    """
    path_str = data_folder + '/_tmp'
    with open(path_str, 'w') as new_file:
        json.dump({'hover': hover, 'core': core}, new_file)


def read_hover(data_folder: str) -> tuple[str, str]:
    """Let's get things complicated by not using global variables :)
    This function read the global variable into a file:
    hover -- the user's hover node;
    core -- the current center of the graph
    """
    hover, core = '', '40meterP: Color of Drops'
    path_str = data_folder + '/_tmp'
    if not os.path.exists(path_str):
        return hover, core
    with open(path_str, 'r') as json_file:
        data = json.load(json_file)
        hover, core = data['hover'], data['core']
    return hover, core


if __name__ == '__main__':
    # app.run_server(debug=True)

    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E9999', 'E9998', 'E1136'],
        'max-nested-blocks': 4
    })

    import python_ta.contracts
    python_ta.contracts.check_all_contracts()
