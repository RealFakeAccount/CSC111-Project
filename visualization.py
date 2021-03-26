import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def generate_title() -> None:
    return html.H1

def generate_layout() -> None:
    ele = list()
    ele.append()
    app.layout = html.Div()

if __name__ == 'main':
    app.run_server(debug=True)