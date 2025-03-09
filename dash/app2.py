from dash import Dash, html, dcc, Input, Output, ctx, callback, DiskcacheManager
import time
import diskcache

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = Dash(__name__, background_callback_manager=background_callback_manager)

app.layout = html.Div([
    dcc.Input(id="input-num", type="number", value=2),
    html.Button("Start Task", id="btn"),
    html.Div(id="output"),
])

@callback(
    Output("output", "children"),
    Input("btn", "n_clicks"),
    Input("input-num", "value"),
    background=True,
    manager=background_callback_manager,
    prevent_initial_call=True
)
def long_task(n_clicks, value):
    time.sleep(value)  # Simulate long processing
    return f"Task completed in {value} seconds"

if __name__ == "__main__":
    app.run_server(debug=True)
