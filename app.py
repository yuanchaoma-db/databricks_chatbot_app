import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = html.Div([
    # Top navigation bar
    html.Div([
        html.Div([
            html.Button([
                html.Img(src="assets/menu_icon.svg", className="menu-icon")
            ], id="sidebar-toggle", className="nav-button"),
            html.Button([
                html.Img(src="assets/speech_icon.svg", className="speech-icon")
            ], className="nav-button"),
            html.Div([
                html.Img(src="assets/databricks_icon.svg", className="databricks-logo"),
                html.Img(src="assets/databricks_text.svg", className="databricks-text"),
            ], className="logo-container")
        ], className="nav-left"),
        html.Div([
                html.Div("S", className="user-avatar")
        ], className="nav-right")
    ], className="top-nav"),
    
    # Main content area with sidebar and chat
    html.Div([
        # Sidebar (hidden by default on mobile)
        html.Div([
            html.Div("Recent chats", className="sidebar-header"),
            html.Div([
                html.Div("Kids activities", className="chat-item active"),
                html.Div("Project Brainstorming", className="chat-item"),
                html.Div("Work discussions", className="chat-item"),
                html.Div("Shared with me discussions", className="chat-item"),
                html.Div("Visual languages for data apps", className="chat-item")
            ], className="chat-list")
        ], id="sidebar", className="sidebar"),
        
        # Main chat area
        html.Div([
            # Chat content
            html.Div([
                # Initial welcome message
                html.Div([
                    html.Div("What can I help with?", className="welcome-message"),
                    
                    # Chat input box (initial state)
                    html.Div([
                        dcc.Textarea(
                            id="chat-input",
                            placeholder="Ask anything",
                            className="chat-input",
                            style={"resize": "none"}
                        ),
                        html.Div([
                            html.Button([
                                html.Img(src="assets/at_icon.svg", className="at-icon")
                            ], className="input-button"),
                            html.Button([
                                html.Img(src="assets/clip_icon.svg", className="clip-icon")
                            ], className="input-button")
                        ], className="input-buttons-left"),
                        html.Div([
                            html.Button([
                                html.Img(src="assets/send_icon.svg", className="send-icon")
                            ], id="send-button", className="input-button")
                        ], className="input-buttons-right")
                    ], className="chat-input-container"),
                    
                    # Suggestion buttons
                    html.Div([
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Span("Find tables to query")
                        ], className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Span("Debug my notebook")
                        ], className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Span("Fix my code")
                        ], className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Span("What is Unity Catalog?")
                        ], className="suggestion-button")
                    ], className="suggestion-buttons"),
                    
                    html.Div("Chatbot may make mistakes. Check important info.", className="disclaimer")
                ], id="welcome-container", className="welcome-container"),
                
                # Chat messages will be added here
                html.Div(id="chat-messages", className="chat-messages")
            ], className="chat-content"),
            
            # Fixed chat input at bottom (for after initial message is sent)
            html.Div([
                dcc.Textarea(
                    id="chat-input-fixed",
                    placeholder="Ask anything",
                    className="chat-input",
                    style={"resize": "none"}
                ),
                html.Div([
                    html.Button([
                        html.Div(className="at-icon")
                    ], className="input-button"),
                    html.Button([
                        html.Div(className="clip-icon")
                    ], className="input-button")
                ], className="input-buttons-left"),
                html.Div([
                    html.Button([
                        html.Div(className="send-icon")
                    ], id="send-button-fixed", className="input-button")
                ], className="input-buttons-right")
            ], id="fixed-input-container", className="fixed-input-container")
        ], className="chat-container")
    ], className="main-content")
])

# Store chat history
chat_history = []

# Callback for sending messages
@app.callback(
    [Output("chat-messages", "children"),
     Output("welcome-container", "style"),
     Output("fixed-input-container", "style"),
     Output("chat-input", "value"),
     Output("chat-input-fixed", "value")],
    [Input("send-button", "n_clicks"),
     Input("send-button-fixed", "n_clicks")],
    [State("chat-input", "value"),
     State("chat-input-fixed", "value"),
     State("chat-messages", "children")]
)
def send_message(n_clicks1, n_clicks2, input1, input2, current_messages):
    # Determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        # Initial load, don't do anything
        return current_messages, {"display": "block"}, {"display": "none"}, "", ""
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Get the appropriate input value
    user_input = input1 if trigger_id == "send-button" else input2
    
    if not user_input:
        # Empty input, don't do anything
        return current_messages, {"display": "block" if not current_messages else "none"},\
              {"display": "none" if not current_messages else "flex"}, "", ""
    
    # Create user message
    user_message = html.Div([
        html.Div(user_input, className="message-text")
    ], className="user-message message")
    
    # Create bot response (this would be where you'd integrate with an actual AI)
    if "weather" in user_input.lower() and "paris" in user_input.lower():
        bot_response = html.Div([
            html.Div("Thinking...", className="thinking-indicator"),
            html.Div([
                html.Div("Currently, the weather in Paris is clear with a temperature of 36°F, feeling like 30°F due to the wind. The forecast indicates it will reach a high of 42°F and a low of 35°F. The wind is coming from the northeast at 7 mph, and the humidity stands at 87%.", 
                         className="message-text"),
                html.Div([
                    html.Button("Sources", className="sources-button"),
                    html.Div([
                        html.Button(className="copy-button"),
                        html.Button(className="refresh-button"),
                        html.Button(className="thumbs-up-button"),
                        html.Button(className="thumbs-down-button")
                    ], className="message-actions")
                ], className="message-footer")
            ], className="message-content")
        ], className="bot-message message")
    elif "kid" in user_input.lower() and "activit" in user_input.lower():
        bot_response = html.Div([
            html.Div("Thinking...", className="thinking-indicator"),
            html.Div([
                html.Div([
                    html.P("Here are a variety of fun and educational activities you can enjoy with kids of all ages."),
                    html.Ol([
                        html.Li([
                            html.Strong("Crafting: "),
                            "Kids can engage in making scratch art, bookmarks, or even transform old crayons into new art pieces. Crafting activities like making a bird feeder or superhero costumes are also popular."
                        ]),
                        html.Li([
                            html.Strong("Educational Games: "),
                            "Try learning-oriented games like building a cardboard castle or conducting simple science experiments like making a bouncy egg or exploring how germs spread."
                        ]),
                        html.Li([
                            html.Strong("Creative Play: "),
                            "Set up activities like indoor hopscotch or a mini obstacle course. You can also organize a treasure hunt or put on a puppet show to spark creativity."
                        ]),
                        html.Li([
                            html.Strong("Cooking and Baking: "),
                            "Involve kids in making simple recipes like ice cream in a bag or baking muffins, which are not only fun but also skill-building."
                        ])
                    ])
                ], className="message-text"),
                html.Div([
                    html.Button("Sources", className="sources-button"),
                    html.Div([
                        html.Button(className="copy-button"),
                        html.Button(className="refresh-button"),
                        html.Button(className="thumbs-up-button"),
                        html.Button(className="thumbs-down-button")
                    ], className="message-actions")
                ], className="message-footer"),
                html.Div([
                    html.Div([
                        html.Div("source name", className="source-name"),
                        html.Div("Metadata", className="source-metadata")
                    ], className="source-item"),
                    html.Div([
                        html.Div("source name", className="source-name"),
                        html.Div("Metadata", className="source-metadata")
                    ], className="source-item"),
                    html.Div([
                        html.Div("source name", className="source-name"),
                        html.Div("Metadata", className="source-metadata")
                    ], className="source-item"),
                    html.Div([
                        html.Div("SQL", className="source-name"),
                        html.Div("Metadata", className="source-metadata")
                    ], className="source-item")
                ], className="sources-list")
            ], className="message-content")
        ], className="bot-message message")
    else:
        bot_response = html.Div([
            html.Div("Thinking...", className="thinking-indicator"),
            html.Div([
                html.Div(f"I understand you're asking about: '{user_input}'. How can I help you with that?", 
                         className="message-text"),
                html.Div([
                    html.Button("Sources", className="sources-button"),
                    html.Div([
                        html.Button(className="copy-button"),
                        html.Button(className="refresh-button"),
                        html.Button(className="thumbs-up-button"),
                        html.Button(className="thumbs-down-button")
                    ], className="message-actions")
                ], className="message-footer")
            ], className="message-content")
        ], className="bot-message message")
    
    # Update chat history
    if current_messages:
        updated_messages = current_messages + [user_message, bot_response]
    else:
        updated_messages = [user_message, bot_response]
    
    # Hide welcome container, show fixed input
    return updated_messages, {"display": "none"}, {"display": "flex"}, "", ""

# Toggle sidebar
@app.callback(
    Output("sidebar", "className"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar", "className")
)
def toggle_sidebar(n_clicks, current_class):
    if n_clicks:
        if "sidebar-open" in current_class:
            return "sidebar"
        else:
            return "sidebar sidebar-open"
    return current_class

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
