import dash
from dash import html, dcc, Input, Output, State, callback, ALL, callback_context, no_update
import dash_bootstrap_components as dbc
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

SERVING_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"  # Replace with your actual endpoint name

# Initialize Databricks workspace client
client = WorkspaceClient()

# Define the layout
app.layout = html.Div([
    # Top navigation bar
    html.Div([
        # Left component containing both nav-left and sidebar
        html.Div([
            # Nav left
            html.Div([
                html.Button([
                    html.Img(src="assets/menu_icon.svg", className="menu-icon")
                ], id="sidebar-toggle", className="nav-button"),
                # Speech icon in top nav (visible when sidebar is closed)
                html.Button([
                    html.Img(src="assets/speech_icon.svg", className="speech-icon")
                ], id="speech-button", className="nav-button"),
                html.Button([
                    html.Img(src="assets/speech_icon.svg", className="speech-icon"),
                    html.Div("New chat", className="new-chat-text")
                ], id="sidebar-new-chat-button", className="new-chat-button", style={"display": "none"})
            ], id="nav-left", className="nav-left"),
            
            # Sidebar (now inside the left component)
            html.Div([
                html.Div([
                    html.Div("Recent chats", className="sidebar-header-text"),
                ], className="sidebar-header"),
                html.Div([
                    html.Div("Kids activities", className="chat-item active", id={"type": "chat-item", "index": 0}),
                    html.Div("Project Brainstorming", className="chat-item", id={"type": "chat-item", "index": 1}),
                    html.Div("Work discussions", className="chat-item", id={"type": "chat-item", "index": 2}),
                    html.Div("Shared with me discussions", className="chat-item", id={"type": "chat-item", "index": 3}),
                    html.Div("Visual languages for data apps Visual languages for data apps Visual languages for data apps Visual languages for data apps", className="chat-item", id={"type": "chat-item", "index": 4})
                ], className="chat-list", id="chat-list")
            ], id="sidebar", className="sidebar")
        ], id="left-component", className="left-component"),
        
        html.Div([
            html.Div([
                html.Img(src="assets/databricks_icon.svg", className="databricks-logo"),
                html.Img(src="assets/databricks_text.svg", className="databricks-text"),
            ], id="logo-container", className="logo-container")
        ], className="nav-center"),
        html.Div([
            html.Div("S", className="user-avatar")
        ], className="nav-right")
    ], className="top-nav"),
    
    # Main content area with chat
    html.Div([
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
                            html.Button(
                                className="input-button at-button"
                            ),
                            html.Button(
                                className="input-button clip-button"
                            )
                        ], className="input-buttons-left"),
                        html.Div([
                            html.Button(
                                id="send-button", 
                                className="input-button send-button"
                            )
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
                html.Div(id="chat-messages", className="chat-messages"),
                
            ], className="chat-content"),
            
            # Fixed chat input at bottom (for after initial message is sent)
            html.Div([
                html.Div([
                    dcc.Textarea(
                        id="chat-input-fixed",
                        placeholder="Ask anything",
                        className="chat-input",
                        style={"resize": "none"}
                    ),
                    html.Div([
                        html.Button(
                            className="input-button at-button"
                        ),
                        html.Button(
                            className="input-button clip-button"
                        )
                    ], className="input-buttons-left"),
                    html.Div([
                        html.Button(
                            id="send-button-fixed", 
                            className="input-button send-button"
                        )
                    ], className="input-buttons-right")
                ], id="fixed-input-container", className="fixed-input-container"),
                
                html.Div("Chatbot may make mistakes. Check important info.", className="disclaimer-fixed")
            ], id="fixed-input-wrapper", className="fixed-input-wrapper", style={"display": "none"})
        ], className="chat-container"),
        
    ], id="main-content", className="main-content"),
    
    # Add dcc.Store components to manage state
    dcc.Store(id="chat-trigger", data={"trigger": False, "message": ""}),
    dcc.Store(id="chat-history-store", data=[])
])

# Store chat history
chat_history = []

# First callback: Add user message and thinking indicator
@app.callback(
    [Output("chat-messages", "children"),
     Output("welcome-container", "style"),
     Output("fixed-input-wrapper", "style"),
     Output("chat-input", "value"),
     Output("chat-input-fixed", "value"),
     Output("chat-trigger", "data")],
    [Input("send-button", "n_clicks"),
     Input("send-button-fixed", "n_clicks")],
    [State("chat-input", "value"),
     State("chat-input-fixed", "value"),
     State("chat-messages", "children")],
    prevent_initial_call=True
)
def add_user_message_and_indicator(n_clicks1, n_clicks2, input1, input2, current_messages):
    # Determine which input triggered the callback
    ctx = callback_context
    if not ctx.triggered:
        # Initial load, don't do anything
        return current_messages, {"display": "block"}, {"display": "none"}, "", "", {"trigger": False, "message": ""}
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Get the appropriate input value
    user_input = input1 if trigger_id == "send-button" else input2
    
    if not user_input:
        # Empty input, don't do anything
        return current_messages, {"display": "block" if not current_messages else "none"},\
              {"display": "none" if not current_messages else "flex"}, "", "", {"trigger": False, "message": ""}
    
    # Create user message
    user_message = html.Div([
        html.Div(user_input, className="message-text")
    ], className="user-message message")
    
    # Add the user message to the chat
    if current_messages:
        updated_messages = current_messages + [user_message]
    else:
        updated_messages = [user_message]
    
    # Add thinking indicator
    thinking_indicator = html.Div([
        html.Div([
            html.Span(className="spinner"),
            html.Span("Thinking...")
        ], className="thinking-indicator")
    ], className="bot-message message")
    
    # Add the thinking indicator to the messages
    updated_messages.append(thinking_indicator)
    
    # Set the trigger for the next callback
    trigger_data = {"trigger": True, "message": user_input}
    
    # Hide welcome container, show fixed input
    return updated_messages, {"display": "none"}, {"display": "flex"}, "", "", trigger_data

# Second callback: Call model API and update with response
@app.callback(
    Output("chat-messages", "children", allow_duplicate=True),
    [Input("chat-trigger", "data")],
    [State("chat-messages", "children")],
    prevent_initial_call=True
)
def get_model_response(trigger_data, current_messages):
    if not trigger_data["trigger"]:
        # No trigger, don't do anything
        return dash.no_update
    
    user_input = trigger_data["message"]
    
    try:
        # Make the API call
        response = client.serving_endpoints.query(
            SERVING_ENDPOINT_NAME,
            temperature=0.7,
            messages=[ChatMessage(content=user_input, role=ChatMessageRole.USER)],
        )
        
        bot_response_text = response.choices[0].message.content
        
        formatted_response = bot_response_text
        
        bot_response = html.Div([
            # Add model name at the top of the message
            html.Div([
                html.Div(className="model-icon"),
                html.Span(SERVING_ENDPOINT_NAME, className="model-name")
            ], className="model-info"),
            
            html.Div([
                dcc.Markdown(formatted_response, className="message-text"),
                html.Div([
                    html.Div([
                        html.Button("Sources", className="sources-button")
                    ], className="sources-row"),
                    html.Div([
                        html.Button(className="copy-button"),
                        html.Button(className="refresh-button"),
                        html.Button(className="thumbs-up-button"),
                        html.Button(className="thumbs-down-button")
                    ], className="message-actions")
                ], className="message-footer")
            ], className="message-content")
        ], className="bot-message message")
        
        # Replace the thinking indicator with the actual response
        updated_messages = current_messages[:-1] + [bot_response]
        
    except Exception as e:
        # Handle errors in AI service communication
        error_response = html.Div([
            html.Div([
                html.Div(f"Sorry, I encountered an error: {str(e)}", 
                         className="message-text"),
                html.Div([
                    html.Div([
                        html.Button(className="copy-button"),
                        html.Button(className="refresh-button"),
                        html.Button(className="thumbs-up-button"),
                        html.Button(className="thumbs-down-button")
                    ], className="message-actions")
                ], className="message-footer")
            ], className="message-content")
        ], className="bot-message message")
        
        # Replace the thinking indicator with the error response
        updated_messages = current_messages[:-1] + [error_response]
    
    # Return just the updated_messages, not wrapped in another list
    return updated_messages

# Toggle sidebar and speech button
@app.callback(
    [Output("sidebar", "className"),
     Output("speech-button", "style"),
     Output("sidebar-new-chat-button", "style"),
     Output("logo-container", "className"),
     Output("nav-left", "className"),
     Output("left-component", "className"),
     Output("main-content", "className")],
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className"),
     State("left-component", "className"),
     State("main-content", "className")]
)
def toggle_sidebar(n_clicks, current_sidebar_class, current_left_component_class, current_main_content_class):
    if n_clicks:
        if "sidebar-open" in current_sidebar_class:
            # Sidebar is closing
            return "sidebar", {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", "main-content"
        else:
            # Sidebar is opening
            return "sidebar sidebar-open", {"display": "none"}, {"display": "flex"}, "logo-container logo-container-open", "nav-left nav-left-open", "left-component left-component-open", "main-content main-content-shifted"
    # Initial state
    return current_sidebar_class, {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", current_main_content_class

# Add callback for chat item selection
@app.callback(
    Output("chat-list", "children"),
    [Input({"type": "chat-item", "index": ALL}, "n_clicks")],
    [State("chat-list", "children")]
)
def update_active_chat(n_clicks, current_items):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_items
    
    # Get the clicked item index
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    clicked_index = json.loads(triggered_id)["index"]
    
    # Update the active class
    updated_items = []
    for i, item in enumerate(current_items):
        if i == clicked_index:
            updated_items.append(html.Div(item["props"]["children"], className="chat-item active", id={"type": "chat-item", "index": i}))
        else:
            updated_items.append(html.Div(item["props"]["children"], className="chat-item", id={"type": "chat-item", "index": i}))
    
    return updated_items

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
