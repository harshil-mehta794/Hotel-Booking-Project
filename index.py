from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from app import app, server
import knn
import rf
import gbm

colors = {
    'background': '#A5F700',
    'text': 'Red'
}

style = {'maxWidth': '3000px', 'margin': 'auto', 'backgroundColor': '#8CDDDA'}

app.layout = html.Div([
    dcc.Markdown('# Prediction of Hotel Cancellation'),
    dcc.Tabs(id='tabs', value='tab-intro', children=[

        dcc.Tab(label='K-nearest Neighbors', value='tab-knn', style={'backgroundColor': colors['background']}),

        dcc.Tab(label='Random Forest', value='tab-RandomForest'),
        dcc.Tab(label='Gradient Boosting Classifier', value='tab-GBM', style={'backgroundColor': colors['background']})

    ], colors={ "border": "white",
        "primary": "gold",
        "background": "#FFFF42"}),
    html.Div(id='tabs-content'),
], style=style)

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):

    if tab == 'tab-knn': return knn.layout
    elif tab == 'tab-RandomForest': return rf.layout
    elif tab == 'tab-GBM': return gbm.layout
    
if __name__ == '__main__':
    app.run_server(debug=True)