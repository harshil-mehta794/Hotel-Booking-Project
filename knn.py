import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input ,Output
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier
import dash
import dash_table
from sklearn.preprocessing import StandardScaler
import joblib
from app import app

data = pd.read_pickle('hotel_bookingsAPP.pkl')

X = data.drop('is_canceled', axis = 1)
y = data['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

def trainKNN(KNN_k_neighbors):
    classifier= KNeighborsClassifier(n_neighbors=int(KNN_k_neighbors))
    model = classifier.fit(X_train_std, y_train)
    return model

def cR_to_df(y_test, y_pred):
    cr = classification_report(y_test, y_pred, output_dict=True)
    row = {}
    row['Precision'] = [round(float(cr['1']['precision'])*100,2)]
    row['Recall'] = [round(float(cr['1']['recall'])*100,2)]
    row['Accuracy'] = [round(metrics.accuracy_score(y_test, y_pred) * 100,2)]



    df = pd.DataFrame.from_dict(row)

    return df

def cm2df(y_test,y_pred):
    cm=confusion_matrix(y_test,y_pred)
    labels=['0','1']
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]


style = {'padding': '1.5em'}
layout = html.Div([
    dcc.Markdown("""
        ### KNN
        Use the controls below for Hyper Parameter Tuning
    """),
        html.Div(id='prediction-content', style={'fontWeight': 'bold'}),
        html.Div([
        dcc.Markdown('###### n_neighbors'),
        dcc.Dropdown(
            id='n_neighbors',
            options=[{'label': i, 'value': i} for i in [2,3,4,5]],
            value= 4
        ),
    ], style=style),
    ]
)

@app.callback(Output('prediction-content', 'children'),
              [Input('n_neighbors', 'value')]
              )
def predict(n_neighbors):
    if(n_neighbors == 4):
        pipeline = joblib.load('pipeline.joblib')

    else:
        pipeline  = trainKNN(n_neighbors)

    y_pred = pipeline.predict(X_test_std)
    performaceDF = cR_to_df(y_test, y_pred)
    confusionDF = cm2df(y_test, y_pred)

    tab1 = html.Div(children=[
        html.Label(children='Performance Matrix', style={'width': '50%', 'display': 'inline-block',
                                                         'margin': 0, 'padding': '8px'}),
        dash_table.DataTable(
            id='table_no1',
            columns=[{"name": i, "id": i} for i in performaceDF.columns],
            data=performaceDF.to_dict("rows"),
            style_table={'width': '80%',
                         },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Parameters'},
                 'width': '50%'}
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={'width': '180px',
                        'height': '60px',
                        'textAlign': 'left',
                        'minWidth': '0px',
                        'maxWidth': '180px'
                        })])

    tab2 = html.Div(
        children=[html.Label(children='Confusion Matrix', style={'width': '50%', 'display': 'inline-block',
                                                                 'margin': 0, 'padding': '8px'}),
                  dash_table.DataTable(
                      id='confusionMatrix',
                      columns=[{"name": i, "id": i} for i in confusionDF.columns],
                      data=confusionDF.to_dict("rows"),
                      style_table={'width': '60%',
                                   },
                      style_data_conditional=[
                          {
                              'if': {'row_index': 'odd'},
                              'backgroundColor': 'rgb(248, 248, 248)'
                          }
                      ],
                      style_header={
                          'backgroundColor': 'rgb(230, 230, 230)',
                          'fontWeight': 'bold'
                      },
                      style_cell={'width': '180px',
                                  'height': '42px',
                                  'textAlign': 'left',
                                  'minWidth': '0px',
                                  'maxWidth': '180px'
                                  })])
    final = html.Div([
        html.Br([]),
        tab1, tab2], style=dict(display='flex'))
    # testdiv=([html.H4('Random Forest', style=dict(color='white', background='red'))])
    return final

