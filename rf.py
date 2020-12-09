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
from sklearn.ensemble import RandomForestClassifier


data = pd.read_pickle('hotel_bookingsAPP.pkl')

X = data.drop('is_canceled', axis = 1)
y = data['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

def trainRF(rf_criterion,rf_max_depth,rf_max_features,rf_min_samples_leaf,rf_n_estimators):
    classifier= RandomForestClassifier(criterion=rf_criterion, n_estimators=int(rf_n_estimators), max_depth=int(rf_max_depth), max_features=int(rf_max_features), min_samples_leaf=int(rf_min_samples_leaf),
                           random_state=16)
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
        ### Random Forest
        Use the controls below for Hyper Parameter Tuning
    """),
        html.Div(id='prediction-content_rf', style={'fontWeight': 'bold'}),
        html.Div([
        dcc.Markdown('###### n_estimators'),
        dcc.Dropdown(
            id='n_estimators',
            options=[{'label': i, 'value': i} for i in [10,50,100,150,200,300,400]],
            value= 200
        ),
    ], style=style),

        html.Div([
        dcc.Markdown('###### criterion'),
        dcc.Dropdown(
            id='criterion',
            options=[{'label': i, 'value': i} for i in ['gini', 'entropy']],
            value= 'gini'
        ),
    ], style=style),

        html.Div([
        dcc.Markdown('###### max_depth'),
        dcc.Dropdown(
            id='max_depth',
            options=[{'label': i, 'value': i} for i in [5,10,20,25]],
            value= 25
        ),
    ], style=style),

        html.Div([
        dcc.Markdown('###### max_features'),
        dcc.Dropdown(
            id='max_features',
            options=[{'label': i, 'value': i} for i in [2,4,6]],
            value= 6
        ),
    ], style=style),

        html.Div([
        dcc.Markdown('###### min_samples_leaf'),
        dcc.Dropdown(
            id='min_samples_leaf',
            options=[{'label': i, 'value': i} for i in [1,2,4]],
            value= 1
        ),
    ], style=style),
    ]
)

@app.callback(Output('prediction-content_rf', 'children'),
              [Input('criterion', 'value'),
               Input('max_depth', 'value'),
               Input('max_features', 'value'),
               Input('min_samples_leaf', 'value'),
               Input('n_estimators', 'value')]
              )
def rf(criterion, max_depth, max_features, min_samples_leaf, n_estimators):
    if(criterion == 'gini' and max_depth == 25 and max_features == 6 and min_samples_leaf == 1 and n_estimators==200):
        pipeline = joblib.load('rf.joblib')

    else:
        pipeline  = trainRF(criterion, max_depth, max_features, min_samples_leaf, n_estimators)

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

    return final

