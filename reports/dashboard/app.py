# reports/dashboard/app.py
from pathlib import Path
import sys
import os
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import joblib
import os
import base64
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).parent.resolve()
print(current_location)
sys.path.append(str(current_location))

report_dir = os.path.join(current_location, "reports")


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to encode images to display in Dash
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read()).decode()
    return "data:image/png;base64,{}".format(encoded)

# Load evaluation metrics
evaluation_metrics = pd.read_json('reports/evaluation_metrics.json', orient='index')

# Load model comparison metrics
model_comparison = pd.read_csv('reports/model_comparison.csv')

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Employee Attrition Executive Dashboard"), className="mb-4")
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Random Forest', 'value': 'RandomForest'},
                    {'label': 'XGBoost', 'value': 'XGBoost'}
                ],
                value='RandomForest',
                clearable=False
            )
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H2(id='roc-auc'),
            html.H3("Classification Metrics:"),
            html.P(id='precision'),
            html.P(id='recall'),
            html.P(id='f1_score')
        ], width=4),
        dbc.Col([
            dcc.Graph(id='feature-importance')
        ], width=8)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H2("SHAP Summary Plot"),
            html.Img(id='shap-image', style={'width':'100%', 'height':'auto'})
        ])
    ])
])

# Callbacks to update dashboard based on selected model
@app.callback(
    [Output('roc-auc', 'children'),
     Output('precision', 'children'),
     Output('recall', 'children'),
     Output('f1_score', 'children'),
     Output('feature-importance', 'figure'),
     Output('shap-image', 'src')],
    [Input('model-dropdown', 'value')]
)
def update_dashboard(selected_model):
    # Fetch metrics for the selected model
    roc_auc = model_comparison[model_comparison['Model'] == selected_model]['ROC_AUC'].values[0]
    precision = model_comparison[model_comparison['Model'] == selected_model]['Precision'].values[0]
    recall = model_comparison[model_comparison['Model'] == selected_model]['Recall'].values[0]
    f1_score = model_comparison[model_comparison['Model'] == selected_model]['F1_Score'].values[0]
    
    # Update metrics display
    roc_auc_text = f"ROC AUC Score: {roc_auc:.4f}"
    precision_text = f"Precision: {precision:.2f}"
    recall_text = f"Recall: {recall:.2f}"
    f1_score_text = f"F1 Score: {f1_score:.2f}"
    
    # Load feature importance
    feature_importance_path = os.path.join(report_dir, f'dashboard_feature_importance_{selected_model.lower()}.csv')
    print("feature_importance_path", feature_importance_path)
    feature_importance = pd.read_csv(feature_importance_path)
    fig_importance = px.bar(feature_importance, x='Importance', y='Feature',
                            orientation='h', title='Feature Importances')
    
    # Load SHAP summary plot
    shap_image_path = f'reports/shap_summary_{selected_model.lower()}.png'
    shap_image = encode_image(shap_image_path)
    
    return roc_auc_text, precision_text, recall_text, f1_score_text, fig_importance, shap_image

if __name__ == '__main__':
    app.run_server(debug=True)
