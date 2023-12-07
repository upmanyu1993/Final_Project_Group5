import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objs as go
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Excel File'),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    dcc.Graph(id='stacked-bar-chart'),
    dcc.Graph(id='pie-chart'),
    dcc.Graph(id='area-chart'),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='extrapolated-values-bar-chart')
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, 'Unsupported file format. Please upload a CSV or Excel file.'
    except Exception as e:
        return None, f'There was an error processing this file: {e}'
    return df, None


@app.callback(
    [Output('stacked-bar-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('area-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('extrapolated-values-bar-chart', 'figure'),
     Output('output-data-upload', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        df, error = parse_contents(contents, filename)
        if df is None:
            return [go.Figure() for _ in range(5)] + [html.Div([error])]
        return list(update_charts(df)) + [html.Div([
            html.H5(f'File uploaded: {filename}'),
            html.Hr(),
            html.P(f'Number of rows in the dataset: {len(df)}')
        ])]
    return [go.Figure() for _ in range(5)] + [html.Div([])]


def update_charts(df):
    df_melted = df.melt(id_vars='Month', var_name='Category', value_name='Amount')
    df_melted['Month_Num'] = df_melted['Month'].astype('category').cat.codes + 1

    # Stacked Bar Chart
    stacked_bar_chart = px.bar(df_melted, x='Month', y='Amount', color='Category', title='Monthly Expenses by Category', barmode='stack')

    # Pie Chart
    pie_chart = px.pie(df_melted, values='Amount', names='Category', title='Expense Distribution')

    # Area Chart
    area_chart = px.area(df_melted, x='Month', y='Amount', color='Category', title='Monthly Expenses Area Chart')

    # Scatter Plot
    scatter_plot = px.scatter(df_melted, x='Category', y='Amount', color='Month', size='Amount', hover_name='Category')

    # Extrapolated Values Bar Chart
    extrapolated_values = []
    for category in df_melted['Category'].unique():
        category_data = df_melted[df_melted['Category'] == category]
        X = category_data[['Month_Num']]
        print (X)
        y = category_data['Amount']
        model = LinearRegression().fit(X, y)
        predicted_value = model.predict([[df_melted['Month_Num'].max() + 1]])[0]
        extrapolated_values.append((category, predicted_value))

    df_extrapolated = pd.DataFrame(extrapolated_values, columns=['Category', 'Amount'])
    extrapolated_bar_chart = px.bar(df_extrapolated, x='Category', y='Amount', title='Extrapolated Expenses')

    return stacked_bar_chart, pie_chart, area_chart, scatter_plot, extrapolated_bar_chart


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
