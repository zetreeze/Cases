from dash import Dash, html, dcc, callback, Output, Input, dash_table
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc

# Incorporate data
country_wise = pd.read_csv('country_wise_latest.csv.xls')
conf_act = country_wise.loc[:, ['Country/Region', 'Confirmed', 'Active']]
df = pd.read_csv('covid_19_clean_complete.csv')
dead_rec = country_wise.loc[:, ['Country/Region','Deaths','Recovered']]
dead_rec['Deaths/Recovered'] = dead_rec['Deaths'] / dead_rec['Recovered'] * 100

top10_conf_contr = country_wise.loc[:, ['Country/Region','Confirmed']]
top10_conf_contr = top10_conf_contr.sort_values('Confirmed', ascending=False).head(20)
top10_conf_contr.columns = ['Country', 'Confirmed']

ghj3 = px.histogram(top10_conf_contr, x=top10_conf_contr['Country'], y=top10_conf_contr['Confirmed'], text_auto=True)
ghj3.update_layout(title_text = 'Топ 20 стран по числу зараженных', title_x = 0.5)

# Initialize the app
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dmc.Title('Covid-19 Dashboard', color="blue", size="h3", align='center'),
        html.Hr(),
        dmc.Text('Скорость распространения заболевания по стране:', color="black", size="h6", align='center'),
        dcc.Dropdown(
            id='dropdown-selection',
            options=[{'label': country, 'value': country} for country in conf_act['Country/Region'].unique()],
            value='Russia'
        ),
        
        dcc.Graph(id='graph1', style={'width': '100%', 'float': 'left'}),
        html.Hr()
    ]),
    html.Div([
        dcc.Dropdown(
            id='dropdown-selection2',
            options=[{'label': country, 'value': country} for country in conf_act['Country/Region'].unique()],
            value='Russia'
        ),
        dcc.Graph(id='graph2', style={'width': '33%', 'float': 'left'})
    ]),
    html.Div([
        dcc.Graph(id='graph3', style={'width': '33%', 'float': 'left'}),
    ]),
    html.Div([
        dcc.Graph(id='graph4', style={'width': '33%', 'float': 'left'})
    ]),    
    html.Div([
        html.Hr(style={'width': '100%', 'float': 'right'}),
        dcc.Graph(figure=ghj3, style={'width': '100%', 'float': 'right'})
    ]),       
])

# Add controls to build the interaction
@callback(
    Output('graph1', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph1(value):
    dff = df[df['Country/Region']==value]
    return px.bar(dff, x='Date', y='Active')

@callback(
    Output('graph2', 'figure'),
    Output('graph3', 'figure'),
    Output('graph4', 'figure'),
    Input('dropdown-selection2', 'value'),
)
def update_graph2(value):
    dff = conf_act[conf_act['Country/Region'] == value]
    ghj = px.histogram(dff, x='Country/Region', y=['Confirmed', 'Active'], barmode='group', labels={'value': 'Confirmed and Active', 'Country/Region':'Country'})
    dff2 = dead_rec[dead_rec['Country/Region'] == value]
    ghj1 = px.histogram(dff2, x='Country/Region', y=['Deaths', 'Recovered'], barmode='group', labels={'value': 'Deaths and Recovered', 'Country/Region':'Country'})
    ghj2 = px.histogram(dff2, x='Country/Region', y='Deaths/Recovered', text_auto=True, labels={'Country/Region':'Country', 'Deaths/Recovered': '(Deaths/Recovered, %)'})
    ghj2.update_layout(title_text = 'Какой-то коэффициент (Смерти/Выздоровевшие), %', title_x = 0.5)
    ghj1.update_layout(title_text = 'Случаи выздоровления и смерти от вируса', title_x = 0.5)
    ghj.update_layout(title_text = 'Подтвержденные и активные случаи заболеваний', title_x = 0.5)
    return ghj, ghj1, ghj2

# Run the app
if __name__ == '__main__':
    app.run(debug=True)