import urllib.request
import json
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def series_information(identifier):
    api_link_series = f'https://api.eia.gov/series/?api_key=122d80f9504786be1543b6425268a29b&series_id={identifier}'
    request = urllib.request.urlopen(api_link_series)
    request_body = request.read()
    body = json.loads(request_body.decode("utf-8"))

    data_frame = pd.DataFrame(body['series'][0]['data']).sort_values(by=[0])
    state_name = body['series'][0]['name'].split(", ")[1]
    data_frame.columns = ['Year', state_name]

    d_frame = pd.DataFrame(body['series'][0]['data']).sort_values(by=[0])
    d_frame.columns = ['Year', 'Consumption']
    d_frame['State'] = body['series'][0]['name'].split(", ")[1]
    d_frame['Code'] = identifier.split(".")[2]

    return data_frame, d_frame


def request_data(link):
    req = urllib.request.urlopen(link)
    req_body = req.read()
    j = json.loads(req_body.decode("utf-8"))
    df = pd.DataFrame(j['category']['childseries'])
    series_ids = df['series_id']

    series_info, state_info = series_information(series_ids[0])
    series_ids = series_ids.drop([0])
    for id_x in series_ids:
        info, state = series_information(id_x)
        series_info = series_info.merge(info, on='Year', how='left')
        state_info = state_info.append(state, ignore_index=True)
    series_info = series_info.drop(columns=['United States'])
    state_info = state_info[state_info.State != "United States"]
    return series_info, state_info


def k_means_cluster(cluster_data):
    # Ver quantos clusters devo fazer
    Nc = range(1, 10)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    score = [kmeans[i].fit(cluster_data).score(cluster_data) for i in range(len(kmeans))]
    plt.plot(Nc, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    # plt.show()

    model = KMeans(n_clusters=4).fit(cluster_data)
    pred = model.labels_
    cluster_data['Cluster'] = pred.astype(str)
    return cluster_data


petroleum_products = "https://api.eia.gov/category/?api_key=122d80f9504786be1543b6425268a29b&category_id=40445"
renewable_energy = "https://api.eia.gov/category/?api_key=122d80f9504786be1543b6425268a29b&category_id=40425"

series_info_petr, state_info_petr = request_data(petroleum_products)
series_info_ren, state_info_ren = request_data(renewable_energy)

# CLUSTERING
cluster_data_petr = k_means_cluster(series_info_petr.drop(columns=['Year']).T)
cluster_data_ren = k_means_cluster(series_info_ren.drop(columns=['Year']).T)

# DASHBOARD

# import style
external_stylesheets = ['mystyle.css']


# figures for graphs
def graph_data(x, y, title, y_name):
    fig = go.Figure(data=go.Scatter(x=x, y=y, marker=dict(color='#B22234')), layout=go.Layout(paper_bgcolor='rgb(0,0,0,0)',
                                                                                              plot_bgcolor='rgba(60, 59, 110, 0.1)'))
    fig.update_layout(title={'text': title, 'y': 0.95, 'x': 0.5, 'font': dict(family='Helvetica')},
                      xaxis_title={'text': "Time", 'font': dict(family='Helvetica')},
                      yaxis_title={'text': y_name, 'font': dict(family='Helvetica')})
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#3C3B6E', mirror=True,
                     showgrid=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#3C3B6E', mirror=True,
                     showgrid=True, gridwidth=1, gridcolor='rgba(60, 59, 110,0.1)')
    return fig


def draw_histogram(x, y1, y2, name1, name2):
    fig = go.Figure()
    fig.add_trace(go.Histogram(autobinx=False, autobiny=False, nbinsx=60, histfunc="sum", y=y1, x=x, name=name1))
    fig.add_trace(go.Histogram(autobinx=False, autobiny=False, nbinsx=60, histfunc="sum", y=y2, x=x, name=name2))
    return fig


def make_ranking(x, y, name):
    fig = go.Figure(data=go.Bar(x=x, y=y, orientation='h', name=name))
    return fig


def order_by_cons(df):
    df.sort_values(by=['Consumption'], inplace=True, ascending=True)
    return df


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server


# create main layout
app.layout = html.Div([
    html.Div([html.Div(className='sudiv_logo'),
              html.Div(html.Center(html.H3('Energy consumption in the USA')),
                       className='subdiv_h'),
              html.Div([html.Button('☰', id='button_dropdown', n_clicks=0, className='dropbtn'),
                        html.Div(id='dropdown_menu', className='dropdown-content')], className='subdiv_btn')],
             className='top-bar'),
    dcc.Location(id='url', refresh=False), html.Div(id='page_layout')
])


# creation of the layouts for each of dropdown menu's page -- START
map_layout = html.Div([dcc.Tabs(id='tabs', value='petroleum', children=[
    dcc.Tab(label='All petroleum sources', value='petroleum', className='color-text'),
    dcc.Tab(label='Renewable energy sources', value='renewable', className='color-text')], style={'font-family': 'Helvetica'}),
                       html.Div([dcc.Dropdown(id='choose-year',
                                      options=[{'label': str(i), 'value': i} for i in range(int(max(series_info_petr['Year'])),
                                                                                            int(min(series_info_petr['Year']))-1, -1)],
                                      value=2019, clearable=False),
                         html.Div(id='show-year', children=html.Div([html.Div(dcc.Graph(figure=make_ranking(order_by_cons(state_info_petr)[order_by_cons(state_info_petr)['Year']=='2019']['Consumption'],
                                                           order_by_cons(state_info_petr)[order_by_cons(state_info_petr)['Year']=='2019']['State'],
                                                           "Ranking")), style={"width": "40%", "display": "table-cell"}), html.Center([html.Div([
                             dcc.Graph(id='map-graph',
                                       figure=px.choropleth(data_frame=state_info_petr[state_info_petr['Year'] == str(2019)],
                                                            locationmode='USA-states',
                                                            locations='Code',
                                                            scope="usa",
                                                            color='Consumption',
                                                            hover_data=['State', 'Consumption'],
                                                            labels={"Energy Consumption from petroleum sources"},
                                                            template='ggplot2')
                                       )])
                         ], style={"width": "60%", "display": "table-cell"})], className='map')
                                  )])])

evolution_layout = html.Div([dcc.Tabs(id='tabs-evol', value='petroleum', children=[
    dcc.Tab(label='All petroleum sources', value='petroleum', className='color-text'),
    dcc.Tab(label='Renewable energy sources', value='renewable', className='color-text')], style={'font-family': 'Helvetica'}),
                        html.Div(id='evol-content')])


comp_source_layout = html.Div([html.Div(dcc.Dropdown(id='comp-options',
                                               options=[{'label': idx, 'value': idx} for idx in state_info_petr['State'].drop_duplicates()],
                                               value='Alaska',
                                               clearable=False),
                                  style={"width": "20%", "display": "table-cell", "vertical-align": "top",
                                         "font-family": "Helvetica"}),
                         html.Div([
                             dcc.Graph(id='comp-graph',
                                 figure=draw_histogram(state_info_petr[state_info_petr['State'] == "Alaska"]['Year'],
                                                       state_info_petr[state_info_petr['State'] == "Alaska"][
                                                           'Consumption'],
                                                       state_info_ren[state_info_ren['State'] == "Alaska"][
                                                           'Consumption'],
                                                       "Petroleum sources",
                                                       "Renewable sources"))],
                            style={"width": "80%", "display": "table-cell"})],
                        className='graphs-comp')


temp = state_info_petr.drop(columns=['Year', 'Consumption']).drop_duplicates().set_index('State')
temp = temp.merge(cluster_data_petr['Cluster'], left_index=True, right_index=True)
clustering_layout = html.Div([dcc.Tabs(id='tabs-cluster', value='petroleum', children=[
                                dcc.Tab(label='Cluster of petroleum sources profiles', value='petroleum', className='color-text'),
                                dcc.Tab(label='Cluster of renewable energy sources profiles', value='renewable', className='color-text')], style={'font-family': 'Helvetica'}),
                             html.Div(children=html.Div([html.Center([html.Div([
                                 dcc.Graph(
                                     id='show-cluster',
                                     figure=px.choropleth(data_frame=temp,
                                                          locationmode='USA-states',
                                                          locations='Code',
                                                          scope="usa",
                                                          color='Cluster',
                                                          hover_data=[temp.index, 'Cluster'],
                                                          color_discrete_map={'0': 'blue', '1': 'red', '2': 'green'},
                                                          labels={"Cluster": "Cluster"},
                                                          template='ggplot2',
                                                          )
                                 )])])]))])

about_layout = html.Div([html.H3("Author: Inês Andrade Rainho.\nSource: https://www.eia.gov/", style={'color': 'black'})])
# -- END


# callbacks
# callback for the button with dropdown menu
@app.callback(Output('dropdown_menu', 'children'),
              Input('button_dropdown', 'n_clicks'))
def toggle_dropdown(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button_dropdown' in changed_id:
        return [html.A('Map of consumption', href='/map'),
                html.A('Consumption evolution by state', href='/evolution'),
                html.A('Comparison of consumption by type of source', href='/comp_source'),
                html.A('Clusters of state profiles', href='/clustering'),
                html.A('About', href='/about')] if n_clicks % 2 != 0 else []
    else:
        return []


@app.callback(Output('show-year', 'children'),
              [Input('tabs', 'value'),
              Input('choose-year', 'value')])
def update_tab_map(value_source, value_year):
    if value_source=='renewable':
        state_info = state_info_ren
        title = 'Energy Consumption from renewable sources'
    else:
        state_info = state_info_petr
        title = 'Energy Consumption from petroleum sources'

    if value_year is None:
        value_year = 2019

    df = state_info.copy()
    df = df[df['Year'] == str(value_year)]

    fig = px.choropleth(data_frame=df,
                        locationmode='USA-states',
                        locations='Code',
                        scope="usa",
                        color='Consumption',
                        hover_data=['State', 'Consumption'],
                        labels={title},
                        template='ggplot2')

    return html.Div([html.Div(dcc.Graph(figure=make_ranking(order_by_cons(df)['Consumption'],
                                                           order_by_cons(df)['State'],
                                                           "Ranking")), style={"width": "40%", "display": "table-cell"}),
                     html.Center([html.Div([
                                dcc.Graph(
                                    id='raw-data-graph',
                                    figure=fig
                                )
                            ])], style={"width": "60%", "display": "table-cell"})], className='map')


@app.callback(Output('evolution-graph', 'figure'),
              [Input('evolution-options', 'value'),
               Input('tabs-evol','value')])
def upgrade_evolution(value_year, value_source):
    if value_year is None:
        value_year = "Alaska"
    if value_source=='renewable':
        series_info = series_info_ren
        title = "Energy consumption for renewable sources (billion Btu)"
    else:
        series_info = series_info_petr
        title = "Energy consumption for petroleum sources (billion Btu)"
    return graph_data(series_info['Year'], series_info[value_year], value_year, title)


@app.callback(Output('comp-graph', 'figure'),
              [Input('comp-options', 'value')])
def upgrade_evolution(value):
    if value is None:
        value = "Alaska"
    return draw_histogram(state_info_petr[state_info_petr['State'] == value]['Year'],
                                                       state_info_petr[state_info_petr['State'] == value][
                                                           'Consumption'],
                                                       state_info_ren[state_info_ren['State'] == value][
                                                           'Consumption'],
                                                       "Petroleum sources",
                                                       "Renewable sources")



@app.callback(Output('evol-content', 'children'),
              Input('tabs-evol', 'value'))
def update_evol(value):
    if value=='renewable':
        state_info = state_info_ren
        series_info = series_info_ren
        title = "Energy consumption from renewable sources (billion Btu)"
    else:
        state_info = state_info_petr
        series_info = series_info_petr
        title = "Energy consumption from petroleum sources (billion Btu)"
    return html.Div([html.Div(dcc.Dropdown(id='evolution-options',
                                               options=[{'label': idx, 'value': idx} for idx in state_info['State'].drop_duplicates()],
                                               value='Alaska',
                                               clearable=False),
                                  style={"width": "20%", "display": "table-cell", "vertical-align": "top",
                                         "font-family": "Helvetica"}),
                         html.Div([
                             dcc.Graph(id='evolution-graph',
                                       figure=graph_data(series_info['Year'], series_info['Alaska'],
                                                         "Alaska", title))],
                            style={"width": "80%", "display": "table-cell"})],
                        className='graphs-evol')


@app.callback(Output('show-cluster', 'figure'),
              Input('tabs-cluster', 'value'))
def update_cluster(value):
    if value=='renewable':
        temp = state_info_ren.drop(columns=['Year', 'Consumption']).drop_duplicates().set_index('State')
        temp = temp.merge(cluster_data_ren['Cluster'], left_index=True, right_index=True)
    else:
        temp = state_info_petr.drop(columns=['Year', 'Consumption']).drop_duplicates().set_index('State')
        temp = temp.merge(cluster_data_petr['Cluster'], left_index=True, right_index=True)
    return px.choropleth(data_frame=temp,
                         locationmode='USA-states',
                         locations='Code',
                         scope="usa",
                         color='Cluster',
                         hover_data=[temp.index, 'Cluster'],
                         color_discrete_map={'0': 'blue', '1': 'red', '2': 'green'},
                         labels={"Cluster": "Cluster"},
                         template='ggplot2'
                         )


@app.callback(Output('page_layout', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/evolution':
        return evolution_layout
    elif pathname == '/comp_source':
        return comp_source_layout
    elif pathname == '/clustering':
        return clustering_layout
    elif pathname == '/about':
        return about_layout
    else:
        return map_layout


if __name__ == '__main__':
    app.run_server(debug=True)