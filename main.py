import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# --- Дані для симуляції ---
initial_population = np.array([100, 65, 78, 140])
fertility_rates = np.array([0.3, 2.5, 3.7, 0.3])
survival_rates = np.array([0.5, 0.9, 0.75])
years = 50
age_groups = ['0 → 1', '1 → 2', '2 → 3', '3 → <4']

# --- Модель Леслі ---
def create_leslie_matrix(fertility, survival, num_age_groups):
    leslie_matrix = np.zeros((num_age_groups, num_age_groups))
    leslie_matrix[0, :] = fertility
    for i in range(num_age_groups - 1):
        leslie_matrix[i + 1, i] = survival[i]
    return leslie_matrix

leslie_matrix = create_leslie_matrix(fertility_rates, survival_rates, len(age_groups))

# --- Симуляція популяції ---
def simulate_population(leslie_matrix, initial_pop, num_years):
    population_data = np.zeros((num_years, len(initial_pop)))
    current_pop = np.copy(initial_pop)
    population_data[0, :] = current_pop
    for year in range(1, num_years):
        current_pop = np.dot(leslie_matrix, current_pop)
        population_data[year, :] = current_pop
    return population_data

population_over_time = simulate_population(leslie_matrix, initial_population, years)

# --- Аналіз ---
total_population = np.sum(population_over_time, axis=1)
growth_rates = np.where(total_population[:-1] > 0,
                        total_population[1:] / total_population[:-1],
                        np.nan)
average_growth = np.nanmean(growth_rates)
is_growing = "зростає" if average_growth > 1 else "зменшується або стабільна"
stability_text = (
    f"Середня швидкість зростання (коеф.) за {years} років: {average_growth:.4f}. "
    f"Оскільки це значення {'> 1' if average_growth > 1 else '≤ 1'}, популяція, ймовірно, {is_growing}."
)

# --- Підготовка даних ---
df = pd.DataFrame(population_over_time, columns=age_groups)
df['Рік'] = np.arange(years)
df_melted = df.melt(id_vars='Рік', var_name='Вікова група', value_name='Кількість особин')

# Частки по роках (для stacked area у відсотках)
df_share = df.copy()
row_sums = df_share[age_groups].sum(axis=1).replace(0, np.nan)
df_share[age_groups] = (df_share[age_groups].div(row_sums, axis=0) * 100)

# Колірна палітра
colors = px.colors.qualitative.Plotly

# --- Фігури, які не потребують callback ---
age_distribution_fig = px.bar(
    df_melted,
    x='Вікова група',
    y='Кількість особин',
    color='Вікова група',
    animation_frame='Рік',
    category_orders={'Вікова група': age_groups},
    labels={'Кількість особин': 'Кількість особин', 'Вікова група': 'Вікова група'},
    title='Анімований розподіл популяції за віком'
)
age_distribution_fig.update_layout(transition={'duration': 0}, uirevision="static")

heatmap_fig = px.imshow(
    df[age_groups].T,
    labels=dict(x='Рік', y='Вікова група', color='Особи'),
    x=np.arange(years),
    y=age_groups,
    aspect='auto',
    title='Heatmap: рік × вікова група'
).update_layout(coloraxis_colorbar=dict(title='Особи'), uirevision="static")

growth_fig = px.line(
    x=np.arange(1, years), y=growth_rates,
    labels={'x': 'Рік', 'y': 'Коеф. зростання'},
    title='Коефіцієнт зростання (t / t-1)'
).update_layout(hovermode='x unified', uirevision="static")

# --- Dash App ---
app = dash.Dash(__name__)
app.title = "Динаміка популяції мишей"

# KPI-картка
kpi_style = {
    'flex': '1',
    'backgroundColor': 'white',
    'padding': '16px',
    'borderRadius': '10px',
    'boxShadow': '0 4px 8px rgba(0,0,0,0.08)',
    'textAlign': 'center',
    'minWidth': '220px'
}

graph_common_style = {'height': '420px'}  # Фіксуємо висоту графіків

app.layout = html.Div(style={
    'backgroundColor': '#eef2f7',
    'fontFamily': 'Arial, sans-serif',
    'padding': '20px',
    'maxWidth': '1400px',
    'margin': '0 auto'
}, children=[
    html.Div(style={
        'backgroundColor': 'white',
        'padding': '24px',
        'borderRadius': '12px',
        'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
        'textAlign': 'center',
        'marginBottom': '20px'
    }, children=[
        html.H1("Динаміка популяції мишей", style={'color': '#2c3e50', 'fontSize': '2.2em', 'marginBottom': '6px'}),
        html.P("Візуалізація та аналіз на основі моделі Леслі", style={'color': '#7f8c8d', 'margin': 0})
    ]),

    html.Div(style={'display': 'flex', 'gap': '16px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style=kpi_style, children=[
            html.Div("Загальна популяція (поч.)", style={'color': '#7f8c8d'}),
            html.H2(f"{int(total_population[0]):,}".replace(",", " "))
        ]),
        html.Div(style=kpi_style, children=[
            html.Div("Загальна популяція (фін.)", style={'color': '#7f8c8d'}),
            html.H2(f"{int(total_population[-1]):,}".replace(",", " "))
        ]),
        html.Div(style=kpi_style, children=[
            html.Div("Середній річний коеф. зростання", style={'color': '#7f8c8d'}),
            html.H2(f"{average_growth:.4f}")
        ]),
        html.Div(style=kpi_style, children=[
            html.Div("Тренд", style={'color': '#7f8c8d'}),
            html.H2(is_growing)
        ]),
    ]),

    html.Div(style={
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    }, children=[
        html.H3("Аналіз популяції", style={'color': '#34495e', 'marginTop': 0}),
        html.P(stability_text, style={'fontSize': '1.05em', 'margin': 0})
    ]),

    dcc.Tabs(id='tabs', value='tab-lines', children=[

        dcc.Tab(label='Лінійні ряди + налаштування', value='tab-lines', children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginTop': '16px'}, children=[
                html.Div(style={
                    'flex': '1 1 420px',
                    'minWidth': '320px',
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                }, children=[
                    html.H4("Зміни кількості особин за роками", style={'color': '#34495e'}),
                    dcc.Graph(id='population-graph', style=graph_common_style, config={'displaylogo': False}),
                    html.Label("Виберіть вікові групи:", style={'fontWeight': 'bold', 'color': '#34495e'}),
                    dcc.Checklist(
                        id='age-group-checklist',
                        options=[{'label': f'Група {group}', 'value': i} for i, group in enumerate(age_groups)],
                        value=list(range(len(age_groups))),
                        inline=True,
                        style={'marginBottom': '10px'}
                    ),
                    html.Br(),
                    html.Label("Масштаб осі Y:", style={'fontWeight': 'bold', 'color': '#34495e'}),
                    dcc.RadioItems(
                        id='y-axis-type',
                        options=[
                            {'label': 'Лінійний', 'value': 'linear'},
                            {'label': 'Логарифмічний', 'value': 'log'}
                        ],
                        value='linear',
                        inline=True
                    )
                ]),

                html.Div(style={
                    'flex': '1 1 420px',
                    'minWidth': '320px',
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                }, children=[
                    html.H4("Динаміка загальної популяції", style={'color': '#34495e'}),
                    dcc.Graph(
                        id='total-population-graph',
                        style=graph_common_style,
                        config={'displaylogo': False},
                        figure={
                            'data': [
                                {'x': np.arange(years), 'y': total_population, 'type': 'scatter', 'mode': 'lines+markers',
                                 'name': 'Загальна популяція'}
                            ],
                            'layout': {
                                'title': 'Загальна кількість особин',
                                'xaxis': {'title': 'Рік'},
                                'yaxis': {'title': 'Кількість особин'},
                                'hovermode': 'x unified',
                                'uirevision': 'static'
                            }
                        }
                    ),
                    html.H4("Темп зростання, разів/рік", style={'color': '#34495e', 'marginTop': '12px'}),
                    dcc.Graph(id='growth-rate-graph', style=graph_common_style,
                              config={'displaylogo': False}, figure=growth_fig)
                ]),
            ])
        ]),

        dcc.Tab(label='Stacked Area (рівні/частки)', value='tab-area', children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginTop': '16px'}, children=[
                html.Div(style={
                    'flex': '1 1 600px',
                    'minWidth': '320px',
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                }, children=[
                    html.Label("Нормувати по роках (у %):", style={'fontWeight': 'bold', 'color': '#34495e'}),
                    dcc.Checklist(
                        id='normalize-share',
                        options=[{'label': 'Показати у %', 'value': 'pct'}],
                        value=[],
                        inline=True
                    ),
                    dcc.Graph(id='stacked-area-graph', style=graph_common_style, config={'displaylogo': False})
                ])
            ])
        ]),

        dcc.Tab(label='Теплова карта', value='tab-heatmap', children=[
            html.Div(style={'padding': '16px'}, children=[
                dcc.Graph(id='heatmap', style=graph_common_style,
                          config={'displaylogo': False}, figure=heatmap_fig)
            ])
        ]),

        dcc.Tab(label='Вікова "піраміда"', value='tab-pyramid', children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'padding': '16px'}, children=[
                html.Div(style={'flex': '1 1 500px', 'minWidth': '320px'}, children=[
                    html.Label("Оберіть рік:", style={'fontWeight': 'bold', 'color': '#34495e'}),
                    dcc.Slider(
                        id='pyramid-year',
                        min=0, max=years-1, step=1, value=0,
                        marks={0: '0', years-1: str(years-1)},
                        tooltip={'always_visible': False}
                    ),
                    dcc.Graph(id='age-pyramid', style=graph_common_style, config={'displaylogo': False})
                ])
            ])
        ]),

        dcc.Tab(label='Анімація розподілу', value='tab-animation', children=[
            html.Div(style={'padding': '16px'}, children=[
                dcc.Graph(id='age-distribution-graph',
                          style={'height': '520px'},  # трішки вище для анімації
                          config={'displaylogo': False},
                          figure=age_distribution_fig)
            ])
        ]),
    ])
])

# --- Callbacks ---

# Лінійні ряди (із захистом лог-шкали)
@app.callback(
    Output('population-graph', 'figure'),
    [Input('age-group-checklist', 'value'),
     Input('y-axis-type', 'value')]
)
def update_population_graph(selected_groups, y_axis_type):
    fig = go.Figure()
    xs = np.arange(years)
    eps = 1e-6  # щоб не ламався логарифм
    for i in selected_groups:
        y = population_over_time[:, i].astype(float)
        if y_axis_type == 'log':
            y = np.where(y <= 0, eps, y)
        fig.add_trace(go.Scatter(
            x=xs,
            y=y,
            mode='lines+markers',
            name=f'Група {age_groups[i]}',
            line=dict(color=colors[i % len(colors)])
        ))
    fig.update_layout(
        title='Зміни кількості особин за роками',
        xaxis_title='Рік',
        yaxis_title='Кількість особин',
        yaxis_type=y_axis_type,
        hovermode='x unified',
        uirevision='static'
    )
    return fig

# Stacked area (рівні або частки %)
@app.callback(
    Output('stacked-area-graph', 'figure'),
    Input('normalize-share', 'value')
)
def update_stacked_area(normalize_flags):
    if 'pct' in normalize_flags:
        _df = df_share.copy()
        y_title = 'Частка, %'
        title = 'Stacked Area: вікові частки у %'
    else:
        _df = df.copy()
        y_title = 'Кількість особин'
        title = 'Stacked Area: рівні чисельності'

    melted = _df.melt(id_vars='Рік', var_name='Вікова група', value_name='Значення')
    fig = px.area(
        melted, x='Рік', y='Значення', color='Вікова група',
        category_orders={'Вікова група': age_groups},
        title=title
    )
    fig.update_layout(yaxis_title=y_title, hovermode='x unified', uirevision='static')
    return fig

# Вікова "піраміда" (горизонтальна діаграма для обраного року)
@app.callback(
    Output('age-pyramid', 'figure'),
    Input('pyramid-year', 'value')
)
def update_pyramid(year_idx):
    values = df.loc[year_idx, age_groups].values.astype(float)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=age_groups,
        orientation='h',
        marker_color=colors[:len(age_groups)],
        name=f'Рік {year_idx}'
    ))
    fig.update_layout(
        title=f'Вікова структура (рік {year_idx})',
        xaxis_title='Кількість особин',
        yaxis_title='Вікова група',
        barmode='stack',
        hovermode='y',
        uirevision='static'
    )
    return fig

# --- Запуск сервера ---
if __name__ == '__main__':
    app.run(debug=True)
