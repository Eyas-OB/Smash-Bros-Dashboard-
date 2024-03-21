import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import plotly.express as px 
import pandas as pd 
import plotly.graph_objects as go
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


app = dash.Dash(__name__)
server = app.server

file_path = 'my_venv\\Cream City Clash\\Data\\Cream City Clash fall 2023.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter out the player name "Grand Total"
# Sort the graph by the plyers with the most tournaments entered 
data_filter = df[df['Players'] != 'Grand Total']
data_filter = data_filter.sort_values(by='Tournaments Entered', ascending=False)

# Select the top 10 players
top_10_players = data_filter.head(10)

# Create a bar chart for the top 10 players
fig1 = px.bar(top_10_players, x='Players', y='Tournaments Entered')

# Update the layout of the first figure with the following details:
# Title: 'Top 10 Players by Number Of Tournaments Entered'
# x-axis title: 'Player Name'
# y-axis title: 'Number Of Tournaments Entered'
fig1.update_layout(
    title=dict(
        text= "Players With The Most Attendance",
        x=0.5,  # Set the title's x position to the center of the plot
        xanchor='center',  # Anchor the x position to the center
        font=dict(size=24),  # Set the font size
    ),
    xaxis_title='',
    yaxis_title='Number Of Tournaments Entered',
    template='plotly_dark',
    width=1000,  # Set the desired width
    height=800,  # Set the desired height
    title_x=0.5,  # Center the title horizontally
    title_y=0.97,  # Position the title closer to the top
)  
# this gives each line on the graph a diffrent color to look cool 
fig1.update_traces(marker_color=top_10_players['Tournaments Entered'])



#CSV file for wieghted average graph

file_path1 = 'my_venv\\Cream City Clash\\Data\\top_players_performance.csv'

# Load the CSV file into a DataFrame
df2 = pd.read_csv(file_path1)

fig2 = px.bar(df2, x= 'Players', y= 'Weighted Average')

fig2.update_layout(
    title=dict(
        text= "Top 10 Players For Wighted Average",
        x=0.5,  # Set the title's x position to the center of the plot
        xanchor='center',  # Anchor the x position to the center
        font=dict(size=24),  # Set the font size
    ),
    width=1000,  # Set the desired width
    height=800,  # Set the desired height
    title_x=0.5,  # Center the title horizontally
    title_y=0.97,  # Position the title closer to the top, 
    xaxis_title='',
    template='plotly_dark'
)
fig2.update_traces(marker_color=df2['Weighted Average'])



# Second graph
# Filter out players who attended less than 3 tournaments

min_tournaments = 3
filtered_df = df[df['Tournaments Entered'] >= min_tournaments]

# Sort the filtered DataFrame by 'Average Win rate' in descending order
filtered_df = filtered_df.sort_values(by='Average Win rate', ascending=False)

# Select the top 10 players
top_10_players_winrate = filtered_df.head(10)


# Third graph
# Manually enter data
players_data = {
    'Samurai': 3,
    'Eyas': 3,
    'Sophist': 2,
    'CHICO': 1,
    'ICHIGO': 1,
    'Comet': 1,
    'Baskin': 1,
}

# Convert the dictionary to a DataFrame
df_players = pd.DataFrame(list(players_data.items()), columns=['Players', 'Wins'])

# Create a pie chart using plotly.graph_objects
fig3 = go.Figure(data=[go.Pie(labels=df_players['Players'], values=df_players['Wins'], textinfo='label')])

fig3.update_layout(
    title=dict(
        text="Distribution of Tournament Winners",
        x=0.5,  # Set the title's x position to the center of the plot
        xanchor='center',  # Anchor the x position to the center
        font=dict(size=24),
    ),
    showlegend = False,
    template='plotly_dark',
    width=1000,  # Set the desired width
    height=800,  # Set the desired height 
    title_x=0.5,  # Center the title horizontally
    title_y=0.97,  # Position the title closer to the top
)

fig4 = px.scatter(data_filter, x='Tournaments Entered', y='Total Sets Played', hover_data=['Players'], color='Players', trendline="ols")

# Add the regression line directly to the scatter plot trace without showing the legend
fig4.add_trace(px.scatter(data_filter, x='Tournaments Entered', y='Total Sets Played', trendline="ols").data[1])

# Update the layout of the figure with the following details:
# Title: 'Interactive Scatter Plot with Hover Labels, Regression Line, and Player Name Colors'
# x-axis title: 'Total Sets Played'
# y-axis title: 'Sets won'
# Hide the legend
fig4.update_layout(
    title=dict(
        text="Bracket Run Compared to Avrage",
        x=0.5,  # Set the title's x position to the center of the plot
        xanchor='center',  # Anchor the x position to the center
        font=dict(size=24),  # Set the font size
    ),
    xaxis_title='Tournaments Entered',
    yaxis_title='Sets Played',
    showlegend=False,
    template='plotly_dark',
    width=1000,  # Set the desired width
    height=800,  # Set the desired height
    title_x=0.5,  # Center the title horizontally
    title_y=0.97,  # Position the title closer to the top
)

#new fig 5 named fig 55 as replacement 

df_weighted_average = pd.read_csv("my_venv\\Cream City Clash\\Data\\top_players_performance_full_list.csv")
df_sets_per_tournament = pd.read_csv("my_venv\\Cream City Clash\\Data\\sets_per_tournament.csv")

# Merge the DataFrames on a common column, assuming 'Player' is the common column
df_combined = pd.merge(df_weighted_average, df_sets_per_tournament, on='Players', how='inner')

# Drop rows with missing values
df_combined = df_combined.dropna()

# Change the scale of 'Sets per Tournament' to go up by 0.5
df_combined['Sets per Tournament'] = df_combined['Sets per Tournament'] 

# Select the columns for clustering
data_for_clustering = df_combined[['Players', 'Weighted Average', 'Sets per Tournament']]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering[['Weighted Average', 'Sets per Tournament']])

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # You can choose the number of clusters
df_combined['Cluster'] = kmeans.fit_predict(scaled_data)

# Invert the cluster scale
df_combined['Cluster'] = 3 - df_combined['Cluster']


# Create a 3D scatter plot using Plotly Express with dark theme
fig55 = px.scatter_3d(df_combined, x='Players', y='Sets per Tournament', z='Weighted Average',
                    color='Cluster', hover_data=['Players'],
                    title='3D K-Means Clustering',
                    labels={'Weighted Average': 'Weighted Average', 'Sets per Tournament': 'Sets per Tournament'},
                    category_orders={'Players': df_combined['Players']},  # Specify the order of players
                    template='plotly_dark')  # Remove the extra comma here


# Manually set the y-axis range
max_weighted_average = df_combined['Weighted Average'].max()
buffer_space = 1.0  # You can adjust this value based on your preference

fig55.update_layout(
    title=dict(
        text= "Sets Per Tournament and Weighted Average",
        x=0.5,  # Set the title's x position to the center of the plot
        xanchor='center',  # Anchor the x position to the center
        font=dict(size=24),  # Set the font size
    ),
    scene=dict(xaxis=dict(showticklabels=False)),
    width=1000,
    height=800,
    title_x=0.5,  # Center the title horizontally
    title_y=0.97, 
    coloraxis_showscale=False
)

# end of fig 55

wl_file_path = "my_venv\\Cream City Clash\\Data\\REAL Total Sets Played Cream City Clash Fall 2023 H2H REAL.csv"
wl_df = pd.read_csv(wl_file_path, index_col=0)  # Assuming the first column is the index


# Transpose the DataFrame to switch rows and columns
wl_df_transposed = wl_df.T

# Create a heatmap using Plotly Express
fig6 = px.imshow(wl_df, color_continuous_scale='reds', aspect='auto')

# Get the list of player names in the order you want them to appear
players = wl_df_transposed.index.tolist()

# Customize the layout and display the plot
fig6.update_layout(
    title=dict(
        text="Common Opponents Heat Map (3 Tournament Minimum)",
        x=0.5,  # Set the title's x position to the center of the plot
        xanchor='center',  # Anchor the x position to the center
        font=dict(size=24),  # Set the font size
        y=0.97,  # Move the title higher up
    ),
    xaxis_title='Opponent',
    yaxis_title=dict(
        text='Player',
        standoff = 80,  # Adjust the standoff to move the Y label further down
    ),
    yaxis={'categoryorder': 'array', 'categoryarray': players},  # Set category order to display Y-axis labels from the top
    xaxis_side='top',  # Move X-axis labels to the top
    xaxis_tickangle=-45,  # Rotate X-axis labels by 45 degrees counterclockwise
    template='plotly_dark',
    width=1000,  # Set the desired width
    height=800,  # Set the desired height
    title_x=0.5,  # Center the title horizontally
)


# Update the hovertemplate to include columns only from sets_played_df
fig6.update_traces(
    hovertemplate='Player: %{y}<br>Opponent: %{x}<br>Sets Played: %{z}<br>')

wl_file_path = "my_venv\\Cream City Clash\\Data\\REAL Cream City Clash Fall 2023 H2H Data.csv"
wl_df = pd.read_csv(wl_file_path, index_col=0)  # Assuming the first column is the index

# Transpose the DataFrame to switch rows and columns
wl_df_transposed = wl_df.T

# Create a heatmap using Plotly Express
fig7 = px.imshow(wl_df, color_continuous_scale='RdBu_r', aspect='auto')

# Get the list of player names in the order you want them to appear
players = wl_df_transposed.index.tolist()

# Customize the layout and display the plot
fig7.update_layout(
    title=dict(
        text="Set Dominance Heat Map (3 Tournament Minimum)",
        x=0.5,  # Set the title's x position to the center of the plot
        xanchor='center',  # Anchor the x position to the center
        font=dict(size=24),  # Set the font size
    ),
    xaxis_title='Opponent',
    yaxis_title='Player',
    yaxis={'categoryorder': 'array', 'categoryarray': players},  # Set category order to display Y-axis labels from the top
    xaxis_side='top',  # Move X-axis labels to the top
    xaxis_tickangle=-45,  # Rotate X-axis labels by 45 degrees counterclockwise
    template='plotly_dark',
    width=1000,  # Set the desired width
    height=800,  # Set the desired height
    title_x=0.5,  # Center the title horizontally
    title_y=0.97,  # Move the title closer to the top
    yaxis_title_standoff=40  # Adjust the standoff to move the Y-axis title closer to the Y-axis
)


# Update the hovertemplate to include columns only from sets_played_df
fig7.update_traces(
    hovertemplate='Player: %{y}<br>Opponent: %{x}<br>Set Differental: %{customdata}<br>',
    customdata=wl_df.applymap(lambda x: f'+{round(x)}' if np.isfinite(x) and x > 0 else round(x) if np.isfinite(x) else 'N/A').values
)


app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "black",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Insights", style={'text-align': 'center'}, className="display-4"),
        html.Hr(),
        html.P("Diffrent Player Insights from Cream City Clash (Fall 2023)", style={'text-align': 'center'}, className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact", style={'text-align': 'center'}),
                dbc.NavLink("Top 10 Players based on Weighted Average", href="/page-1", active="exact", style={'text-align': 'center'}),
                dbc.NavLink("Bracket Run Compared to Average", href="/page-2", active="exact", style={'text-align': 'center'}),
                dbc.NavLink("3D K-means Clster Sample Graph", href="/page-3", active="exact", style={'text-align': 'center'}),
                dbc.NavLink("Heat map of Common Opponents", href="/page-4", active="exact", style={'text-align': 'center'}),
                dbc.NavLink("Heat map of Set Dominance", href="/page-5", active="exact", style={'text-align': 'center'}),
                dbc.NavLink("Top 10 Most Active Players", href="/page-6", active="exact", style={'text-align': 'center'}),
                dbc.NavLink("Distribution of Tournament Winners", href="/page-7", active="exact", style={'text-align': 'center'}),
                dbc.NavLink("About Me", href="/page-8", active="exact", style={'text-align': 'center'}),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)



content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
server = app.server

        # Inside your 'render_page_content' callback
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([
            html.H1("Cream City Clash Data Visualizations - Fall 2023", style={'text-align': 'center', 'margin-top': '40px', 'font-size': 'px'}),
            html.Img(src=app.get_asset_url('CreamCityClashLogo.jpg'), style={'width': '20%', 'margin-top': '20px', 'border-radius': '10px'}),
            html.P("Using Data Analysis to determine the top 10 players at UWM during the Fall 2023 Semester: The top metrics that were determined to be important are Placements, Win Rate, Head-to-Head results, and Tournament attendance. To qualify for the top 10 a player must attend a minimum of 3 tournaments.",
                style={'margin-top': '20px', 'text-align': 'center', 'font-size': '18px'}
            ),
            html.Div(
                [
                    html.P("Top 10 Players for Fall 2023", style={'font-size': '24px', 'font-weight': 'bold', 'margin-top': '30px'}),
                    html.Ul(
                        [
                            html.P([
    "1.) Samurai ",
    html.Img(src=app.get_asset_url('Ness 5.png'), style={'width': '20px', 'height': '20px'}),
    html.Img(src=app.get_asset_url('Palutena 6.png'), style={'width': '20px', 'height': '20px'})]), 
                           html.P([
    "2.) Eyas ", 
    html.Img(src=app.get_asset_url('Bowser 6.png'), style={'width': '20px', 'height': '20px'}),
    html.Img(src=app.get_asset_url('Dark Samus 2.png'), style={'width': '20px', 'height': '20px'})
]),


                            html.P([
    "3.) Sophist ",
    html.Img(src=app.get_asset_url('King Dedede 3.png'), style={'width': '20px', 'height': '20px'}),
                                ]),
                            html.P([
    "4.) ICHIGO ",
    html.Img(src=app.get_asset_url('Pac-Man 5.png'), style={'width': '20px', 'height': '20px'}),
    html.Img(src=app.get_asset_url('Kazuya 7.png'), style={'width': '20px', 'height': '20px'})
                                ]),
                            html.P([
    "5.) Kaiju ",
    html.Img(src=app.get_asset_url('Ken 3.png'), style={'width': '20px', 'height': '20px'}),
    html.Img(src=app.get_asset_url('Ryu 4.png'), style={'width': '20px', 'height': '20px'})
                                ]),
                            html.P([
    "6.) CHICO ",
    html.Img(src=app.get_asset_url('Hero 3.png'), style={'width': '20px', 'height': '20px'}),
                                ]),
                            html.P([
    "7.) Pidgeon ",
    html.Img(src=app.get_asset_url('Terry 4.png'), style={'width': '20px', 'height': '20px'}),
    ]),
                            html.P([
    "8.) Cal ",
    html.Img(src=app.get_asset_url('Young Link 1.png'), style={'width': '20px', 'height': '20px'}),
    html.Img(src=app.get_asset_url('Dr. Mario 6.png'), style={'width': '20px', 'height': '20px'}),
    ]),
                            html.P([
    "9.) TheDood22 ",
    html.Img(src=app.get_asset_url('Ness 5.png'), style={'width': '20px', 'height': '20px'}),
                                ]),
                            html.P([
    "10.) Shinta ", 
    html.Img(src=app.get_asset_url('Marth 4.png'), style={'width': '20px', 'height': '20px'}),
    html.Img(src=app.get_asset_url('Piranha Plant 0.png'), style={'width': '20px', 'height': '20px'}),
    ]),

    html.P([
    "HM.) Bellyache ", 
    html.Img(src=app.get_asset_url('Yoshi 4.png'), style={'width': '20px', 'height': '20px'}),
    html.Img(src=app.get_asset_url('Mario 0.png'), style={'width': '20px', 'height': '20px'}),
    ]),
                            # Add more featured insights as needed
                        ],
                        style={'text-align': 'center', 'margin-top': '10px', 'font-size': '18px'}
                    ),
                ],
                style={'margin-top': '40px', 'text-align': 'center'}
            )
        ], style={'text-align': 'center'})

        #Weighted average graph

    elif pathname == "/page-1":
        return html.Div([
            centered_graph_div('graph1', fig2, [
            html.P(""),
            html.P("Graph Description:"),
            html.P("The graph visually represents the overall performance of the top 10 players at UWM, taking into account two key factors: average win rate and head-to-head performance. The weighted average is calculated using a formula that assigns a 60% weight to the average win rate and a 40% weight to the head-to-head wins. This approach provides a comprehensive view of a player's performance, considering both individual success (win rate) and performance against specific opponents (head-to-head)."),
            html.P("Key Points:"),
            html.P("Players with higher bars have better overall performance, indicating a stronger combination of win rate and success in head-to-head matchups."),
            html.P("Conclusion:"),
            html.P("The graph serves as a visual representation of showcasing the diffrences in a players weighted average allowing for easy comparison of the top players at UWM.")])
        ])

        #Regression graph

    elif pathname == "/page-2":
        return html.Div([
        centered_graph_div('graph2', fig4, [
            html.P(""),
            html.P("Graph Description:"),
            html.P("The graph displayed in showcases the total number of sets played by each player alongside the number of tournaments they entered. Additionally, a regression line was added to compare players' bracket runs to the average number of sets played, to see how far a player makes it in bracket on average. "),
            html.P("Key Points:"),
            html.P("Players that are placed above the regression line indicate that they go further in tournament compared to an average player. The higher a player is above the regression is correlated with how much further a player will go in tournament on average. For example, Samurai will make further bracket runs on average compared to Kaiju because Samurai is slightly higher on the regression line than Kaiju."),
            html.P("Conclusion:"),
            html.P("This graph depicts how far a player goes in brackets by comparing the total sets played to the number of tournaments that they have entered. This graph was made in effort to identify players that have the furthest bracket runs on average.")
        ])
    ])

        # K-cluster graph 3D

    elif pathname == "/page-3":
        return html.Div([
            centered_graph_div('graph3', fig55, [
            html.P(""),
            html.P("Graph Description:"),
            html.P("The 3D graph visualizes player performance at the UWM Smash Club using K-means clustering, with the x-axis representing Weighted Average, the y-axis representing Sets per Tournament, and the z-axis representing Players. Each player is color-coded based on their assigned cluster, indicating similar performance characteristics within each group."),
            html.P("Key Points:"),
            html.P("Observing deviations from the regression line can highlight players who consistently outperform relative to their average number of sets played. Players who consistently exceed expectations in bracket runs compared to their average sets played will achieve a higher placement on average."),
            html.P("Conclusion:"),
            html.P("The 3D graph and K-means clustering provide valuable insights into player performance at the UWM Smash Club. By categorizing players into distinct clusters, we can better understand the varying skill levels and playing styles within the community. This analysis serves as a foundation for further statistical analysis to identify the top 10 players.")
        ])
        ])
    
        #Total Sets played heat map 

    elif pathname == "/page-4":
        return html.Div([
            centered_graph_div('graph4', fig6, [
            html.P(""),    
            html.P("Graph Description:"),
            html.P("The heat map visualizes the frequency of matchups between different players at Cream City Clash. Each cell in the heatmap represents the number of times two players have competed against each other. The rows and columns represent individual players, and the color intensity of each cell relates to the frequency of matchups between the respective players."),
            html.P("Key Points:"),
            html.P("Clusters and Cliques: Players that have played each other 5 or more times can indicate a strong rivalry at Cream City Clash. The most common matchups at cream city clash are:"),
            html.P("1.) Samurai vs Eyas – 9 sets "),
            html.P("2.) Eyas vs Kaiju – 8 sets"),
            html.P("3.) Sophist vs Ichigo - 8 sets"),
            html.P("4.) Samurai vs Ichigo - 8 sets"),
            html.P("5.) The Dood22 vs goober - 7 sets"),
            html.P("Conclusion:"),
            html.P("This graph depicts how far a player goes in brackets by comparing the total sets played to the number of tournaments that they have entered. This graph was made in effort to identify players that have the furthest bracket runs on average.")
        ])
        ])
    
        #Set Dominance heat map 

    elif pathname == "/page-5":
        return html.Div([
            centered_graph_div('graph5', fig7, [
            html.P(""),
            html.P("Graph Description:"),
            html.P("This heat map visualizes set dominance between different players at Cream City Clash. Each cell in the heatmap represents how dominant a player’s set record is over their opponents. The rows and columns represent individual players, and the color intensity of each cell relates to the set dominance between respective players."),
            html.P("Key Points:"),
            html.P("Players that have a large amount of winning set records demonstrate the ability to succeed in a wide variety of player/character matchups. Players that have winning records in the cells under the boundary line are considered upsets and can show that a certain player may have an “X” factor. "),
            html.P("Conclusion:"),
            html.P("The heat map not only highlights patterns of dominance but also demonstrates instances of unexpected outcomes.")
        ])
        ])
    
        # Tie breaker stat: most tournaments attended

    elif pathname == "/page-6":
        return html.Div([
            centered_graph_div('graph4', fig1,[
            html.P(""),
            html.P("Graph Description:"),
            html.P("This graph shows the players that have attended the most tournaments at Cream City Clash."),
            html.P("Key Points:"),
            html.P("This graph was used as a tie breaker statistic. Players that have attended tournaments often while maintaining similar results as were favored over players that have not attended as many tournaments with similar results."),
            html.P("Conclusion:"),
            html.P("In short, this graph spotlights the players who've shown up the most at Cream City Clash tournaments. It's a tiebreaker tool, helping distinguish between players with similar results. Those who've been regulars got a nod over others with similar performance but fewer appearances.")
        ])
        ])
    
        #Tiebreaker stat: Total amount of cream city clash winners 

    elif pathname == "/page-7":
        return html.Div([
            centered_graph_div('graph7', fig3, [
            html.P(""),
            html.P("Graph Description:"),
            html.P("This graph shows the players that have taken 1st place Cream City Clash."),
            html.P("Key Points:"),
            html.P("This graph was used as a tie breaker statistic. Players that have won tournaments often while maintaining similar results as were favored over players that have not attended as many tournaments with similar results."),
            html.P("Conclusion:"),
            html.P("In short, this graph spotlights the players who've won Cream City Clash tournaments. Much like the player attendance graph, It's a tiebreaker tool, helping distinguish between players with similar results. Those who've been regulars got a nod over others with similar performance but fewer appearances.")
        ])
        ])
    

    elif pathname == "/page-8":
        centered_graph_style = {'max-width': '800px', 'margin': 'auto'}  # Define the centered style
    connect_with_me_style = {'text-align': 'right', 'margin-top': '20px'}  # Align to the right

    return html.Div([
        html.Div([
            html.H2("About Me", style={'text-align': 'center'}, className="display-4"),
            html.Img(src=app.get_asset_url('8ps1wdop.png'), style={'width': '50%', 'margin-top': '20px', 'border-radius': '10px', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
            html.P(
                "I am a Data Science student at the University of Wisconsin-Milwaukee.  I’m passionate about learning all things data science! In this project I used statistical analysis and data visualizations to determine the top 10 players at my schools’ Super Smash Bros Club combining my interests for Smash Bros and Data Science. ",
                style={'text-align': 'center'},
                className="lead"
            ),
            
        ], style=centered_graph_style),

        html.Div([
            html.Hr(),
            html.H4("Connect with me:", style={'text-align': 'right'}),  # Align to the right
            html.Div([
                html.A(html.Img(src=app.get_asset_url('github-mark-white.png'), alt="Github", style={'width': '30px', 'height': '30px'}),
                       href="https://github.com/Eyas-OB"),
                html.Br(),  # Add line break for vertical arrangement
                html.A(html.Img(src=app.get_asset_url("icons8-linkedin-96.png"), alt="Linkedin", style={'width': '32px', 'height': '32px'}),
                       href="https://www.linkedin.com/in/eyas-hamdan-b381092a4"),
                html.Br(),  # Add line break for vertical arrangement
            ], style=connect_with_me_style),
        ], style=centered_graph_style),
    ])

# Helper function to create a centered graph div
def centered_graph_div(graph_id, figure, additional_text):
    return html.Div([
        html.Div([
            dcc.Graph(id=graph_id, figure=figure),
            html.P(additional_text, style={'color': 'white', 'margin-bottom': '20px'})
        ], style={'margin': 'auto', 'width': '80%', 'text-align': 'center'})
    ], style={'margin-top': '20px', 'text-align': 'center'})


if __name__ == "__main__":
    app.run_server(port=8080)

