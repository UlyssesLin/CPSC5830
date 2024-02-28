import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import numpy as np
from pyvis.network import Network
import pyvis

# TODO: Explain UI usage in README
# FOCUS_MATCH = 'ESPORTSTMNT01/1641087' # Match to focus on
# MAX_EVENT_DISPLAY = 50 # custom number of neighbors to display; not impactful if FOCUS_MATCH set
# MAX_EVENT_DISPLAY = None
MAX_EVENTS_TEAMS = 120 # Capped events to display, for ease of viewing - for 2 teams method
MAX_EVENTS_ADJ = 80 # For node adjacency method
# Earliest timestamp: 2014-01-14 17:52:02
# Latest timestamp: 2023-11-20 20:40:26
IFRAME_DIM = 1105 # Width and height of iframe for graph


### ----------------------------------------------------


### GRAPH CREATION SECTION

# Prepare the dataframe for the graph
df = pd.read_csv('data/processed/lol/events_with_gameid.csv')
dfLolTeams = pd.read_csv('data/processed/lol/teams_with_names.csv')
dfLolPlayers = pd.read_csv('data/processed/lol/players_with_names.csv')

df = pd.merge(df, dfLolTeams[['team_num', 'teamname']], how='left', left_on='u', right_on='team_num')
df = df.rename(columns={'teamname': 'u_name'})
df = df.drop('team_num', axis=1)
df_1 = df[df['v_type'] == 1]
df_2 = df[df['v_type'] == 2]

# Get team names in for v
df_1 = pd.merge(df_1, dfLolTeams[['team_num', 'teamname']], how='left', left_on='v', right_on='team_num')
df_1 = df_1.drop('team_num', axis=1)
df_1 = df_1.rename(columns={'teamname': 'v_name'})
# Get player names in for v
df_2 = pd.merge(df_2, dfLolPlayers[['player_num', 'playername']], how='left', left_on='v', right_on='player_num')
df_2 = df_2.drop('player_num', axis=1)
df_2 = df_2.rename(columns={'playername': 'v_name'})

df_recombined = pd.concat([df_1, df_2])
df = df_recombined.sort_values('e_idx')
df['ts'] = pd.to_datetime(df['ts'], unit='s')


# Create graph after getting input from Streamlit UI
def createGraph(teamA, teamB, nodeType, findAdj, maxEventDisplay, unifiedEdges):
    print('=================CREATING GRAPH=================')
    if nodeType and findAdj:
        focusNodes = [
            {
                'name': findAdj,
                'nodeList': [],
                'edgeList': []
            }
        ]
    elif teamA and teamB:
        focusNodes = [
            {
                'name': teamA,
                'nodeList': [],
                'edgeList': []
            },
            {
                'name': teamB,
                'nodeList': [],
                'edgeList': []
            }
        ]

    nodeLabelMapper = {
        1: '  Team:  \n',
        2: '  Player:  \n'
    }

    edgeLabelMapper = {
        1: 'Lost',
        2: 'Won',
        3: 'Joined',
        4: 'Info'
    }

    # Keys the same as node type
    # (1) team
    # (2) player
    nodeSizeMapper = {
        1: 30,
        2: 15
    }

    nodeColorMapper = {
        1: '#000099',
        2: '#0099FF'
    }

    # Keys the same as edge_type
    # (1) Lost
    # (2) Won
    # (3) Played
    # (4) Game info
    edgeColorMapper = {
        1: '#FF0000',
        2: '#15B01A',
        3: '#0099FF',
        4: '#C0C0C0'
    }

    edgeFontSizeMapper = {
        1: 24,
        2: 24,
        3: 12,
        4: 10
    }

    def edgeWeightMapper(type, ts):
        if unifiedEdges:
            if type == 1 or type == 2:
                return len(ts) * 3
            elif type == 3:
                # return 1 + (len(ts) % 2)
                return len(ts)
            return 1
        else:
            if type == 1 or type == 2:
                return ts * 3
            else:
                return 1 + (ts % 10)
        
    def reformatLabel(text):
        return text.replace(' ', '\n')

    def fontAdjuster(type, text):
        if type == 1:
            return 24 if len(text) < 20 else 20
        else:
            return 18 if len(text) < 20 else 14

    def edgeTitleMaker(tsList):
        tsList = pd.to_datetime(tsList).strftime('%h %d %Y %H:%M:%S').tolist()
        concat = '\n'.join(str(ts) for ts in tsList)
        return concat

    for focusNode in focusNodes:
        if (nodeType == 'team') or (teamA and teamB):
            selectedTeam = df.loc[
                (df['u'] == focusNode['name']) | ((df['v'] == focusNode['name']) & (df['v_type'] == 1)),
                ['u', 'v', 'u_type', 'v_type', 'e_type', 'u_name', 'v_name', 'ts']]
        elif nodeType == 'player':
            selectedTeam = df.loc[
                (df['v'] == focusNode['name']) & (df['v_type'] == 2),
                ['u', 'v', 'u_type', 'v_type', 'e_type', 'u_name', 'v_name', 'ts']]
        if maxEventDisplay:
            selectedTeam = selectedTeam.head(maxEventDisplay)
        teamTypeDF = selectedTeam.drop_duplicates()
        us = list(teamTypeDF[['u', 'u_type', 'u_name', 'ts']].itertuples(index=False, name=None))
        vs = list(teamTypeDF[['v', 'v_type', 'v_name', 'ts']].itertuples(index=False, name=None))
        nodesList = list(list(dict.fromkeys(us + vs)))
        nodesList = list(map(lambda x: (x[0], {
            'type': x[1],
            'label': reformatLabel(x[2]),
            'title': 'TODO',
            'margin': 15,
            'color': {
                'background': nodeColorMapper[x[1]], 
                'highlight': {
                    'background': nodeColorMapper[x[1]],
                    'border': 'magenta'
                },
                'hover': {
                    'border': 'gray'
                }
            },
            # 'physics': False,
            # 'mass': 8 if x[1] == 1 else 2,
            'mass': 5,
            'shape': 'circle',
            'font': {'size': fontAdjuster(x[1], x[2]), 'color': 'white'},
            'borderWidthSelected': 6
            }), nodesList))
        edgesWithDups = selectedTeam.groupby(selectedTeam.columns.tolist(), as_index=False).size()
        if unifiedEdges:
            edgesNoDupsDF = edgesWithDups.groupby(['u', 'v', 'e_type'])
            edgesNoDupsList = []
            for ed, group in edgesNoDupsDF:
                edgesNoDupsList.append(ed + (group['ts'].tolist(),))
            edgesNoDupsList = list(map(lambda x: (int(x[1] if x[2] == 3 else x[0]), int(x[0] if x[2] == 3 else x[1]), { 
                'edge_type': int(x[2]), 
                'weight': edgeWeightMapper(x[2], x[3]),
                'label': edgeLabelMapper[x[2]],
                'title': edgeTitleMaker(x[3]),
                'color': edgeColorMapper[x[2]],
                'font': {'size': edgeFontSizeMapper[x[2]]},
                'smooth': 0,
                # 'arrowSize': 5,
                # 'physics': False
                }), edgesNoDupsList))
        else:
            edgesWithDupsList = list(edgesWithDups[['u', 'v', 'e_type', 'size', 'ts']].itertuples(index=False, name=None))
            # Change arrow display direction if edge type is 3 (to show player joining team):
            edgesWithDupsList = list(map(lambda x: (x[1] if x[2] == 3 else x[0], x[0] if x[2] == 3 else x[1], { 
                'edge_type': x[2], 
                'weight': edgeWeightMapper(x[2], x[3]),
                'label': edgeLabelMapper[x[2]],
                'title': str(pd.to_datetime(str(x[4])).strftime('%h %d %Y %H:%M:%S')),
                'color': edgeColorMapper[x[2]],
                'font': {'size': edgeFontSizeMapper[x[2]]},
                'smooth': True,
                # 'arrowSize': 5,
                # 'physics': False
                }), edgesWithDupsList))
        focusNode['nodeList'] = nodesList
        focusNode['edgeList'] = edgesNoDupsList if unifiedEdges else edgesWithDupsList
        focusGraph = nx.MultiDiGraph()

        lineWidthMapper = {
            1: 6,
            2: 6,
            3: 3,
            4: 1
        }

        arrowSizeMapper = {
            1: 10,
            2: 40,
            3: 12,
            4: 1
        }

        allNodesList = focusNodes[0]['nodeList'] + ([] if findAdj else focusNodes[1]['nodeList'])
        allEdgesList = focusNodes[0]['edgeList'] + ([] if findAdj else focusNodes[1]['edgeList'])

        focusGraph.add_nodes_from(allNodesList)
        focusGraph.add_edges_from(allEdgesList)

        options = {
            'arrowstyle': '->',
            'arrowsize': list(arrowSizeMapper[edge_type] for u, v, edge_type in list(focusGraph.edges(data='edge_type')))
        }

        net = Network(
            '1000px', '1000px',
            directed=True,
            # heading='League of Legends',
            # select_menu=True,
            # filter_menu=True,
            # bgcolor='#222222',
            # font_color='white'
        )
        net.from_nx(focusGraph) # Create directly from nx graph

        options = {
            # 'physics':{ # physics very distracting with large/non-capped number of edges, even with just 2 teams
            #     'barnesHut':{
            #         'gravitationalConstant': -15000, # seemingly best around -15000
            #         'centralGravity': 5, # seemingly best at 5
            #         'springLength': 250 if (teamA and teamB) else 400 if nodeType == 'team' else 600,
            #         'springConstant': 0.7,
            #         'damping': 3,
            #         'avoidOverlap': 0 # higher vals push nodes away from each other actively
            #     }
            # },
            'interaction':{   
                'selectConnectedEdges': True,
                'hover': True
            },
            'edges': {
                'arrowStrikethrough': False
            }
        }
        if unifiedEdges:
            options['physics'] = { # physics should be added with 'player' otherwise there is not enough edge length
                'barnesHut':{
                    'gravitationalConstant': -15000, # seemingly best around -15000
                    'centralGravity': 5, # seemingly best at 5
                    'springLength': 250 if (teamA and teamB) else 400 if nodeType == 'team' else 600,
                    'springConstant': 0.7,
                    'damping': 3,
                    'avoidOverlap': 0 # higher vals push nodes away from each other actively
                }
            }
        if nodeType == 'player':
            options['physics'] = { # physics should be added with 'player' otherwise there is not enough edge length
                'barnesHut':{
                    'gravitationalConstant': -15000, # seemingly best around -15000
                    'centralGravity': 5, # seemingly best at 5
                    'springLength': 250 if (teamA and teamB) else 400 if nodeType == 'team' else 600,
                    'springConstant': 0.7,
                    'damping': 3,
                    'avoidOverlap': 0 # higher vals push nodes away from each other actively
                }
            }

        net.options=options

        # net.show('LolGraph.html', notebook=False, ) # Renders to html file and opens in Chrome; do NOT remove the notebook=False
        net.save_graph('LolGraph.html')


###-------------------------------------------------------------------------------


### STREAMLIT UI SECTION
        
styl = f"""
    <style>
        .st-emotion-cache-1y4p8pa{{
            max-width: none !important;
        }}
        .card{{
            border: none !important;
        }}
    </style>
    """

def renderGraph():
    HtmlFile = open('LolGraph.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=IFRAME_DIM, width=IFRAME_DIM-103)

st.markdown(styl, unsafe_allow_html=True)
st.title('League of Legends')
demoCol1, demoCol2 = st.sidebar.columns(2)
useDemoTeams = demoCol1.button('Demo Teams')
useDemoTimes = demoCol2.button('Demo Times')
mainOption = st.sidebar.selectbox('Choose a graph focus method:', ('With 2 teams', 'Node adjacency'))

# TODO: Flesh out demos
if useDemoTeams:
    st.sidebar.text('Demo Teams:\nNongshim RedForce Academy (Home)\nDRX Academy (Away)')
    teamAOption = 'Fnatic --- Team: 827'
    teamBOption = 'Gambit Gaming --- Team: 391'

if mainOption == 'With 2 teams':
    with st.sidebar.form(key='my_form'):
        st.title('You may select:')
        st.markdown('* Two teams\n * OR - Two teams and a match between them\n * OR - Two teams and a time window')
        dfLolTeams = pd.read_csv('data/processed/lol/teams_with_names.csv')
        dfLolTeams = dfLolTeams.sort_values('teamname')
        dfLolTeams['displayname'] = dfLolTeams['teamname'] + ' --- Team: ' + dfLolTeams['team_num'].astype(str)
        listLolTeams = ['No selection'] + dfLolTeams['displayname'].tolist()

        # st.empty() workaround for updating fields before submitting
        # Order here determines order of components on page
        st.subheader('Choose the two teams here:')
        placeholder_Team_A_component = st.empty()
        placeholder_Team_A_react = st.empty()
        placeholder_Team_B_component = st.empty()
        placeholder_Team_B_react = st.empty()
        placeholder_toggle = st.empty()
        placeholder_toggle_react = st.empty()
        st.subheader('Or, input team numbers here:')
        team_A_text_col, team_B_text_col = st.columns(2)
        with team_A_text_col:
            placeholder_Team_A_text_component = st.empty()
            placeholder_Team_A_text_react = st.empty()
        with team_B_text_col:
            placeholder_Team_B_text_component = st.empty()
            placeholder_Team_B_text_react = st.empty()
        st.divider()
        st.subheader('Choose a match between the teams:')
        placeholder_match_component = st.empty()
        placeholder_match_react = st.empty()
        st.divider()
        st.subheader('Input a time window (Optional):')
        st.caption('Time windows are mutually exclusive with match selection; if both are submitted, the graph will filter by match.')
        time_X_text_col, time_Y_text_col = st.columns(2)
        with time_X_text_col:
            time_X_text = st.text_input('Start:')
        with time_Y_text_col:
            time_Y_text = st.text_input('End:')
        placeholder_unified_edges_component = st.empty()
        placeholder_unified_edges_react = st.empty()

        teamAOption=''
        teamBOption=''
        truncTeamA=''
        truncTeamB=''
        chosenTeamMatch=''
        listChosenTeamMatches = ['No selection']
        unifiedEdges = False
        
        submit_button = st.form_submit_button(label='Submit')

    with placeholder_Team_A_component:
        teamAOption = st.selectbox('Team A:', listLolTeams, key='teamA')

    with placeholder_Team_A_react:
        print('Team A selected')

    with placeholder_Team_B_component:
        teamBOption = st.selectbox('Team B:', listLolTeams, key='teamB')

    with placeholder_Team_B_react:
        print('Team B selected')

    with placeholder_Team_A_text_component:
        Team_A_text = st.text_input('Team A:')

    with placeholder_Team_A_text_react:
        print('Team A text inputted')

    with placeholder_Team_B_text_component:
        Team_B_text = st.text_input('Team B:')

    with placeholder_Team_B_text_react:
        print('Team B text inputted')

    with placeholder_match_component:
        print('placeholder_match_component')
        textInputed = Team_A_text and Team_B_text
        selectBoxesChosen = (teamAOption != 'No selection') and (teamBOption != 'No selection')
        if textInputed:
            truncTeamA = int(Team_A_text)
            truncTeamB = int(Team_B_text)
        elif selectBoxesChosen:
            truncTeamA = int(teamAOption.split(' --- Team: ')[1])
            truncTeamB = int(teamBOption.split(' --- Team: ')[1])
        
        if textInputed or selectBoxesChosen:
            print('team A: ' + str(truncTeamA))
            print('team B: ' + str(truncTeamB))
            teams_df = df.loc[(df['u'] == truncTeamA) | (df['v'] == truncTeamB)]
            dfGames = df.loc[(df['u'].isin([truncTeamA, truncTeamB])) & (df['v'].isin([truncTeamA, truncTeamB])) & (df['e_type'].isin([1, 2]))] # should be sorted by date
            listGames = list(dfGames[['u', 'v', 'ts', 'gameid']].itertuples(index=False))
            listChosenTeamMatches = ['No selection']
            for game in listGames:
                listChosenTeamMatches.append(game[3] + ' : ' + str(game[2]) + ' : ' + str(game[0]) + ' (H) vs ' + str(game[1]) + ' (A)')
        chosenTeamMatch = st.selectbox('(Optional) Choose a match between those teams:', options=listChosenTeamMatches, key='chosenTeamMatch')

    with placeholder_match_react:
        print('Chosen teams match')
        if chosenTeamMatch == 'No selection':
            print('No selection')

    with placeholder_unified_edges_component:
        unifiedEdges = st.toggle('Show unified edges', key='toggle1')

    with placeholder_unified_edges_react:
        print('Unified edges toggled')

    # with st.sidebar:
    if submit_button:
        if truncTeamA and truncTeamB:
            dfOriginal = df
            print('--------------USER SUBMITTED 2 TEAMS FORM--------------')
            print('TEAM A: ' + str(truncTeamA))
            print('TEAM B: ' + str(truncTeamB))

            # Time window selection
            if chosenTeamMatch != 'No selection':
                chosenTeamMatch = chosenTeamMatch.split(' : ')[0]
                print('Selected match: ' + str(chosenTeamMatch))
                df = df.loc[(df['gameid'] == chosenTeamMatch)]
            elif time_X_text and time_Y_text:
                print('Inputed time window: ' + str(time_X_text) + ' --> ' + str(time_Y_text))
                time_X_text = np.datetime64(time_X_text)
                time_Y_text = np.datetime64(time_Y_text)
                df = df.loc[(df['ts'] >= time_X_text) & (df['ts'] <= time_Y_text)]
                print('SELECTING TIME WINDOW BETWEEN ' + str(time_X_text) + ' AND ' + str(time_Y_text))

            createGraph(truncTeamA, truncTeamB, None, None, MAX_EVENTS_TEAMS, unifiedEdges)
            df = dfOriginal # reset - after submit and show graph
            renderGraph()
        else:
            print('--------------TEAMS NOT CHOSEN--------------')



if mainOption == 'Node adjacency':
    with st.sidebar.form(key='my_form2'):
        st.title('Select the type of node:')
        # st.empty() workaround for updating fields before submitting
        # Order here determines order of components on page
        placeholder_node_typ_component = st.empty()
        placeholder_node_typ_react = st.empty()
        st.divider()
        st.subheader('Then choose a node:')
        placeholder_node_list_component = st.empty()
        adjNodeText = st.text_input('Or, input a node number (after choosing a type):')
        placeholder_unified_edges_component = st.empty()
        placeholder_unified_edges_react = st.empty()

        submit_button = st.form_submit_button(label='Submit')

        nodesList = ['No selection']
        adjNode = ''
        unifiedEdges = False
        
    with placeholder_node_typ_component:
        adjNodeType = st.selectbox('Select one:', ['Team', 'Player'], key='adjtype')

    with placeholder_node_typ_react:
        print('Type <' + adjNodeType + '> selected')
        if adjNodeType == 'Team':
            adjNodeType = 'team'
            dfNodesList = pd.read_csv('data/processed/lol/teams_with_names.csv').sort_values('teamname')
        else:
            adjNodeType = 'player'
            dfNodesList = pd.read_csv('data/processed/lol/players_with_names.csv').sort_values('playername')
        dfNodesList['displayname'] = dfNodesList[adjNodeType + 'name'] + ' --- Number: ' + dfNodesList[adjNodeType + '_num'].astype(str)
        nodesList = nodesList + dfNodesList['displayname'].tolist()

    with placeholder_node_list_component:
        adjNodeSelection = st.selectbox('Select a node:', nodesList, key='adjlist')

    with placeholder_unified_edges_component:
        unifiedEdges = st.toggle('Show unified edges', key='toggle2')

    with placeholder_unified_edges_react:
        print('Unified edges toggled')

    if submit_button:
        if adjNodeText:
            adjNode = int(adjNodeText)
        elif adjNodeSelection != 'No selection':
            adjNode = int(adjNodeSelection.split(' --- Number: ')[1])

        if adjNodeType and (adjNodeText or adjNodeSelection != 'No selection'):
            print('--------------USER SUBMITTED NODE ADJACENCY FORM--------------')
            # dfOriginal = df
            print('<' + adjNodeType + '> type node, number: ' + str(adjNode))
            createGraph(None, None, adjNodeType, adjNode, MAX_EVENTS_ADJ, unifiedEdges)
            # df = dfOriginal # reset - after submit and show graph
            renderGraph()
        else:
            print('--------------FAILED NODE ADJACENCY FORM--------------')