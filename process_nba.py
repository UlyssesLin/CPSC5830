import pandas as pd
import numpy as np
import json

OUT_DIR = './data/processed/nba'
OUT_EV = f'{OUT_DIR}/events.csv'
OUT_DESC = f'{OUT_DIR}/desc.json'
OUT_EDGE_FEAT = f'{OUT_DIR}/edge_ft.npy'
IN_DIR = './data/raw/nba'

def calctime(time):
    times = time.split(':')
    if len(times) == 2:
        return (60 * int(float(times[0]))) + int(float(times[1]))
    elif len(times) == 1:
        return (60 * int(float(times[0])))
    else:
        print(times)
        return
    
matches = pd.read_csv('./data/raw/nba/games.csv').assign(ts = lambda _d: (pd.to_datetime(_d['game_date']).astype(int) / 10**9).astype(int))
players = pd.read_csv('./data/raw/nba/players.csv')

# matches = matches.loc[matches['season_type'].isin(['Regular Season', ''])]
# players = players.loc[players['gameid'].isin(matches['game_id'])]

matches_cols = ['fgm_home', 'fga_home', 'fg_pct_home', 'fg3m_home', 'fg3a_home',
                'fg3_pct_home', 'ftm_home', 'fta_home', 'ft_pct_home', 'oreb_home',
                'dreb_home', 'reb_home', 'ast_home', 'stl_home', 'blk_home', 'tov_home',
                'pf_home', 'pts_home', 'plus_minus_home', 'fgm_away', 'fga_away', 'fg_pct_away',
                'fg3m_away', 'fg3a_away', 'fg3_pct_away', 'ftm_away', 'fta_away',
                'ft_pct_away', 'oreb_away', 'dreb_away', 'reb_away', 'ast_away',
                'stl_away', 'blk_away', 'tov_away', 'pf_away', 'pts_away',
                'plus_minus_away', 'season_type']
matches_cols_normal = matches_cols[:-1]
players_cols = ['minutes', 'fieldgoalsmade', 'fieldgoalsattempted',
                'fieldgoalspercentage', 'threepointersmade', 'threepointersattempted',
                'threepointerspercentage', 'freethrowsmade', 'freethrowsattempted',
                'freethrowspercentage', 'reboundsoffensive', 'reboundsdefensive',
                'reboundstotal', 'assists', 'steals', 'blocks', 'turnovers',
                'foulspersonal', 'points', 'plusminuspoints']
# matches_cols = ['SEASON', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home',
#                 'AST_home', 'REB_home', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away',
#                 'FG3_PCT_away', 'AST_away', 'REB_away']
# players_cols = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM',
#                 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TO',
#                 'PF', 'PTS', 'PLUS_MINUS']

matchesfeat = matches[['team_id_home', 'team_id_away', 'wl_home', 'ts'] + matches_cols].dropna().drop_duplicates()
players['minutes'] = players['minutes'].fillna(0)
playersfeat = players[['gameid', 'teamid', 'personid'] + players_cols].dropna().drop_duplicates()

timestamps = matches[['game_id', 'ts']].dropna().drop_duplicates()
matches = matches[['team_id_home', 'team_id_away', 'wl_home', 'ts']].dropna().drop_duplicates()
players = players[['gameid', 'teamid', 'personid']].dropna().drop_duplicates()
gamelength = 2880

players = (players
           .rename(columns={'teamid': 'u', 'personid': 'v', 'gameid': 'game_id'})
           .merge(timestamps, on='game_id', how='inner')
           [['u', 'v', 'ts']])

playersfeat = (playersfeat
               .rename(columns={'teamid': 'u', 'personid': 'v', 'gameid': 'game_id'})
               .merge(timestamps, on='game_id', how='inner')
               .drop(columns=['game_id']))

playersfeat['minutes'] = playersfeat['minutes'].astype(str).apply(calctime)
matchesfeat = pd.concat([matchesfeat, pd.get_dummies(matchesfeat['season_type']).astype(int)], axis=1).drop(columns=['season_type'])
matchesfeat[matches_cols_normal] = matchesfeat[matches_cols_normal].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
playersfeat[players_cols] = playersfeat[players_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

matches['wl_home'] = matches['wl_home'].map({'W': 2, 'L': 1})
matches = (matches
           .assign(ts = lambda _d: _d['ts'] + 1)
           .rename(columns={'team_id_home': 'u', 'team_id_away': 'v', 'wl_home': 'e_type'})
           .assign(u_type = 1)
           .assign(v_type = 1)
           .reset_index(drop=True))

players = (players
           .assign(u_type = 1)
           .assign(v_type = 2)
           .assign(e_type = 3)
           .reset_index(drop=True))

matchesfeat['wl_home'] = matchesfeat['wl_home'].map({'W': 1, 'L': 0})
matchesfeat = (matchesfeat
               .assign(ts = lambda _d: _d['ts'] + gamelength + 1)
               .rename(columns={'team_id_home': 'u', 'team_id_away': 'v'})
               .assign(e_type = 4)
               .assign(u_type = 1)
               .assign(v_type = 1)
               .reset_index(drop=True))

playersfeat = (playersfeat
               .assign(ts = lambda _d: _d['ts'] + gamelength + 1)
               .assign(u_type = 1)
               .assign(v_type = 2)
               .assign(e_type = 4)
               .reset_index(drop=True))

teams = pd.concat([matches['u'], matches['v']]).sort_values().unique()
idx_teams = np.arange(len(teams))
teams_dict = {teams[k]: k for k in idx_teams}

player = players['v'].sort_values().unique()
idx_player = np.arange(len(player))
player_dict = {player[k]: k + len(teams) for k in idx_player}

matches = (matches
           .assign(u = lambda _d: _d['u'].map(lambda x: teams_dict[x]))
           .assign(v = lambda _d: _d['v'].map(lambda x: teams_dict[x])))

players = (players
           .assign(u = lambda _d: _d['u'].map(lambda x: teams_dict[x]))
           .assign(v = lambda _d: _d['v'].map(lambda x: player_dict[x])))

matchesfeat = (matchesfeat
           .assign(u = lambda _d: _d['u'].map(lambda x: teams_dict[x]))
           .assign(v = lambda _d: _d['v'].map(lambda x: teams_dict[x])))

playersfeat = (playersfeat
           .assign(u = lambda _d: _d['u'].map(lambda x: teams_dict[x]))
           .assign(v = lambda _d: _d['v'].map(lambda x: player_dict[x])))

pd.DataFrame.from_dict(player_dict, orient='index').to_csv(f'{OUT_DIR}/player_dict.csv')
pd.DataFrame.from_dict(teams_dict, orient='index').to_csv(f'{OUT_DIR}/teams_dict.csv')

events = pd.concat([matches, players]).sort_values('ts').reset_index(drop=True).fillna(0)
# events = pd.concat([matches, players, matchesfeat, playersfeat]).sort_values('ts').reset_index(drop=True).fillna(0)

# e_ft = events.iloc[:, 6:].values
# max_dim = e_ft.shape[1]
# max_dim = max_dim + 4 - (max_dim % 4)
# empty = np.zeros((e_ft.shape[0], max_dim-e_ft.shape[1]))
# e_ft = np.hstack([e_ft, empty])
# e_feat = np.vstack([np.zeros(max_dim), e_ft])
# np.save(OUT_EDGE_FEAT, e_feat)

NUM_NODE = len(player) + len(teams)
NUM_EV = len(events)
NUM_N_TYPE = 2
NUM_E_TYPE = 3
CLASSES = [1, 2]
# NUM_E_TYPE = 4

events = (events
          .assign(e_idx = np.arange(1, NUM_EV + 1))
          [['u', 'v', 'u_type', 'v_type', 'e_type', 'ts', 'e_idx']])

print("num node:", NUM_NODE)
print("num events:", NUM_EV)
events.to_csv('./data/processed/nba/events.csv', index=None)
desc = {
        "num_node": NUM_NODE,
        "num_edge": NUM_EV,
        "num_node_type": NUM_N_TYPE,
        "num_edge_type": NUM_E_TYPE,
        "classes": CLASSES
    }
with open('./data/processed/nba/desc.json', 'w') as f:
    json.dump(desc, f, indent=4)