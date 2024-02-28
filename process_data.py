import pandas as pd
import numpy as np
import json
import glob
import os

import category_encoders as ce

OUT_DIR = './data/processed/lol'
OUT_EV = f'{OUT_DIR}/events.csv'
OUT_DESC = f'{OUT_DIR}/desc.json'
OUT_EDGE_FEAT = f'{OUT_DIR}/edge_ft.npy'
IN_DIR = './data/raw/lol'

all_files = glob.glob(os.path.join(IN_DIR, '*.csv'))
df = (pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
      .assign(ts = lambda _d: (pd.to_datetime(_d['date']).astype(int) / 10**9).astype('int64')))

team_cols = ['patch', 'towers', 'barons', 'inhibitors', 'dragons', 'opp_towers', 'opp_barons', 'opp_inhibitors', 'opp_dragons']
player_cols = ['position', 'champion', 'kills', 'deaths', 'assists', 'firstblood', 'damagetochampions', 'wardsplaced',
              'wardskilled', 'controlwardsbought', 'earnedgold', 'total cs', 'golddiffat10', 'xpdiffat10', 'csdiffat10',
              'killsat10', 'assistsat10', 'deathsat10', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']

matches_normal_cols = team_cols
player_normal_cols = player_cols[2:]

home = df.loc[df['side'] == 'Blue', ['teamid', 'gameid', 'result', 'ts', 'gamelength']].dropna().drop_duplicates()
away = df.loc[df['side'] == 'Red', ['teamid', 'gameid']].dropna().drop_duplicates()

matches = home.merge(away, how='inner', on='gameid')
matches = matches.loc[matches['teamid_x'] != matches['teamid_y']]
players = df.loc[df['gameid'].isin(matches['gameid'].unique()), ['teamid', 'playerid', 'ts']].dropna().drop_duplicates()

homefeat = df.loc[df['side'] == 'Blue', ['teamid', 'gameid', 'result', 'ts', 'gamelength'] + team_cols].dropna().drop_duplicates()
awayfeat = df.loc[df['side'] == 'Red', ['teamid', 'gameid']].dropna().drop_duplicates()
matchesfeat = homefeat.merge(awayfeat, how='inner', on='gameid')
playersfeat = df.loc[df['gameid'].isin(matchesfeat['gameid'].unique()), ['teamid', 'playerid', 'ts', 'gamelength'] + player_cols].dropna().drop_duplicates()

matches = (matches
           .assign(ts = lambda _d: _d['ts'] + 1)
           .rename(columns={'teamid_x': 'u', 'teamid_y': 'v', 'result': 'e_type'})
           .assign(e_type = lambda _d: _d['e_type'] + 1)
           .assign(u_type = 1)
           .assign(v_type = 1)
           [['u', 'v', 'ts', 'e_type', 'u_type', 'v_type']]
           .reset_index(drop=True))

players = (players
           .assign(e_type = 3)
           .assign(u_type = 1)
           .assign(v_type = 2)
           .rename(columns={'teamid': 'u', 'playerid': 'v'})
           .reset_index(drop=True))

matchesfeat = (matchesfeat
               .assign(ts = lambda _d: _d['ts'] + _d['gamelength'] + 1)
               .rename(columns={'teamid_x': 'u', 'teamid_y': 'v', 'result': 'e_type'})
               .assign(e_type = 4)
               .assign(u_type = 1)
               .assign(v_type = 1)
               .drop(columns=['gameid', 'gamelength'])
               .reset_index(drop=True))

playersfeat = (playersfeat
               .assign(ts = lambda _d: _d['ts'] + _d['gamelength'] + 1)
               .assign(e_type = 4)
               .assign(u_type = 1)
               .assign(v_type = 2)
               .rename(columns={'teamid': 'u', 'playerid': 'v'})
               .drop(columns=['gamelength'])
               .reset_index(drop=True))

teams = pd.concat([matches['u'], matches['v']]).sort_values().unique()
idx_teams = np.arange(len(teams))
teams_dict = {teams[k]: k for k in idx_teams}

player = players['v'].sort_values().unique()
idx_player = np.arange(len(player))
player_dict = {player[k]: k + len(teams) for k in idx_player}

pd.DataFrame.from_dict(player_dict, orient='index').to_csv(f'{OUT_DIR}/player_dict.csv')
pd.DataFrame.from_dict(teams_dict, orient='index').to_csv(f'{OUT_DIR}/teams_dict.csv')

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

matchesfeat[matches_normal_cols] = matchesfeat[matches_normal_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
playersfeat[player_normal_cols] = playersfeat[player_normal_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

encoder = ce.BinaryEncoder(cols=['champion', 'position'], return_df=True)
encoded_df = encoder.fit_transform(playersfeat[['champion', 'position']])

playersfeat = (pd.concat([playersfeat, encoded_df], axis=1).drop(columns=['champion', 'position']))

events = pd.concat([matches, players, matchesfeat, playersfeat]).sort_values('ts').reset_index(drop=True).fillna(0)

e_ft = events.iloc[:, 6:].values
max_dim = e_ft.shape[1]
max_dim = max_dim + 4 - (max_dim % 4)
empty = np.zeros((e_ft.shape[0], max_dim-e_ft.shape[1]))
e_ft = np.hstack([e_ft, empty])
e_feat = np.vstack([np.zeros(max_dim), e_ft])

NUM_NODE = len(player) + len(teams)
NUM_EV = len(events)
NUM_N_TYPE = 2
NUM_E_TYPE = 4
CLASSES = [1, 2]

events = (events
          .assign(e_idx = np.arange(1, NUM_EV + 1))
          [['u', 'v', 'u_type', 'v_type', 'e_type', 'ts', 'e_idx']])

print("num node:", NUM_NODE)
print("num events:", NUM_EV)
np.save(OUT_EDGE_FEAT, e_feat)
events.to_csv(OUT_EV, index=None)
desc = {
        "num_node": NUM_NODE,
        "num_edge": NUM_EV,
        "num_node_type": NUM_N_TYPE,
        "num_edge_type": NUM_E_TYPE,
        "classes": CLASSES
    }

with open(OUT_DESC, 'w') as f:
    json.dump(desc, f, indent=4)