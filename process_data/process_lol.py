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
TEAM_COL_APPEND = ['teamname'] # any team cols needed for viz labeling
PLAYER_COL_APPEND = ['playername'] # any player cols needed for viz labeling

all_files = glob.glob(os.path.join(IN_DIR, '*.csv'))
df = (pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
      .assign(ts = lambda _d: (pd.to_datetime(_d['date']).astype('int64') / 10**9).astype('int')))

team_cols = ['patch', 'towers', 'barons', 'inhibitors', 'dragons', 'opp_towers', 'opp_barons', 'opp_inhibitors', 'opp_dragons']
player_cols = ['position', 'champion', 'kills', 'deaths', 'assists', 'firstblood', 'damagetochampions', 'wardsplaced',
              'wardskilled', 'controlwardsbought', 'earnedgold', 'total cs', 'golddiffat10', 'xpdiffat10', 'csdiffat10',
              'killsat10', 'assistsat10', 'deathsat10', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']

team_cols.extend(TEAM_COL_APPEND)
player_cols.extend(PLAYER_COL_APPEND)

matches_normal_cols = team_cols
player_normal_cols = player_cols[2:]

home = df.loc[df['side'] == 'Blue', ['teamid', 'gameid', 'result', 'ts', 'gamelength']].dropna().drop_duplicates()
away = df.loc[df['side'] == 'Red', ['teamid', 'gameid']].dropna().drop_duplicates()

matches = home.merge(away, how='inner', on='gameid')
matches = matches.loc[matches['teamid_x'] != matches['teamid_y']]
players = df.loc[df['gameid'].isin(matches['gameid'].unique()), ['teamid', 'playerid', 'ts', 'gameid']].dropna().drop_duplicates()
homefeat = df.loc[df['side'] == 'Blue', ['teamid', 'gameid', 'result', 'ts', 'gamelength'] + team_cols].dropna().drop_duplicates()
awayfeat = df.loc[df['side'] == 'Red', ['teamid', 'gameid']].dropna().drop_duplicates()
matchesfeat = homefeat.merge(awayfeat, how='inner', on='gameid')
playersfeat = df.loc[df['gameid'].isin(matchesfeat['gameid'].unique()), ['teamid', 'playerid', 'ts', 'gamelength', 'gameid'] + player_cols].dropna().drop_duplicates()
print('playersfeat:')
print(playersfeat.columns)
matches = (matches
           .assign(ts = lambda _d: _d['ts'] + 1)
           .rename(columns={'teamid_x': 'u', 'teamid_y': 'v', 'result': 'e_type'})
           .assign(e_type = lambda _d: _d['e_type'] + 1)
           .assign(u_type = 1)
           .assign(v_type = 1)
           [['u', 'v', 'ts', 'e_type', 'u_type', 'v_type', 'gameid']]
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
               .drop(columns=['gamelength'])
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

for name_type in ['team', 'player']:
    isTeam = name_type == 'team'
    currNum = name_type + '_num'
    currName = name_type + 'name'
    curr_df = pd.DataFrame.from_dict(teams_dict if isTeam else player_dict, orient='index')
    curr_df['long_' + name_type] = curr_df.index
    curr_df = curr_df.rename(columns={0: currNum})
    if isTeam:
        right_df = matchesfeat.loc[:, ['u', 'gameid'] + TEAM_COL_APPEND].drop_duplicates()
    else:
        right_df = playersfeat.loc[:, ['v', 'gameid'] + PLAYER_COL_APPEND].drop_duplicates(subset='v')
    curr_df = pd.merge(curr_df, right_df, how='left', left_on='long_' + name_type, right_on='u' if isTeam else 'v')
    curr_df = curr_df.loc[:, ['long_' + name_type, currNum, currName]].drop_duplicates()
    curr_df[currName] = curr_df[currName].fillna('No Name')
    curr_df.to_csv(f'{OUT_DIR}/' + name_type + 's_with_names.csv')

pd.DataFrame.from_dict(player_dict, orient='index').to_csv(f'{OUT_DIR}/player_dict.csv')
pd.DataFrame.from_dict(teams_dict, orient='index').to_csv(f'{OUT_DIR}/teams_dict.csv')

# Reset modded data for viz
for appended in TEAM_COL_APPEND:
    matchesfeat = matchesfeat.drop(appended, axis=1)
    matches_normal_cols.remove(appended)
for appended in PLAYER_COL_APPEND:
    playersfeat = playersfeat.drop(appended, axis=1)
    player_normal_cols.remove(appended)

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
np.save(OUT_EDGE_FEAT, e_feat)

# events = pd.concat([matches, players]).sort_values('ts').reset_index(drop=True)

NUM_NODE = len(player) + len(teams)
NUM_EV = len(events)
NUM_N_TYPE = 2
CLASSES = [1, 2]
# NUM_E_TYPE = 3
NUM_E_TYPE = 4

events_with_gameid = (events
          .assign(e_idx = np.arange(1, NUM_EV + 1))
          [['u', 'v', 'u_type', 'v_type', 'e_type', 'ts', 'e_idx', 'gameid']])
events = events_with_gameid.drop('gameid', axis=1)
print(events_with_gameid.shape)
# events_with_gameid = events_with_gameid.drop(events_with_gameid[events_with_gameid['gameid'] == 0].index)

events_with_gameid.to_csv(f'{OUT_DIR}/events_with_gameid.csv')

print("num node:", NUM_NODE)
print("num events:", NUM_EV)
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