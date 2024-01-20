import pandas as pd
import numpy as np
import json

OUT_DIR = './data/processed/lol'
OUT_EV = f'{OUT_DIR}/events.csv'
OUT_DESC = f'{OUT_DIR}/desc.json'
df = (pd.read_csv('data/raw/lol/2023_LoL_esports_match_data_from_OraclesElixir.csv')
      .assign(ts = lambda _d: (pd.to_datetime(_d['date']).astype(int) / 10**9).astype(int)))
matches = (df[['teamid', 'gameid', 'result', 'ts', 'gamelength']]
           .dropna()
           .drop_duplicates())
players = (df[['teamid', 'playerid', 'ts']]
           .dropna()
           .drop_duplicates())

matches = (matches
           .assign(ts = lambda _d: _d['ts'] + _d['gamelength'])
           .assign(u_type = 1)
           .assign(v_type = 2)
           .drop(columns=['gamelength'])
           .rename(columns={'teamid': 'u', 'gameid': 'v', 'result': 'e_type'})
           .reset_index(drop=True))

players = (players
           .assign(u_type = 1)
           .assign(v_type = 3)
           .assign(e_type = 2)
           .rename(columns={'teamid': 'u', 'playerid': 'v'})
           .reset_index(drop=True))

events = pd.concat([matches, players]).sort_values('ts').reset_index(drop=True)

u = None
for utype in events['u_type'].unique():
    if u is None:
        u = events.loc[events['u_type'] == utype,'u'].sort_values().unique()
    else:
        t_u = events.loc[events['u_type'] == utype,'u'].sort_values().unique()
        u = np.concatenate([u, t_u])
idx_u = np.arange(len(u))
u_dict = {u[k]: k for k in idx_u}
NUM_N_U = len(u)

v = None
for vtype in events['v_type'].unique():
    if v is None:
        v = events.loc[events['v_type'] == vtype,'v'].sort_values().unique()
    else:
        t_v = events.loc[events['v_type'] == vtype,'v'].sort_values().unique()
        v = np.concatenate([v, t_v]) 
idx_v = np.arange(len(v))
v_dict = {v[k]: k for k in idx_v}
NUM_N_V = len(v)

NUM_NODE = NUM_N_U + NUM_N_V
NUM_EV = len(events)
NUM_N_TYPE = 3
NUM_E_TYPE = 3

events = (events
          .assign(u = lambda _d: _d['u'].map(lambda x: u_dict[x]))
          .assign(v = lambda _d: _d['v'].map(lambda x: v_dict[x]))
          .assign(e_type = lambda _d: _d['e_type'] + 1)
          .assign(e_idx = np.arange(1, NUM_EV + 1)))

events.v += NUM_N_U

print("num node:", NUM_NODE)
print("num events:", NUM_EV)
events.to_csv(OUT_EV, index=None)
desc = {
        "num_node": NUM_NODE,
        "num_edge": NUM_EV,
        "num_node_type": NUM_N_TYPE,
        "num_edge_type": NUM_E_TYPE,
        "num_node_u": NUM_N_U,
        "num_node_v": NUM_N_V,
    }
with open(OUT_DESC, 'w') as f:
    json.dump(desc, f, indent=4)