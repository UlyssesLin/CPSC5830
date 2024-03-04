import pandas as pd
import numpy as np
import json
import glob
import os

all_files = glob.glob(os.path.join('./data/raw/flight/', '*.csv'))
df = (pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True))

df = df.sample(200000)

df['FL_DATE'] = pd.to_datetime(df['FL_DATE']).dt.date
df['CRS_DEP_TIME'] = pd.to_datetime(df['CRS_DEP_TIME'].astype(str).str.zfill(4), format='%H%M').dt.time
df['ts'] = (pd.to_datetime(df['FL_DATE'].astype(str) + ' ' + df['CRS_DEP_TIME'].astype(str)).astype(int) / 10**9).astype('int64')

df = df.dropna().sort_values('ts').reset_index(drop=True).reset_index()[['index', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'DEST', 'DEP_DELAY', 'ts']]

delay = df[['ORIGIN', 'DEST', 'DEP_DELAY', 'ts']]
carrier = df[['ORIGIN', 'OP_UNIQUE_CARRIER', 'ts']]
tail = df[['ORIGIN', 'TAIL_NUM', 'ts']]

delay.loc[delay['DEP_DELAY'] > 0, 'DEP_DELAY'] = 1
delay.loc[delay['DEP_DELAY'] <= 0, 'DEP_DELAY'] = 0

delay = (delay
         .rename(columns={'ORIGIN': 'u', 'DEST': 'v', 'DEP_DELAY': 'e_type'})
         .assign(e_type = lambda _d: _d['e_type'] + 1)
         .assign(u_type = 1)
         .assign(v_type = 1))

carrier = (carrier
           .rename(columns={'ORIGIN': 'u', 'OP_UNIQUE_CARRIER': 'v'})
           .assign(u_type = 1)
           .assign(v_type = 2)
           .assign(e_type = 3))

tail = (tail
        .rename(columns={'ORIGIN': 'u', 'TAIL_NUM': 'v'})
        .assign(u_type = 1)
        .assign(v_type = 3)
        .assign(e_type = 4))

airports = pd.concat([df['ORIGIN'], df['DEST']]).unique()
carriers = df['OP_UNIQUE_CARRIER'].unique()
tails = df['TAIL_NUM'].unique()

idx_airports = np.arange(len(airports))
idx_carriers = np.arange(len(carriers))
idx_tails = np.arange(len(tails))

airports_dict = {airports[k]: k for k in idx_airports}
carriers_dict = {carriers[k]: k + len(airports) for k in idx_carriers}
tails_dict = {tails[k]: k + len(airports) + len(carriers) for k in idx_tails}

delay = (delay
         .assign(v = lambda _d: _d['v'].map(lambda x: airports_dict[x]))
         .assign(u = lambda _d: _d['u'].map(lambda x: airports_dict[x])))

carrier = (carrier
         .assign(v = lambda _d: _d['v'].map(lambda x: carriers_dict[x]))
         .assign(u = lambda _d: _d['u'].map(lambda x: airports_dict[x])))

tail = (tail
         .assign(v = lambda _d: _d['v'].map(lambda x: tails_dict[x]))
         .assign(u = lambda _d: _d['u'].map(lambda x: airports_dict[x])))

events = pd.concat([delay, carrier, tail]).sort_values('ts').reset_index(drop=True)

NUM_NODE = len(airports) + len(carriers) + len(tails)
NUM_EV = len(events)
NUM_N_TYPE = 3
NUM_E_TYPE = 4
CLASSES = [1, 2]

events = (events
          .assign(e_idx = np.arange(1, NUM_EV + 1))
          [['u', 'v', 'u_type', 'v_type', 'e_type', 'ts', 'e_idx']])

print("num node:", NUM_NODE)
print("num events:", NUM_EV)
events.to_csv('./data/processed/flight/events.csv', index=None)
desc = {
        "num_node": NUM_NODE,
        "num_edge": NUM_EV,
        "num_node_type": NUM_N_TYPE,
        "num_edge_type": NUM_E_TYPE,
        "classes": CLASSES
    }

with open('./data/processed/flight/desc.json', 'w') as f:
    json.dump(desc, f, indent=4)