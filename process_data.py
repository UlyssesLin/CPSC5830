import pandas as pd

df = pd.read_csv('data/raw/lol/2023_LoL_esports_match_data_from_OraclesElixir.csv')
df[['teamid', 'gameid', 'result', 'date']].dropna().drop_duplicates()
df[['teamid', 'playerid', 'date']].dropna().drop_duplicates()