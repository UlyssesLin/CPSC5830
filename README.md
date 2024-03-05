# Predicting Match Outcomes Using Heterogeneous Temporal Graph Attention Networks
### Owen Clinton, Ulysses Lin
### CPSC5830, WI Quarter 2024, Seattle University

## Project Description
Our goal is to take sports and eSports historical data and train it on our modified THAN (https://github.com/moli-L/THAN) model to be able to predict game match results.\
THAN utilizes a Graph Attention Model that embeds complex temporal data onto its graph edges, and uses an optional memory model to examine wider graph neighborhood areas.\
All embeddings are sent to an attention layer that outputs node embeddings for edges to predict, and an MLP returns probabilities that certain edges exist.\
The goal is edge classification without extensive feature engineering.\
Our data comes from League of Legends, NBA, and soccer datasets.

## Instructions for Running Training and Testing
Run the following command:\
`python driver.py -d lol --prefix THAN-mem --n_degree 10 --n_epoch 30 --lr 1e-3 --bs 200`

If you wish to use the cpu instead of gpu for training, run the following:\
`python driver.py -d lol --prefix THAN-mem --n_degree 10 --n_epoch 30 --lr 1e-3 --bs 200 --gpu -1`

The above commands run against League of Legends data. You may change the dataset to the below options:
* `-d nba`
* `-d soccer`

<img width="502" alt="image" src="https://github.com/UlyssesLin/CPSC5830/assets/9372321/4061aab1-b1bc-4e82-8994-4c915d1b1c03">

## Demo for UI
A YouTube demo for usage (no audio) can be found here:
[https://youtu.be/8SheVyxQ36k](https://youtu.be/8SheVyxQ36k)

## Instructions for Running Streamlit UI
Streamlit is a python library that provides our front-end, displaying the UI for exploring our interactive graphs. On the Command Line, you run the below command to run the Streamlit server on localhost.\
`streamlit run <local path to file>/vis_ui.py`\
The functionality is controlled by the sidebar to the left, with ways to filter what is represented by the graph on screen.\
You may choose to focus on either two teams and their connections, or just one team or player and their immediate neighborhood.
#### Team Option
When choosing the two-team option, you can select the two teams from the dropdowns or simply input the team numbers.\
Once you have chosen the two teams, you can optionally select a time window by inputting a start and end time to the window in datetime (YYYY-MM-DD hh:mm:ss) format.\
You can alternatively choose a match played between the two teams from the dropdown. Note that this will ignore any time window previously inputted.\
By default the graph shown will only display a single directed edge for each event (see below section on interpreting the graph) - if the graph has many edges this can become convoluted.\
To reduce the visual clutter, you may toggle on 'Show unifed edges', so that all edges of the same type between the same two nodes will display as one unified line; the thickness represents the number of edges that have been condensed.\
If you wish to only see events that have been tested during the testing phase (note that only a fraction of all edges have been tested), then toggle 'Show predictions only'.\
Note that doing so will ignore the time window and match filters, and display all tested edges (which are all match edges) over time.

#### Node Adjacency Option
Node Adjacency refers to focusing on a single team or player node, and seeing the immediate connections around it.\
Here you may either select a node, or input a team or player node number.
The recommendation is to toggle unified edges for a clearer viewing experience.

## Interpreting the Graphs
Light blue nodes represent players while dark blue nodes are teams.
Connecting the nodes are directed edges of various colors. Matches are in red or green to indicate a loss or win; the arrow direction indicates home and away teams - i.e., the arrow originates with the home team and points to the away team.\
Players joining a team are arrows in light blue, while game info edges are gray and always thin.\
To learn more information about an edge, hover over it. Match tooltips will display rows of dates for when the match took place.\
If a match has been used during the testing phase, it will additionally display (in brackets) our prediction and a percentage chance, and whether our prediction was correct.
