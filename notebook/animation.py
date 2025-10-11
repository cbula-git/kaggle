import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
from textwrap import wrap

import sys

if __name__ == "__main__":
    print(f"Script name: {sys.argv[0]}")
    if len(sys.argv) > 1:
        print("Arguments provided:")
        print(f"Game ID: {sys.argv[1]}")
        print(f"Play ID: {sys.argv[2]}")
    else:
        print("No command-line arguments provided.")

base_dir = "../consolidated_data/"

game_id = sys.argv[1].strip()
play_id = sys.argv[2].strip()

play_keys = ['game_id', 'play_id']
player_keys = play_keys + ['nfl_id']
player_dtls = ['player_name', 'player_height', 'player_weight', 'player_birth_date', 'player_position', 'player_side', 'player_role']
player_mvmt = ['frame_id', 'x', 'y']

input_cols = player_keys + player_dtls + player_mvmt + ['player_to_predict', 'play_direction', 'absolute_yardline_number', 's', 'a', 'dir', 'o', 'num_frames_output', 'ball_land_x', 'ball_land_y', 'week']
input_df = pd.read_parquet(f'{base_dir}/master_input.parquet')

output_cols = player_keys + player_mvmt
output_df = pd.read_parquet(f'{base_dir}/master_output.parquet')

supp_df = pd.read_parquet(f'{base_dir}/supplementary.parquet')

output_xdf = pd.merge(
    output_df[output_cols], 
    input_df[player_keys + player_dtls].drop_duplicates(), 
    on=player_keys, 
    how='left'
)

fmt = {
    "Defensive Coverage": ['red', 'x'],
    "Other Route Runner": ['blue', 'o'],
    "Passer": ['navy', 'o'],
    "Targeted Receiver": ['cyan', 'o'],
    "Ball": ['black', 'X']
}

mfmt = {
    "Defensive Coverage": ['red', '-'],
    "Other Route Runner": ['blue', '-'],
    "Passer": ['navy', '-'],
    "Targeted Receiver": ['cyan', '-']
}

ball = "Ball"

play_df = input_df.query(f"game_id == {game_id} and play_id == {play_id}")
oplay_df = output_xdf.query(f"game_id == {game_id} and play_id == {play_id}")
s_df = supp_df.query(f"game_id == {game_id} and play_id == {play_id}")

fig, ax = plt.subplots(figsize=(14,7))

# Configure chart
ax.set(xlim=[0, 120], ylim=[0, 53.3])

title = "\n".join(wrap(s_df['play_description'].iloc[0], 100))
ax.set_title(f"{title}")
ax.set_facecolor('#2E8B57')

# add field markings
for yard in range(10, 111, 10):
    ax.axvline(x=yard, color='white', linestyle='-', linewidth=0.5, alpha=0.5)

ax.set_xticks(
    ticks=[x for x in range(0, 130, 10)], 
    labels=['', '', 10, 20, 30, 40, 50, 40, 30, 20, 10, '', '']
)

# add end zones
ax.add_patch(Rectangle((0, 0), 10, 53.3, facecolor='green', alpha=0.2))
ax.add_patch(Rectangle((110, 0), 10, 53.3, facecolor='green', alpha=0.2))

# info text
home_team = f"home = {s_df['home_team_abbr'].iloc[0]}"
away_team = f"away = {s_df['visitor_team_abbr'].iloc[0]}"
route_of_receiver = f"route = {s_df['route_of_targeted_receiver'].iloc[0]}"
yards_gained = f"yards_gained = {s_df['yards_gained'].iloc[0]}"
team_coverage = f"coverage = {s_df['team_coverage_type'].iloc[0]}"


plt.figtext(
    0.07, 0.015, 
    f"{home_team} ; {away_team}\n{route_of_receiver} ; {yards_gained}\n{team_coverage}", 
    wrap=False, 
    horizontalalignment='left', 
    fontsize=10
)

num_iframes = play_df['frame_id'].max()
num_oframes = oplay_df['frame_id'].max()

oplay_df.loc[:, ('frame_id')] = oplay_df['frame_id'] + num_iframes

iframes = [x for x in range(num_iframes)]
oframes = [x for x in range(num_oframes)]
frames = [x for x in range(num_iframes+num_oframes)]

# plot ball position
ax.plot(
    play_df['ball_land_x'].iloc[0], 
    play_df['ball_land_y'].iloc[0], 
    color=fmt[ball][0],
    marker=fmt[ball][1]
)

# plot starting positions
union_df = pd.concat([play_df[oplay_df.columns], oplay_df]).sort_values(by=['nfl_id', 'frame_id'])
ogrouped = oplay_df.groupby('nfl_id')
grouped = union_df.groupby('nfl_id')

path_tracer = {}

# Player marker - before pass
for player_id, player_df in grouped:
    ax.plot(
        player_df['x'].iloc[0], 
        player_df['y'].iloc[0], 
        color=fmt[player_df['player_role'].iloc[0]][0],
        marker=fmt[player_df['player_role'].iloc[0]][1],
    )

# Player marker - after pass
for player_id, player_df in ogrouped:
    ax.plot(
        player_df['x'].iloc[0], 
        player_df['y'].iloc[0], 
        color=fmt[player_df['player_role'].iloc[0]][0],
        marker='.',
    )

def init():
    # Path tracer
    for player_id, player_df in grouped:
        path_tracer[player_id] = ax.plot(
            player_df['x'].iloc[0], 
            player_df['y'].iloc[0], 
            color=mfmt[player_df['player_role'].iloc[0]][0],
            linestyle=mfmt[player_df['player_role'].iloc[0]][1],
            label=player_df['player_name'].iloc[0]
        )[0]

    return list(path_tracer.values())

    
def update(frame):
    # Path tracer
    for player_id, player_df in grouped:
        path_tracer[player_id].set_data(player_df['x'].iloc[:frame+1], player_df['y'].iloc[:frame+1])
        path_tracer[player_id].set_color(mfmt[player_df['player_role'].iloc[0]][0])
        path_tracer[player_id].set_linestyle(mfmt[player_df['player_role'].iloc[0]][1])
    return list(path_tracer.values())

ani = animation.FuncAnimation(fig=fig, func=update, init_func=init, frames=frames, interval=200, blit=True, repeat=False)


plt.legend(loc='upper left')
plt.show(block=False)

plt.pause(30)
