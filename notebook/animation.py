import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

base_dir = "../consolidated_data/"

game_id = '2023090700'
play_id = '1679'

player_keys = ['game_id', 'play_id', 'nfl_id']
player_dtls = ['player_name', 'player_height', 'player_weight', 'player_birth_date', 'player_position', 'player_side', 'player_role']
player_mvmt = ['frame_id', 'x', 'y']

input_cols = player_keys + player_dtls + player_mvmt + ['player_to_predict', 'play_direction', 'absolute_yardline_number', 's', 'a', 'dir', 'o', 'num_frames_output', 'ball_land_x', 'ball_land_y', 'week']
input_df = pd.read_parquet(f'{base_dir}/master_input.parquet')

output_cols = player_keys + player_mvmt
output_df = pd.read_parquet(f'{base_dir}/master_output.parquet')

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
    "Targeted Receiver": ['green', 'o'],
    "Ball": ['black', 'X']
}

mfmt = {
    "Defensive Coverage": ['red', '-'],
    "Other Route Runner": ['blue', '-'],
    "Passer": ['navy', '-'],
    "Targeted Receiver": ['green', '-']
}

ball = "Ball"

play_df = input_df.query(f"game_id == {game_id} and play_id == {play_id}")
oplay_df = output_xdf.query(f"game_id == {game_id} and play_id == {play_id}")

fig, ax = plt.subplots(figsize=(14,7))
ax.set(xlim=[0, 120], ylim=[0, 53.3])

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
grouped = union_df.groupby('nfl_id')

artists = {}
for player_id, player_df in grouped:
    print(player_id)
    ax.plot(
        player_df['x'].iloc[0], 
        player_df['y'].iloc[0], 
        color=fmt[player_df['player_role'].iloc[0]][0],
        marker=fmt[player_df['player_role'].iloc[0]][1],
    )

def init():
    grouped = play_df.groupby('nfl_id')
    for player_id, player_df in grouped:
        artists[player_id] = ax.plot(
            player_df['x'].iloc[0], 
            player_df['y'].iloc[0], 
            color=mfmt[player_df['player_role'].iloc[0]][0],
            linestyle=mfmt[player_df['player_role'].iloc[0]][1],
        )[0]
    return list(artists.values())

    
def update(frame):
    for player_id, player_df in grouped:
        artists[player_id].set_data(player_df['x'].iloc[:frame+1], player_df['y'].iloc[:frame+1])
        artists[player_id].set_color(mfmt[player_df['player_role'].iloc[0]][0])
        artists[player_id].set_linestyle(mfmt[player_df['player_role'].iloc[0]][1])
    return list(artists.values())

ani = animation.FuncAnimation(fig=fig, func=update, init_func=init, frames=frames, interval=200, blit=True, repeat=False)
plt.show()