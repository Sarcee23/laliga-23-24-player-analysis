import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="La Liga Player Comparison & Scout", layout="wide", page_icon="⚽")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('laliga_players_23-24_cleaned.csv')
    return df

df = load_data()

# Calculate additional metrics
if 'save_percentage' not in df.columns:
    df['save_percentage'] = ((df['saves_made'] / (df['saves_made'] + df['goals_conceded'])) * 100).fillna(0)
    df['gk_distribution_success'] = ((df['gk_successful_distribution'] / 
                                      (df['gk_successful_distribution'] + df['gk_unsuccessful_distribution'])) * 100).fillna(0)

# Add more derived metrics
df['xG_per90'] = (df['shots_on_target_inc_goals'] * 0.35 + df['shots_off_target_inc_woodwork'] * 0.05) / (df['minutes_played'] / 90)
df['progressive_passes_per90'] = (df['forward_passes'] / (df['minutes_played'] / 90)).fillna(0)
df['ball_retention'] = ((df['successful_dribbles'] + df['duels_won']) / (df['unsuccessful_dribbles'] + df['duels_lost'] + 1) * 100).fillna(0)
df['defensive_actions_per90'] = ((df['total_tackles'] + df['interceptions'] + df['blocks']) / (df['minutes_played'] / 90)).fillna(0)
df['attacking_output_per90'] = ((df['goals'] + df['goal_assists']) / (df['minutes_played'] / 90)).fillna(0)
df['aerial_dominance'] = (df['aerial_duels_won'] / (df['aerial_duels'] + 1) * 100).fillna(0)

# Define position-specific metrics
def get_position_metrics(position):
    """Return relevant metrics based on position"""
    metrics = {
        'Forward': {
            'Goals per 90': 'goals_per90',
            'xG per 90': 'xG_per90',
            'Assists per 90': 'assists_per90',
            'Shot Accuracy %': 'shot_accuracy',
            'Successful Dribbles': 'successful_dribbles',
            'Duels Won %': 'duel_win_rate',
            'Ball Retention': 'ball_retention'
        },
        'Midfielder': {
            'Attacking Output': 'attacking_output_per90',
            'Pass Accuracy %': 'pass_accuracy',
            'Progressive Passes': 'progressive_passes_per90',
            'Key Passes': 'key_passes_attempt_assists',
            'Duels Won %': 'duel_win_rate',
            'Defensive Actions': 'defensive_actions_per90',
            'Ball Retention': 'ball_retention'
        },
        'Defender': {
            'Defensive Actions': 'defensive_actions_per90',
            'Tackles per 90': 'tackles_per90',
            'Interceptions': 'interceptions',
            'Duels Won %': 'duel_win_rate',
            'Aerial Dominance %': 'aerial_dominance',
            'Pass Accuracy %': 'pass_accuracy',
            'Clearances': 'total_clearances'
        },
        'Goalkeeper': {
            'Saves Made': 'saves_made',
            'Save %': 'save_percentage',
            'Clean Sheets': 'clean_sheets',
            'Distribution Success': 'gk_distribution_success',
            'Catches': 'catches',
            'Punches': 'punches'
        }
    }
    return metrics.get(position, metrics['Midfielder'])

def get_extended_stats(position):
    """Return extended stats list for detailed comparison"""
    stats = {
        'Forward': [
            ('Goals', 'goals'),
            ('Goals per 90', 'goals_per90'),
            ('xG per 90', 'xG_per90'),
            ('Assists', 'goal_assists'),
            ('Assists per 90', 'assists_per90'),
            ('Attacking Output/90', 'attacking_output_per90'),
            ('Total Shots', 'total_shots'),
            ('Shots per 90', 'shots_per90'),
            ('Shots on Target', 'shots_on_target_inc_goals'),
            ('Shot Accuracy %', 'shot_accuracy'),
            ('Successful Dribbles', 'successful_dribbles'),
            ('Dribble Success %', 'ball_retention'),
            ('Duels Won %', 'duel_win_rate'),
            ('Aerial Duels Won', 'aerial_duels_won'),
            ('Touches in Opp Box', 'total_touches_in_opposition_box'),
            ('Pass Accuracy %', 'pass_accuracy'),
        ],
        'Midfielder': [
            ('Goals', 'goals'),
            ('Assists', 'goal_assists'),
            ('Attacking Output/90', 'attacking_output_per90'),
            ('Key Passes', 'key_passes_attempt_assists'),
            ('Through Balls', 'through_balls'),
            ('Progressive Passes/90', 'progressive_passes_per90'),
            ('Pass Accuracy %', 'pass_accuracy'),
            ('Successful Long Passes', 'successful_long_passes'),
            ('Passes Opp Half', 'successful_passes_opposition_half'),
            ('Defensive Actions/90', 'defensive_actions_per90'),
            ('Tackles per 90', 'tackles_per90'),
            ('Interceptions', 'interceptions'),
            ('Duels Won %', 'duel_win_rate'),
            ('Ball Retention', 'ball_retention'),
            ('Recoveries', 'recoveries'),
        ],
        'Defender': [
            ('Defensive Actions/90', 'defensive_actions_per90'),
            ('Tackles', 'total_tackles'),
            ('Tackles per 90', 'tackles_per90'),
            ('Tackles Won', 'tackles_won'),
            ('Interceptions', 'interceptions'),
            ('Clearances', 'total_clearances'),
            ('Blocks', 'blocks'),
            ('Duels Won %', 'duel_win_rate'),
            ('Aerial Duels Won', 'aerial_duels_won'),
            ('Aerial Dominance %', 'aerial_dominance'),
            ('Recoveries', 'recoveries'),
            ('Pass Accuracy %', 'pass_accuracy'),
            ('Progressive Passes/90', 'progressive_passes_per90'),
            ('Fouls Conceded', 'total_fouls_conceded'),
            ('Yellow Cards', 'yellow_cards'),
        ],
        'Goalkeeper': [
            ('Saves Made', 'saves_made'),
            ('Save %', 'save_percentage'),
            ('Clean Sheets', 'clean_sheets'),
            ('Goals Conceded', 'goals_conceded'),
            ('Saves from Inside Box', 'saves_made_from_inside_box'),
            ('Saves from Outside Box', 'saves_made_from_outside_box'),
            ('Catches', 'catches'),
            ('Punches', 'punches'),
            ('Distribution Success %', 'gk_distribution_success'),
            ('Penalties Faced', 'penalties_faced'),
            ('Penalties Saved', 'penalties_saved'),
            ('Crosses Not Claimed', 'crosses_not_claimed'),
        ]
    }
    return stats.get(position, stats['Midfielder'])

def create_radar_chart(player1_data, player2_data, metrics_dict, player1_name, player2_name):
    """Create radar chart comparing two players"""
    
    categories = list(metrics_dict.keys())
    metric_cols = list(metrics_dict.values())
    
    player1_values = []
    player2_values = []
    
    for metric_col in metric_cols:
        p1_val = player1_data[metric_col] if pd.notna(player1_data[metric_col]) else 0
        p2_val = player2_data[metric_col] if pd.notna(player2_data[metric_col]) else 0
        player1_values.append(p1_val)
        player2_values.append(p2_val)
    
    normalized_p1 = []
    normalized_p2 = []
    
    for i, metric_col in enumerate(metric_cols):
        position = player1_data['position']
        same_position = df[df['position'] == position]
        all_values = same_position[metric_col].dropna()
        
        if len(all_values) > 0:
            min_val = all_values.min()
            max_val = all_values.max()
            
            if max_val > min_val:
                norm_p1 = ((player1_values[i] - min_val) / (max_val - min_val)) * 100
                norm_p2 = ((player2_values[i] - min_val) / (max_val - min_val)) * 100
            else:
                norm_p1 = 50
                norm_p2 = 50
        else:
            norm_p1 = 50
            norm_p2 = 50
            
        normalized_p1.append(max(0, min(100, norm_p1)))
        normalized_p2.append(max(0, min(100, norm_p2)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_p1 + [normalized_p1[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=player1_name,
        line=dict(color='#FF4B4B', width=2),
        fillcolor='rgba(255, 75, 75, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_p2 + [normalized_p2[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=player2_name,
        line=dict(color='#4B4BFF', width=2),
        fillcolor='rgba(75, 75, 255, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False
            )
        ),
        showlegend=True,
        title="Player Comparison (Normalized by Position)",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig, player1_values, player2_values, categories

def get_percentile_rank(value, all_values):
    """Calculate percentile rank of a value"""
    if pd.isna(value) or len(all_values) == 0:
        return 50
    return (all_values < value).sum() / len(all_values) * 100

def create_comparison_table(p1_data, p2_data, position, player1_name, player2_name):
    """Create color-coded comparison table"""
    
    stats_list = get_extended_stats(position)
    same_position_df = df[df['position'] == position]
    
    table_data = []
    
    for stat_name, stat_col in stats_list:
        if stat_col not in df.columns:
            continue
            
        p1_val = p1_data[stat_col] if pd.notna(p1_data[stat_col]) else 0
        p2_val = p2_data[stat_col] if pd.notna(p2_data[stat_col]) else 0
        
        all_values = same_position_df[stat_col].dropna()
        p1_percentile = get_percentile_rank(p1_val, all_values)
        p2_percentile = get_percentile_rank(p2_val, all_values)
        
        if p1_val > p2_val:
            winner = player1_name
        elif p2_val > p1_val:
            winner = player2_name
        else:
            winner = "Tie"
        
        table_data.append({
            'Stat': stat_name,
            player1_name: f"{p1_val:.2f}",
            f'{player1_name} %ile': f"{p1_percentile:.0f}",
            player2_name: f"{p2_val:.2f}",
            f'{player2_name} %ile': f"{p2_percentile:.0f}",
            'Winner': winner
        })
    
    return pd.DataFrame(table_data)

def find_similar_players(player_data, position, top_n=5):
    """Find similar players using cosine similarity"""
    
    # Feature selection based on position
    if position == 'Forward':
        features = ['goals_per90', 'assists_per90', 'shots_per90', 'successful_dribbles', 
                   'duel_win_rate', 'shot_accuracy', 'pass_accuracy']
    elif position == 'Midfielder':
        features = ['assists_per90', 'passes_per90', 'pass_accuracy', 'key_passes_attempt_assists',
                   'tackles_per90', 'duel_win_rate', 'progressive_passes_per90']
    elif position == 'Defender':
        features = ['tackles_per90', 'interceptions', 'total_clearances', 'duel_win_rate',
                   'aerial_dominance', 'pass_accuracy', 'blocks']
    else:  # Goalkeeper
        features = ['saves_made', 'save_percentage', 'clean_sheets', 'gk_distribution_success']
    
    # Filter to same position with min minutes
    same_position_df = df[(df['position'] == position) & (df['minutes_played'] >= 450)].copy()
    
    # Get player index
    player_name = player_data['name']
    player_idx = same_position_df[same_position_df['name'] == player_name].index
    
    if len(player_idx) == 0:
        return pd.DataFrame()
    
    # Prepare feature matrix
    feature_df = same_position_df[features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)
    
    # Calculate similarity
    player_features = scaled_features[same_position_df.index.get_loc(player_idx[0])].reshape(1, -1)
    similarities = cosine_similarity(player_features, scaled_features)[0]
    
    # Get top similar players (excluding the player itself)
    same_position_df['similarity'] = similarities
    similar_players = same_position_df[same_position_df['name'] != player_name].nlargest(top_n, 'similarity')
    
    return similar_players[['name', 'team', 'age', 'similarity', 'goals', 'goal_assists', 'minutes_played']]

def calculate_player_score(player_data, position):
    """Calculate overall player score based on position"""
    
    if position == 'Forward':
        score = (
            player_data['goals_per90'] * 10 +
            player_data['assists_per90'] * 8 +
            player_data['shot_accuracy'] * 0.5 +
            player_data['successful_dribbles'] * 0.3 +
            player_data['duel_win_rate'] * 0.3
        )
    elif position == 'Midfielder':
        score = (
            player_data['assists_per90'] * 8 +
            player_data['key_passes_attempt_assists'] * 0.5 +
            player_data['pass_accuracy'] * 0.5 +
            player_data['tackles_per90'] * 3 +
            player_data['duel_win_rate'] * 0.3 +
            player_data['progressive_passes_per90'] * 0.2
        )
    elif position == 'Defender':
        score = (
            player_data['tackles_per90'] * 3 +
            player_data['interceptions'] * 0.3 +
            player_data['duel_win_rate'] * 0.4 +
            player_data['aerial_dominance'] * 0.2 +
            player_data['pass_accuracy'] * 0.3 +
            player_data['total_clearances'] * 0.1
        )
    else:  # Goalkeeper
        score = (
            player_data['save_percentage'] * 0.8 +
            player_data['clean_sheets'] * 2 +
            player_data['saves_made'] * 0.2 +
            player_data['gk_distribution_success'] * 0.3
        )
    
    return score

def get_league_ranking(player_data, stat_col, position):
    """Get player's rank in league for a specific stat"""
    same_position_df = df[(df['position'] == position) & (df['minutes_played'] >= 450)]
    
    if stat_col not in same_position_df.columns:
        return None, None
    
    sorted_df = same_position_df.sort_values(by=stat_col, ascending=False, na_position='last')
    
    player_value = player_data[stat_col]
    if pd.isna(player_value):
        return None, None
    
    rank = (sorted_df[stat_col] > player_value).sum() + 1
    total = len(sorted_df[sorted_df[stat_col].notna()])
    
    return rank, total

def create_playing_style_chart(player_data, position):
    """Create a chart showing player's playing style distribution"""
    
    if position == 'Forward':
        categories = ['Finishing', 'Creativity', 'Dribbling', 'Physical', 'Link-up Play']
        values = [
            min(100, player_data['goals_per90'] * 30 + player_data['shot_accuracy'] * 0.5),
            min(100, player_data['assists_per90'] * 30 + player_data['key_passes_attempt_assists'] * 2),
            min(100, player_data['successful_dribbles'] * 2),
            min(100, player_data['duel_win_rate']),
            min(100, player_data['pass_accuracy'])
        ]
    elif position == 'Midfielder':
        categories = ['Attacking', 'Passing', 'Defending', 'Ball Retention', 'Work Rate']
        values = [
            min(100, player_data['attacking_output_per90'] * 30),
            min(100, player_data['pass_accuracy']),
            min(100, player_data['defensive_actions_per90'] * 10),
            min(100, player_data['ball_retention']),
            min(100, player_data['duels_won'] * 0.5)
        ]
    elif position == 'Defender':
        categories = ['Tackling', 'Positioning', 'Aerial', 'Passing', 'Physicality']
        values = [
            min(100, player_data['tackles_per90'] * 20),
            min(100, player_data['interceptions'] * 2),
            min(100, player_data['aerial_dominance']),
            min(100, player_data['pass_accuracy']),
            min(100, player_data['duel_win_rate'])
        ]
    else:  # Goalkeeper
        categories = ['Shot Stopping', 'Distribution', 'Command of Area', 'Penalty Saving', 'Consistency']
        values = [
            min(100, player_data['save_percentage']),
            min(100, player_data['gk_distribution_success']),
            min(100, player_data['catches'] * 2),
            min(100, (player_data['penalties_saved'] / max(player_data['penalties_faced'], 1)) * 100),
            min(100, player_data['clean_sheets'] * 3)
        ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Playing Style',
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="Playing Style Profile",
        height=400
    )
    
    return fig

# Title with tabs
st.title("⚽ La Liga 2023-24: Advanced Player Analysis")

tab1, tab2, tab3 = st.tabs(["Player Comparison", "Scout Underrated Players", "Player Profile"])

# Sidebar filters
st.sidebar.header("Filters")
positions = ['All'] + sorted(df['position'].dropna().unique().tolist())
selected_position = st.sidebar.selectbox("Position", positions)
teams = ['All'] + sorted(df['team'].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Team", teams)


df_filtered = df.copy()
if selected_position != 'All':
    df_filtered = df_filtered[df_filtered['position'] == selected_position]
if selected_team != 'All':
    df_filtered = df_filtered[df_filtered['team'] == selected_team]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(df_filtered)}** players match filters")

# TAB 1: PLAYER COMPARISON
with tab1:
    st.header("Select Players to Compare")
    
    col1, col2 = st.columns(2)
    
    player_mapping = {}
    for _, row in df_filtered[['name', 'name_normalized', 'team']].drop_duplicates().iterrows():
        key = row['name_normalized']
        value = f"{row['name']} ({row['team']})"
        player_mapping[key] = {
            'display': value,
            'real_name': row['name']
        }
    
    sorted_keys = sorted(player_mapping.keys())
    player_options = [player_mapping[key]['display'] for key in sorted_keys]
    display_to_real = {player_mapping[key]['display']: player_mapping[key]['real_name'] 
                       for key in sorted_keys}
    
    with col1:
        player1_display = st.selectbox("Player 1", options=player_options, key='player1')
        player1 = display_to_real[player1_display]
    
    with col2:
        player2_display = st.selectbox("Player 2", options=player_options, key='player2')
        player2 = display_to_real[player2_display]
    
    p1_data = df_filtered[df_filtered['name'] == player1].iloc[0]
    p2_data = df_filtered[df_filtered['name'] == player2].iloc[0]
    
    same_position = p1_data['position'] == p2_data['position']
    
    if not same_position:
        st.warning("⚠️ These players play different positions. Comparison may not be meaningful.")
    
    # Player basic info with enhanced metrics
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{player1}")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write(f"**Team:** {p1_data['team']}")
            st.write(f"**Position:** {p1_data['position']}")
            age = 2024 - pd.to_datetime(p1_data['date_of_birth']).year if pd.notna(p1_data['date_of_birth']) else 'N/A'
            st.write(f"**Age:** {age}")
            st.write(f"**Nationality:** {p1_data['country']}")
        with info_col2:
            st.write(f"**Appearances:** {int(p1_data['appearances'])}")
            st.write(f"**Minutes:** {int(p1_data['minutes_played'])}")
            if p1_data['position'] != 'Goalkeeper':
                st.write(f"**Goals:** {int(p1_data['goals'])}")
                st.write(f"**Assists:** {int(p1_data['goal_assists'])}")
            else:
                st.write(f"**Clean Sheets:** {int(p1_data['clean_sheets'])}")
                st.write(f"**Saves:** {int(p1_data['saves_made'])}")
        
        # Overall Score
        p1_score = calculate_player_score(p1_data, p1_data['position'])
        st.metric("Overall Performance Score", f"{p1_score:.1f}")
    
    with col2:
        st.subheader(f"{player2}")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write(f"**Team:** {p2_data['team']}")
            st.write(f"**Position:** {p2_data['position']}")
            age = 2024 - pd.to_datetime(p2_data['date_of_birth']).year if pd.notna(p2_data['date_of_birth']) else 'N/A'
            st.write(f"**Age:** {age}")
            st.write(f"**Nationality:** {p2_data['country']}")
        with info_col2:
            st.write(f"**Appearances:** {int(p2_data['appearances'])}")
            st.write(f"**Minutes:** {int(p2_data['minutes_played'])}")
            if p2_data['position'] != 'Goalkeeper':
                st.write(f"**Goals:** {int(p2_data['goals'])}")
                st.write(f"**Assists:** {int(p2_data['goal_assists'])}")
            else:
                st.write(f"**Clean Sheets:** {int(p2_data['clean_sheets'])}")
                st.write(f"**Saves:** {int(p2_data['saves_made'])}")
        
        p2_score = calculate_player_score(p2_data, p2_data['position'])
        st.metric("Overall Performance Score", f"{p2_score:.1f}")
    
    # Radar Chart
    st.markdown("---")
    st.header("Performance Radar")
    
    position_for_metrics = p1_data['position']
    metrics_dict = get_position_metrics(position_for_metrics)
    
    radar_fig, p1_values, p2_values, categories = create_radar_chart(
        p1_data, p2_data, metrics_dict, player1, player2
    )
    
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Head-to-Head Stats
    st.markdown("---")
    st.header("Head-to-Head Statistics")
    
    comparison_df = create_comparison_table(p1_data, p2_data, position_for_metrics, player1, player2)
    
    # Calculate wins
    p1_wins = (comparison_df['Winner'] == player1).sum()
    p2_wins = (comparison_df['Winner'] == player2).sum()
    ties = (comparison_df['Winner'] == 'Tie').sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{player1} Wins", p1_wins)
    col2.metric("Ties", ties)
    col3.metric(f"{player2} Wins", p2_wins)
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=500)
    
    # League Rankings
    st.markdown("---")
    st.header("League Rankings")
    
    ranking_stats = {
        'Forward': [('Goals', 'goals'), ('Goals per 90', 'goals_per90'), ('Assists', 'goal_assists'), ('Shot Accuracy %', 'shot_accuracy')],
        'Midfielder': [('Assists', 'goal_assists'), ('Pass Accuracy %', 'pass_accuracy'), ('Key Passes', 'key_passes_attempt_assists'), ('Defensive Actions', 'defensive_actions_per90')],
        'Defender': [('Tackles', 'total_tackles'), ('Interceptions', 'interceptions'), ('Duels Won %', 'duel_win_rate'), ('Clearances', 'total_clearances')],
        'Goalkeeper': [('Saves', 'saves_made'), ('Save %', 'save_percentage'), ('Clean Sheets', 'clean_sheets'), ('Distribution', 'gk_distribution_success')]
    }
    
    key_stats = ranking_stats.get(position_for_metrics, ranking_stats['Midfielder'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{player1}")
        for stat_name, stat_col in key_stats:
            rank, total = get_league_ranking(p1_data, stat_col, position_for_metrics)
            if rank and total:
                percentile = ((total - rank) / total) * 100
                color = "🟢" if percentile >= 90 else "🟡" if percentile >= 70 else "🟠" if percentile >= 50 else "🔴"
                st.write(f"{color} **{stat_name}:** #{rank} of {total} ({percentile:.0f}th %ile)")
    
    with col2:
        st.subheader(f"{player2}")
        for stat_name, stat_col in key_stats:
            rank, total = get_league_ranking(p2_data, stat_col, position_for_metrics)
            if rank and total:
                percentile = ((total - rank) / total) * 100
                color = "🟢" if percentile >= 90 else "🟡" if percentile >= 70 else "🟠" if percentile >= 50 else "🔴"
                st.write(f"{color} **{stat_name}:** #{rank} of {total} ({percentile:.0f}th %ile)")

# TAB 2: SCOUT UNDERRATED PLAYERS
with tab2:
    st.header("Scout Underrated Players")
    st.write("Find hidden gems based on performance metrics vs. market perception")
    
    # Filters for scouting
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scout_position = st.selectbox("Scout Position", 
                                     ['Forward', 'Midfielder', 'Defender', 'Goalkeeper'],
                                     key='scout_pos')
    
    with col2:
        max_age = st.slider("Maximum Age", 18, 35, 26)
    
    with col3:
        min_minutes_scout = st.slider("Min Minutes", 450, 3000, 900)
    
    # Filter for scouting
    scout_df = df[(df['position'] == scout_position) & 
                  (df['minutes_played'] >= min_minutes_scout)].copy()
    
    # Calculate age
    scout_df['age'] = 2024 - pd.to_datetime(scout_df['date_of_birth']).dt.year
    scout_df = scout_df[scout_df['age'] <= max_age]
    
    # Calculate performance score
    scout_df['performance_score'] = scout_df.apply(
        lambda row: calculate_player_score(row, scout_position), axis=1
    )
    
    # Sort by performance score
    scout_df = scout_df.sort_values('performance_score', ascending=False)
    
    st.subheader(f"Top {min(20, len(scout_df))} {scout_position}s Under {max_age}")
    
    # Create display dataframe
    if scout_position == 'Forward':
        display_cols = ['name', 'team', 'age', 'minutes_played', 'goals', 'goal_assists', 
                       'goals_per90', 'shot_accuracy', 'successful_dribbles', 'performance_score']
    elif scout_position == 'Midfielder':
        display_cols = ['name', 'team', 'age', 'minutes_played', 'goal_assists', 
                       'pass_accuracy', 'key_passes_attempt_assists', 'tackles_per90', 
                       'defensive_actions_per90', 'performance_score']
    elif scout_position == 'Defender':
        display_cols = ['name', 'team', 'age', 'minutes_played', 'tackles_per90', 
                       'interceptions', 'duel_win_rate', 'aerial_dominance', 
                       'pass_accuracy', 'performance_score']
    else:  # Goalkeeper
        display_cols = ['name', 'team', 'age', 'minutes_played', 'saves_made', 
                       'save_percentage', 'clean_sheets', 'gk_distribution_success', 
                       'performance_score']
    
    display_df = scout_df[display_cols].head(20).reset_index(drop=True)
    display_df.index = display_df.index + 1
    
    st.dataframe(display_df, use_container_width=True, height=500)
    
    # Detailed player cards
    st.markdown("---")
    st.subheader("Player Deep Dive")
    
    selected_scout_player = st.selectbox(
        "Select a player for detailed analysis",
        options=scout_df['name'].head(20).tolist()
    )
    
    if selected_scout_player:
        scout_player_data = scout_df[scout_df['name'] == selected_scout_player].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Performance Score", f"{scout_player_data['performance_score']:.1f}")
            st.write(f"**Team:** {scout_player_data['team']}")
            st.write(f"**Age:** {int(scout_player_data['age'])}")
            st.write(f"**Minutes:** {int(scout_player_data['minutes_played'])}")
        
        with col2:
            # Key stats
            if scout_position == 'Forward':
                st.metric("Goals per 90", f"{scout_player_data['goals_per90']:.2f}")
                st.metric("Shot Accuracy %", f"{scout_player_data['shot_accuracy']:.1f}")
            elif scout_position == 'Midfielder':
                st.metric("Pass Accuracy %", f"{scout_player_data['pass_accuracy']:.1f}")
                st.metric("Key Passes", f"{scout_player_data['key_passes_attempt_assists']:.0f}")
            elif scout_position == 'Defender':
                st.metric("Tackles per 90", f"{scout_player_data['tackles_per90']:.2f}")
                st.metric("Duel Win Rate %", f"{scout_player_data['duel_win_rate']:.1f}")
            else:
                st.metric("Save %", f"{scout_player_data['save_percentage']:.1f}")
                st.metric("Clean Sheets", f"{int(scout_player_data['clean_sheets'])}")
        
        with col3:
            # League ranking for key stat
            key_stat_map = {
                'Forward': ('goals_per90', 'Goals per 90'),
                'Midfielder': ('pass_accuracy', 'Pass Accuracy'),
                'Defender': ('defensive_actions_per90', 'Defensive Actions'),
                'Goalkeeper': ('save_percentage', 'Save %')
            }
            
            stat_col, stat_name = key_stat_map[scout_position]
            rank, total = get_league_ranking(scout_player_data, stat_col, scout_position)
            
            if rank and total:
                percentile = ((total - rank) / total) * 100
                st.metric(f"{stat_name} Rank", f"#{rank} / {total}")
                st.metric("Percentile", f"{percentile:.0f}th")
        
        # Playing style chart
        st.markdown("---")
        style_chart = create_playing_style_chart(scout_player_data, scout_position)
        st.plotly_chart(style_chart, use_container_width=True)
        
        # Similar players
        st.markdown("---")
        st.subheader("Similar Players")
        similar = find_similar_players(scout_player_data, scout_position, top_n=5)
        
        if not similar.empty:
            similar_display = similar.copy()
            similar_display['similarity'] = (similar_display['similarity'] * 100).round(1)
            similar_display.columns = ['Name', 'Team', 'Age', 'Similarity %', 'Goals', 'Assists', 'Minutes']
            st.dataframe(similar_display, use_container_width=True, hide_index=True)
        else:
            st.info("No similar players found")

# TAB 3: PLAYER PROFILE
with tab3:
    st.header("Individual Player Profile")
    
    # Player selection
    profile_player_display = st.selectbox(
        "Select Player",
        options=player_options,
        key='profile_player'
    )
    profile_player = display_to_real[profile_player_display]
    
    profile_data = df_filtered[df_filtered['name'] == profile_player].iloc[0]
    
    # Header with key info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Position", profile_data['position'])
        st.metric("Team", profile_data['team'])
    
    with col2:
        age = 2024 - pd.to_datetime(profile_data['date_of_birth']).year if pd.notna(profile_data['date_of_birth']) else 'N/A'
        st.metric("Age", age)
        st.metric("Nationality", profile_data['country'])
    
    with col3:
        st.metric("Appearances", int(profile_data['appearances']))
        st.metric("Minutes", int(profile_data['minutes_played']))
    
    with col4:
        if profile_data['position'] != 'Goalkeeper':
            st.metric("Goals", int(profile_data['goals']))
            st.metric("Assists", int(profile_data['goal_assists']))
        else:
            st.metric("Clean Sheets", int(profile_data['clean_sheets']))
            st.metric("Saves", int(profile_data['saves_made']))
    
    # Overall performance score
    st.markdown("---")
    profile_score = calculate_player_score(profile_data, profile_data['position'])
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("Overall Performance Score", f"{profile_score:.1f}", 
                 help="Composite score based on position-specific metrics")
    
    # Playing style
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Playing Style Profile")
        style_fig = create_playing_style_chart(profile_data, profile_data['position'])
        st.plotly_chart(style_fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Strengths")
        
        # Get percentiles for all relevant stats
        position = profile_data['position']
        same_pos = df[df['position'] == position]
        
        strengths = []
        
        if position == 'Forward':
            stat_checks = [
                ('Goals per 90', 'goals_per90'),
                ('Shot Accuracy', 'shot_accuracy'),
                ('Dribbling', 'successful_dribbles'),
                ('Assists per 90', 'assists_per90')
            ]
        elif position == 'Midfielder':
            stat_checks = [
                ('Pass Accuracy', 'pass_accuracy'),
                ('Key Passes', 'key_passes_attempt_assists'),
                ('Defensive Actions', 'defensive_actions_per90'),
                ('Ball Retention', 'ball_retention')
            ]
        elif position == 'Defender':
            stat_checks = [
                ('Tackling', 'tackles_per90'),
                ('Aerial Dominance', 'aerial_dominance'),
                ('Interceptions', 'interceptions'),
                ('Pass Accuracy', 'pass_accuracy')
            ]
        else:
            stat_checks = [
                ('Save Percentage', 'save_percentage'),
                ('Distribution', 'gk_distribution_success'),
                ('Saves Made', 'saves_made'),
                ('Clean Sheets', 'clean_sheets')
            ]
        
        for stat_name, stat_col in stat_checks:
            if stat_col in profile_data.index:
                all_vals = same_pos[stat_col].dropna()
                percentile = get_percentile_rank(profile_data[stat_col], all_vals)
                
                if percentile >= 75:
                    emoji = "🌟" if percentile >= 90 else "⭐"
                    strengths.append(f"{emoji} **{stat_name}** ({percentile:.0f}th percentile)")
        
        if strengths:
            for strength in strengths:
                st.markdown(strength)
        else:
            st.info("Developing player with room for growth")
    
    # Detailed statistics
    st.markdown("---")
    st.subheader("Complete Statistics")
    
    stats_list = get_extended_stats(profile_data['position'])
    
    # Create two columns for stats
    col1, col2 = st.columns(2)
    
    mid_point = len(stats_list) // 2
    
    with col1:
        for stat_name, stat_col in stats_list[:mid_point]:
            if stat_col in profile_data.index:
                value = profile_data[stat_col]
                if pd.notna(value):
                    all_vals = same_pos[stat_col].dropna()
                    percentile = get_percentile_rank(value, all_vals)
                    
                    color = "🟢" if percentile >= 80 else "🟡" if percentile >= 60 else "🟠" if percentile >= 40 else "🔴"
                    st.write(f"{color} **{stat_name}:** {value:.2f} ({percentile:.0f}th %ile)")
    
    with col2:
        for stat_name, stat_col in stats_list[mid_point:]:
            if stat_col in profile_data.index:
                value = profile_data[stat_col]
                if pd.notna(value):
                    all_vals = same_pos[stat_col].dropna()
                    percentile = get_percentile_rank(value, all_vals)
                    
                    color = "🟢" if percentile >= 80 else "🟡" if percentile >= 60 else "🟠" if percentile >= 40 else "🔴"
                    st.write(f"{color} **{stat_name}:** {value:.2f} ({percentile:.0f}th %ile)")
    
    # Similar players section
    st.markdown("---")
    st.subheader("Similar Players")
    
    similar_players = find_similar_players(profile_data, profile_data['position'], top_n=8)
    
    if not similar_players.empty:
        similar_display = similar_players.copy()
        similar_display['similarity'] = (similar_display['similarity'] * 100).round(1)
        similar_display.columns = ['Name', 'Team', 'Age', 'Similarity %', 'Goals', 'Assists', 'Minutes']
        
        st.dataframe(similar_display, use_container_width=True, hide_index=True)
        
        # Quick comparison with most similar
        st.markdown("---")
        st.subheader(f"Quick Comparison: {profile_player} vs {similar_players.iloc[0]['name']}")
        
        most_similar_name = similar_players.iloc[0]['name']
        most_similar_data = df[df['name'] == most_similar_name].iloc[0]
        
        metrics_dict = get_position_metrics(profile_data['position'])
        
        quick_radar, _, _, _ = create_radar_chart(
            profile_data, most_similar_data, metrics_dict,
            profile_player, most_similar_name
        )
        
        st.plotly_chart(quick_radar, use_container_width=True)
    else:
        st.info("No similar players found in the dataset")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>La Liga 2023-24 Player Analysis Tool</p>
        <p>Data includes players with at least 5 appearances for meaningful analysis</p>
        <p>Made by Subhayan Roy Chowdhury</p>
    </div>
""", unsafe_allow_html=True)