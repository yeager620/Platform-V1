import asyncio
import json
import datetime
from datetime import datetime, timedelta

import aiohttp
import numpy as np
import requests
import pandas as pd
from collections import defaultdict
import logging
from tqdm import tqdm  # Import tqdm for progress bars
from tqdm.asyncio import tqdm_asyncio

# Configure logging at the beginning of your script or module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping for Batting Stats
BATTING_STAT_MAP = {
    "gamesPlayed": "B_G",
    "plateAppearances": "B_PA",
    "atBats": "B_AB",
    "runs": "B_R",
    "hits": "B_H",
    "totalBases": "B_TB",
    "doubles": "B_2B",
    "triples": "B_3B",
    "homeRuns": "B_HR",
    "rbi": "B_RBI",
    "baseOnBalls": "B_BB",
    "intentionalWalks": "B_IBB",
    "strikeOuts": "B_SO",
    "groundIntoDoublePlay": "B_GDP",
    "groundIntoTriplePlay": "B_GTP",
    "hitByPitch": "B_HP",
    "caughtStealing": "B_CS",
    "stolenBases": "B_SB",
    # "stolenBasePercentage": "B_SB_PCT",
    "leftOnBase": "B_LOB",
    "sacBunts": "B_SH",
    "sacFlies": "B_SF",
    "catchersInterference": "B_XI",
    "pickoffs": "B_PK",
    # "atBatsPerHomeRun": "B_AB_HR",
    "flyOuts": "B_FO",
    "groundOuts": "B_GO"
}

# Mapping for Pitching Stats
PITCHING_STAT_MAP = {
    "gamesPlayed": "P_G",
    "gamesStarted": "P_GS",
    "groundOuts": "P_GO",
    "airOuts": "P_AO",
    "runs": "P_R",
    "doubles": "P_2B",
    "triples": "P_3B",
    "homeRuns": "P_HR",
    "strikeOuts": "P_SO",
    "baseOnBalls": "P_BB",
    "intentionalWalks": "P_IBB",
    "hits": "P_H",
    "hitByPitch": "P_HP",
    "atBats": "P_AB",
    # "obp": "P_OBP",
    "caughtStealing": "P_CS",
    "stolenBases": "P_SB",
    # "stolenBasePercentage": "P_SB_PCT",
    "numberOfPitches": "P_PITCH",
    # "era": "P_ERA",
    "inningsPitched": "P_OUT",  # Convert innings to outs if applicable / necessary
    "wins": "P_W",
    "losses": "P_L",
    "saves": "P_SV",
    "saveOpportunities": "P_SVO",
    "holds": "P_HOLD",
    "blownSaves": "P_BLSV",
    "earnedRuns": "P_ER",
    # "whip": "P_WHIP",
    "battersFaced": "P_TBF",
    "outs": "P_OUTS",
    "completeGames": "P_CG",
    "shutouts": "P_SHO",
    "pitchesThrown": "P_PITCHES",
    "balls": "P_BALLS",
    "strikes": "P_STRIKES",
    # "strikePercentage": "P_STRIKE_PCT",
    "hitBatsmen": "P_HBP",
    "balks": "P_BK",
    "wildPitches": "P_WP",
    "pickoffs": "P_PK",
    # "groundOutsToAirouts": "P_GO_AO",
    "rbi": "P_RBI",
    # "winPercentage": "P_W_PCT",
    # "pitchesPerInning": "P_PITCHES_IP",
    "gamesFinished": "P_GF",
    # "strikeoutWalkRatio": "P_SO_BB",
    # "strikeoutsPer9Inn": "P_SO9",
    # "walksPer9Inn": "P_BB9",
    # "hitsPer9Inn": "P_H9",
    # "runsScoredPer9": "P_R9",
    # "homeRunsPer9": "P_HR9",
    "inheritedRunners": "P_IR",
    "inheritedRunnersScored": "P_IRS",
    "catchersInterference": "P_CI",
    "sacBunts": "P_SH",
    "sacFlies": "P_SF",
    "passedBall": "P_PB",
}

# Mapping for Fielding Stats based on Position Abbreviation
FIELDING_STAT_MAP = {
    'P': {
        "gamesStarted": "F_P_GS",
        "chances": "F_P_TC",
        "putOuts": "F_P_PO",
        "assists": "F_P_A",
        "errors": "F_P_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    },
    'C': {
        "gamesStarted": "F_C_GS",
        "chances": "F_C_TC",
        "putOuts": "F_C_PO",
        "assists": "F_C_A",
        "errors": "F_C_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    },
    '1B': {
        "gamesStarted": "F_1B_GS",
        "chances": "F_1B_TC",
        "putOuts": "F_1B_PO",
        "assists": "F_1B_A",
        "errors": "F_1B_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    },
    '2B': {
        "gamesStarted": "F_2B_GS",
        "chances": "F_2B_TC",
        "putOuts": "F_2B_PO",
        "assists": "F_2B_A",
        "errors": "F_2B_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    },
    '3B': {
        "gamesStarted": "F_3B_GS",
        "chances": "F_3B_TC",
        "putOuts": "F_3B_PO",
        "assists": "F_3B_A",
        "errors": "F_3B_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    },
    'SS': {
        "gamesStarted": "F_SS_GS",
        "chances": "F_SS_TC",
        "putOuts": "F_SS_PO",
        "assists": "F_SS_A",
        "errors": "F_SS_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    },
    'LF': {
        "gamesStarted": "F_LF_GS",
        "chances": "F_LF_TC",
        "putOuts": "F_LF_PO",
        "assists": "F_LF_A",
        "errors": "F_LF_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    },
    'CF': {
        "gamesStarted": "F_CF_GS",
        "chances": "F_CF_TC",
        "putOuts": "F_CF_PO",
        "assists": "F_CF_A",
        "errors": "F_CF_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    },
    'RF': {
        "gamesStarted": "F_RF_GS",
        "chances": "F_RF_TC",
        "putOuts": "F_RF_PO",
        "assists": "F_RF_A",
        "errors": "F_RF_E",
        "caughtStealing": "N/A",
        "stolenBases": "N/A"
    }
}


def get_current_date():
    """
    Retrieves the current date in "MM/DD/YYYY" format.

    Returns:
        str: Current date as a string.
    """
    return datetime.now().strftime("%m/%d/%Y")


class SavantVectorGeneratorV3:
    def __init__(self, start_date, end_date=get_current_date()):
        """
        Version 1.2

        Generates mlb player statistic feature vectors for all games in a set date range using BaseballSavant

        Parameters:
            start_date (str): The start date in "MM/DD/YYYY" format.
            end_date (str): The end date in "MM/DD/YYYY" format.
        """
        self.start_date = start_date  # Expected format: "MM/DD/YYYY"
        self.end_date = end_date  # Expected format: "MM/DD/YYYY"
        self.base_schedule_url = (
            "https://statsapi.mlb.com/api/v1/schedule?"
            "sportId=1&startDate={start_date}&endDate={end_date}"
            "&fields=dates,date,games,gamePk"
        )
        self.base_gamelog_url = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
        self.gamelogs = self.get_unique_game_jsons()
        self.batter_data = "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-PlayerStatLogs/batters_2021_2024.csv"
        self.pitcher_data = "/Users/yeager/Desktop/Maquoketa-Platform-V1/y-data/v1.2-PlayerStatLogs/pitchers_2021_2024.csv"

    def reconstruct_player_stats(self, game_id, game_json, player_info, is_home, is_pitcher=False, half_life=5):
        """
        Reconstructs a statistics dictionary for a single player using half-life weighted aggregation.

        Parameters:
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            player_info (dict): Player information from the JSON.
            is_home (bool): Flag indicating if the player is on the home team.
            is_pitcher (bool): Indicates if the player is a pitcher.
            half_life (int): The number of games after which weights decay to 50%.

        Returns:
            dict: A dictionary representing the statistics for the player up to the game date.
        """
        stats_dict = {}

        # Basic Fields
        position_abb = player_info.get('position', {}).get('abbreviation', 'None')
        stats_dict['game_id'] = game_id
        game_date = game_json.get('gameDate', '')
        try:
            parsed_date = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
        except ValueError:
            parsed_date = '1970-01-01'  # Default if parsing fails
        stats_dict['date'] = parsed_date
        stats_dict['appearance_date'] = parsed_date
        stats_dict['team_id'] = player_info.get('parentTeamId', 0)
        stats_dict['player_id'] = player_info.get('person', {}).get('id', 0)
        stats_dict['home_flag'] = 1 if is_home else 0

        # Opponent ID
        home_team_id = game_json['boxscore']['teams']['home']['team']['id']
        away_team_id = game_json['boxscore']['teams']['away']['team']['id']
        stats_dict['opponent_id'] = away_team_id if is_home else home_team_id

        # Park ID
        stats_dict['park_id'] = game_json.get('park', {}).get('id', 0)

        # Load player data
        if is_pitcher:
            data_source = pd.read_csv(self.pitcher_data)
            stat_map = PITCHING_STAT_MAP
        else:
            data_source = pd.read_csv(self.batter_data)
            stat_map = BATTING_STAT_MAP

        player_data = data_source[
            (data_source['player_id'] == stats_dict['player_id']) &
            (data_source['date'] < parsed_date)
            ].sort_values(by='date', ascending=False)

        if player_data.empty:
            logging.warning(
                f"No data found for player {stats_dict['player_id']} before {parsed_date}. Returning Nones.")
            return {stat: None for stat in stat_map.values()}

        # Compute weights using half-life
        weights = np.exp(-np.log(2) * np.arange(len(player_data)) / half_life)
        weights /= weights.sum()  # Normalize weights to sum to 1

        # Aggregate stats using weighted average
        for stat_key, stat_column in stat_map.items():
            if stat_key in player_data.columns:
                stats_dict[stat_column] = np.dot(weights, player_data[stat_key].fillna(0))
            else:
                stats_dict[stat_column] = None  # Default if stat not available

        return stats_dict

    def process_games_retrosheet_with_outcome(self):
        """
        Processes all games in chronological order, constructs the Retrosheet DataFrame,
        and adds the 'Home_Win' target variable.

        Ensures that feature vectors for each game only include data from previous games,
        preventing lookahead bias.

        Returns:
            pd.DataFrame: DataFrame containing concatenated Retrosheet rows for all games with 'Home_Win' column.
        """
        # Process games to get the Retrosheet DataFrame
        retrosheet_df = self.process_games_retrosheet()

        # Add the game outcome target variable
        retrosheet_df = self.add_game_outcome(retrosheet_df)

        # Reorder columns to have specific ones at the beginning
        first_columns = [
            'Game_Date', 'Game_PK', 'Home_Team_Abbr', 'Away_Team_Abbr',
            'park_id', 'Home_Win'
        ]

        # Keep other columns in their original order
        other_columns = [col for col in retrosheet_df.columns if col not in first_columns]

        # Reorder the DataFrame
        retrosheet_df = retrosheet_df[first_columns + other_columns]

        return retrosheet_df

    def aggregate_team_retrosheet_stats(self, starting_batters, starting_pitcher, bullpen, bench, game_id, game_json,
                                        team_type):
        """
        Aggregates Retrosheet statistics for an entire team based on individual player stats using cumulative stats.

        Parameters:
            starting_batters (list of int): List of player IDs in the team's starting batting lineup
            starting_pitcher (int): Probable pitcher id
            bullpen (list of int): List of player ids in team pitcher bullpen (possible pitcher substitutions)
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            team_type (str): 'home' or 'away'.

        Returns:
            dict: A dictionary representing the aggregated Retrosheet statistics for the team.
        """
        aggregated_stats = defaultdict(float)

        batting_order_map = {}
        batting_order_set = set()
        allowed_order_values = set(range(1, 10))

        no_order_map = {}

        num_batters = 0
        num_pitchers = 0
        num_infielders = 0
        num_outfielders = 0

        if starting_pitcher not in starting_batters:
            starting_lineup = starting_batters + [starting_pitcher]
        else:
            starting_lineup = starting_batters

        # Determine if there is a DH in the lineup
        has_dh = False
        for player_id in starting_batters:
            player_info = self.get_player_info(player_id, game_json, team_type)
            if player_info is None:
                continue
            batting_order_map[player_id] = int(player_info.get('battingOrder', -100)) // 100
            print(f"batting order for player {player_id} is {batting_order_map[player_id]}")
            if batting_order_map[player_id] == -1:
                no_order_map[player_id] = player_info.get('seasonStats', {}).get('batting', {}).get('gamesPlayed', 1)
            position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
            if position_abbr == 'DH':
                has_dh = True
                break  # No need to continue once a DH is found

        for player_id in starting_lineup:
            player_info = self.get_player_info(player_id, game_json, team_type)
            if player_info is None:
                print(f"Player info not found for player_id {player_id} in team {team_type}")
                continue

            position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
            player_stats = self.reconstruct_player_stats(
                game_id=game_id,
                game_json=game_json,
                player_info=player_info,
                is_home=(team_type == 'home')
            )

            # Determine if player is a pitcher
            is_pitcher = position_abbr == 'P'

            # Determine if player is a batter
            if has_dh:
                # POSSIBLY INCORRECT: If there is a DH, the pitcher is not a batter
                print("Starting lineup has DH")
                is_batter = not is_pitcher
            else:
                # POSSIBLY INCORRECT: If there is no DH, all players are batters
                is_batter = not is_pitcher

            # Aggregate Batting Stats
            if is_batter:
                num_batters += 1
                batting_order = batting_order_map.get(player_id, -1)

                if batting_order in allowed_order_values:
                    if batting_order not in batting_order_set:
                        # Assign stats to the correct slot
                        batting_order_set.add(batting_order)
                        for stat in BATTING_STAT_MAP.keys():
                            aggregated_stats[f"B{batting_order}_{stat}"] = player_stats.get(f"B_{stat}", 0)
                    else:
                        # Duplicate batting order: Add to no_order_map for reassignment later
                        no_order_map[player_id] = player_info.get('seasonStats', {}).get('batting', {}).get(
                            'gamesPlayed', 1)
                else:
                    # Invalid or missing batting order: Add to no_order_map
                    no_order_map[player_id] = player_info.get('seasonStats', {}).get('batting', {}).get('gamesPlayed',
                                                                                                        1)

                # Aggregate Fielding Stats
                if position_abbr in FIELDING_STAT_MAP:
                    if position_abbr in ['1B', '2B', '3B', 'SS']:
                        num_infielders += 1
                        for stat in FIELDING_STAT_MAP[position_abbr].keys():
                            aggregated_stats[f"F_IF_{stat}"] += player_stats.get(f"F_{stat}", 0)
                    if position_abbr in ['LF', 'CF', 'RF']:
                        num_outfielders += 1
                        for stat in FIELDING_STAT_MAP[position_abbr].keys():
                            aggregated_stats[f"F_OF_{stat}"] += player_stats.get(f"F_{stat}", 0)
                else:
                    if position_abbr != 'DH':  # DH doesn't have fielding stats
                        print(f"Unrecognized position {position_abbr} for player_id {player_id}")

            # Aggregate Pitching Stats
            if is_pitcher:
                num_pitchers += 1
                for stat in PITCHING_STAT_MAP.keys():
                    aggregated_stats[f"P_{stat}"] += player_stats.get(f"P_{stat}", 0)

        num_subs = 0
        total_sub_innings_pitched = 0

        for player_id in bullpen:
            player_info = self.get_player_info(player_id, game_json, team_type)
            if player_info is None:
                print(f"Player info not found for bullpen pitcher player_id {player_id} in team {team_type}")
                continue
            num_subs += 1

            position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
            player_stats = self.reconstruct_player_stats(
                game_id=game_id,
                game_json=game_json,
                player_info=player_info,
                is_home=(team_type == 'home')
            )
            try:
                season_innings_pitched = float(
                    player_info.get('seasonStats', {}).get('pitching', {}).get('inningsPitched', 0))
                game_innings_pitched = float(player_info.get('stats', {}).get('pitching', {}).get('inningsPitched', 0))
                sub_innings_pitched = season_innings_pitched - game_innings_pitched
                total_sub_innings_pitched += sub_innings_pitched
            except (ValueError, TypeError):
                sub_innings_pitched = 0  # Fallback for invalid innings values

            for stat in PITCHING_STAT_MAP.keys():
                # SP: Substitution Pitcher
                aggregated_stats[f"SP_{stat}"] += player_stats.get(f"P_{stat}", 0) * sub_innings_pitched

        '''
        num_bench = 0
        total_bench_at_bats = 0

        for player_id in bench:
            player_info = self.get_player_info(player_id, game_json, team_type)
            if player_info is None:
                print(f"Player info not found for benched batter player_id {player_id} in team {team_type}")
                continue
            num_bench += 1

            position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
            player_stats = self.reconstruct_player_stats(
                game_id=game_id,
                game_json=game_json,
                player_info=player_info,
                is_home=(team_type == 'home')
            )
            try:
                season_at_bats = float(
                    player_info.get('seasonStats', {}).get('batting', {}).get('atBats', 0))
                game_at_bats = float(
                    player_info.get('stats', {}).get('batting', {}).get('atBats', 0))
                bench_at_bats = season_at_bats - game_at_bats
                total_bench_at_bats += bench_at_bats
            except (ValueError, TypeError):
                bench_at_bats = 0  # Fallback for invalid innings values

            for stat in BATTING_STAT_MAP.keys():
                # SP: Substitution Pitcher
                aggregated_stats[f"SB_{stat}"] += player_stats.get(f"B_{stat}", 0) * bench_at_bats
        '''

        if len(no_order_map) != (9 - len(batting_order_map)):
            print("Warning: Length of batters without batting order not equal to (9 - size of the batting order map)")

        empty_slots = sorted(allowed_order_values - batting_order_set)
        for player_id, games_played in sorted(no_order_map.items(), key=lambda item: item[1], reverse=True):
            if not empty_slots:
                print(f"Warning: Not enough slots for all players without batting orders.")
                break

            slot = empty_slots.pop(0)  # Assign the smallest available slot
            batting_order_set.add(slot)

            # Add player's stats to the corresponding slot in aggregated_stats
            player_info = self.get_player_info(player_id, game_json, team_type)
            player_stats = self.reconstruct_player_stats(
                game_id=game_id,
                game_json=game_json,
                player_info=player_info,
                is_home=(team_type == 'home')
            )
            for stat in BATTING_STAT_MAP.keys():
                aggregated_stats[f"B{slot}_{stat}"] = player_stats.get(f"B_{stat}", 0)

        # Verify all slots are filled
        if len(batting_order_set) < 9:
            print(
                f"Warning: Only {len(batting_order_set)} slots filled. Remaining slots will be filled with pitcher statistics or be left blank.")

            slot = empty_slots.pop(0)  # Assign the smallest available slot
            batting_order_set.add(slot)

            # Add player's stats to the corresponding slot in aggregated_stats
            player_info = self.get_player_info(starting_pitcher, game_json, team_type)
            player_stats = self.reconstruct_player_stats(
                game_id=game_id,
                game_json=game_json,
                player_info=player_info,
                is_home=(team_type == 'home')
            )
            for stat in BATTING_STAT_MAP.keys():
                aggregated_stats[f"B{slot}_{stat}"] = player_stats.get(f"B_{stat}", 0)

        # Normalize Bullpen Pitching Stats
        if num_subs > 0 and total_sub_innings_pitched > 0:
            for stat in PITCHING_STAT_MAP.keys():
                aggregated_stats[f"SP_{stat}"] /= total_sub_innings_pitched

        '''
        if num_bench > 0 and total_bench_at_bats > 0:
            for stat in BATTING_STAT_MAP.keys():
                aggregated_stats[f"SB_{stat}"] /= total_bench_at_bats
        '''

        # Normalize Fielding Stats

        # num_fielders = num_batters  # Fielders have the same number as batters (excluding DH)

        if num_infielders > 0:
            for stat in FIELDING_STAT_MAP['1B'].keys():
                if f"F_IF_{stat}" in aggregated_stats:
                    aggregated_stats[f"F_IF_{stat}"] /= num_infielders
        if num_outfielders > 0:
            for stat in FIELDING_STAT_MAP['LF'].keys():
                if f"F_OF_{stat}" in aggregated_stats:
                    aggregated_stats[f"F_IF_{stat}"] /= num_outfielders

        # Convert defaultdict to regular dict
        aggregated_stats = dict(aggregated_stats)

        return aggregated_stats

    def process_games_retrosheet(self):
        """
        Processes all games in chronological order and returns a DataFrame where each row represents a game.
        Each row concatenates the Retrosheet rows of the home and away teams,
        with the home team on the left and the away team on the right.
        Includes team names and abbreviations for both teams.

        Ensures that feature vectors for each game only include data from previous games,
        preventing lookahead bias.

        Returns:
            pd.DataFrame: DataFrame containing concatenated Retrosheet rows for all games.
        """
        # Sort games in chronological order based on gameDate
        gamelogs_sorted = sorted(self.gamelogs, key=lambda x: datetime.strptime(x['gameDate'], "%m/%d/%Y"))

        retrosheet_rows = []

        for idx, game_json in enumerate(tqdm(gamelogs_sorted, desc="Processing games"), start=1):
            game_id = game_json['scoreboard']['gamePk']
            game_date = game_json['gameDate']

            # Get park ID
            park_id = game_json.get('scoreboard', {}).get('teams', {}).get('home', {}).get('venue', {}).get('id',
                                                                                                            'Unknown')

            # Extract Home Team Information
            home_team_info = game_json['home_team_data']
            home_team_id = home_team_info['id']
            home_team_name = home_team_info['name']
            home_team_abbr = home_team_info.get('abbreviation', '')

            # Extract Away Team Information
            away_team_info = game_json['away_team_data']
            away_team_id = away_team_info['id']
            away_team_name = away_team_info['name']
            away_team_abbr = away_team_info.get('abbreviation', '')

            # Get Home and Away Lineups (List of player_ids)
            home_lineup = game_json['home_lineup']
            away_lineup = game_json['away_lineup']

            home_batters = game_json['boxscore']['teams']['home']['batters']
            away_batters = game_json['boxscore']['teams']['away']['batters']

            def filter_batters(batter_ids, game_json, team):
                return [
                    batter_id
                    for batter_id in batter_ids
                    if game_json['boxscore']['teams'][team]['players'][f"ID{batter_id}"]['stats']['fielding'].get(
                        'gamesStarted', 0) >= 1
                ]

            def exclude_new_batters(batter_ids, game_json, team):
                filtered_batters = []
                for batter_id in batter_ids:
                    # Get player info
                    player_info = game_json['boxscore']['teams'][team]['players'].get(f"ID{batter_id}")
                    if not player_info:
                        continue

                    # Check if the player has cumulative stats for the current season
                    season_stats = player_info.get('seasonStats', {}).get('batting', {})
                    current_game_state = player_info.get('stats', {}).get('batting', {})
                    previous_games_played = season_stats.get('gamesPlayed', 0) - current_game_state.get('gamesPlayed',
                                                                                                        0)

                    # Include batter only if they have stats for the current season
                    if previous_games_played > 0:
                        filtered_batters.append(batter_id)

                return filtered_batters

            # Filter home and away batters
            starting_home_batters = filter_batters(
                home_batters,
                game_json,
                'home'
            )

            starting_away_batters = exclude_new_batters(
                away_batters,
                game_json,
                'away'
            )

            valid_starting_home_batters = exclude_new_batters(
                starting_home_batters,
                game_json,
                'home'
            )

            valid_starting_away_batters = filter_batters(
                starting_away_batters,
                game_json,
                'away'
            )

            home_batting_order = game_json['boxscore']['teams']['home']['battingOrder']
            away_batting_order = game_json['boxscore']['teams']['away']['battingOrder']

            home_bench = game_json['boxscore']['teams']['home']['bench']
            away_bench = game_json['boxscore']['teams']['away']['bench']

            # Get home and away probable pitchers: Use as starting pitcher
            # TODO: Implement edge case handling for no available probable pitchers
            home_probable_pitcher = game_json['scoreboard']['probablePitchers'].get('home', {}).get('id', None)
            away_probable_pitcher = game_json['scoreboard']['probablePitchers'].get('away', {}).get('id', None)

            home_pitchers = game_json['boxscore']['teams']['home']['pitchers']
            away_pitchers = game_json['boxscore']['teams']['away']['pitchers']

            home_bullpen_pitchers = game_json['boxscore']['teams']['home']['bullpen']
            home_bullpen_pitchers = [pitcher for pitcher in home_bullpen_pitchers if pitcher != home_probable_pitcher]
            away_bullpen_pitchers = game_json['boxscore']['teams']['away']['bullpen']
            away_bullpen_pitchers = [pitcher for pitcher in away_bullpen_pitchers if pitcher != away_probable_pitcher]

            home_bench_batters = game_json['boxscore']['teams']['home']['bench']
            home_bench_batters = [batter for batter in home_bench_batters if batter not in valid_starting_home_batters]
            away_bench_batters = game_json['boxscore']['teams']['away']['bench']
            away_bench_batters = [batter for batter in away_bench_batters if batter not in valid_starting_away_batters]

            # Log the number of excluded batters for debugging
            excluded_home_batters = len(starting_home_batters) - len(valid_starting_home_batters)
            excluded_away_batters = len(starting_away_batters) - len(valid_starting_away_batters)
            print(
                f"Warning (gamePk {game_id}): Excluded {excluded_home_batters} home batters and {excluded_away_batters} away batters without stats.")

            # Aggregate Home Team Retrosheet Stats using cumulative stats
            home_retrosheet_dict = self.aggregate_team_retrosheet_stats(
                starting_batters=valid_starting_home_batters,
                starting_pitcher=home_probable_pitcher,
                bullpen=home_bullpen_pitchers,
                bench=home_bench_batters,
                game_id=game_id,
                game_json=game_json,
                team_type='home'
            )

            # Aggregate Away Team Retrosheet Stats using cumulative stats
            away_retrosheet_dict = self.aggregate_team_retrosheet_stats(
                starting_batters=valid_starting_away_batters,
                starting_pitcher=away_probable_pitcher,
                bullpen=away_bullpen_pitchers,
                bench=away_bench_batters,
                game_id=game_id,
                game_json=game_json,
                team_type='away'
            )

            # Prefix stat names with 'Home_' and 'Away_'
            home_prefixed = {f"Home_{key}": value for key, value in home_retrosheet_dict.items()}
            away_prefixed = {f"Away_{key}": value for key, value in away_retrosheet_dict.items()}

            # Merge dictionaries correctly using dictionary unpacking (Python 3.5+)
            concatenated_row = {**home_prefixed, **away_prefixed}

            # Include Team Names and Abbreviations, and gamePk and gameDate
            concatenated_row.update({
                'Game_Date': game_date,
                'Game_PK': game_id,
                'Home_Team_Abbr': home_team_abbr,
                'Away_Team_Abbr': away_team_abbr,
                'park_id': park_id,
            })

            retrosheet_rows.append(concatenated_row)

        logger.info(f"Total games processed: {len(retrosheet_rows)}")

        # Create DataFrame
        df = pd.DataFrame(retrosheet_rows)

        return df

    def add_game_outcome(self, retrosheet_df):
        """
        Adds a target variable indicating whether the home team won (1) or lost (0) to the Retrosheet DataFrame.

        Parameters:
            retrosheet_df (pd.DataFrame): DataFrame containing Retrosheet game feature vectors with 'Game_PK' column.

        Returns:
            pd.DataFrame: Updated DataFrame with 'Home_Win' column added.
        """

        # Create a mapping from gamePk to home team win (1) or loss (0)
        game_outcome = {}
        for game_json in self.gamelogs:
            game_pk = game_json['scoreboard']['gamePk']
            try:
                # Extract final scores
                home_runs = game_json['scoreboard']['linescore']['teams']['home']['runs']
                away_runs = game_json['scoreboard']['linescore']['teams']['away']['runs']
                # Determine if home team won
                home_win = 1 if home_runs >= away_runs else 0
                game_outcome[game_pk] = home_win
            except Exception as e:
                print(f"Error processing game_pk {game_pk}: {e}")
                game_outcome[game_pk] = None  # Default to None if error occurs

        # Map the 'Game_PK' column to 'Home_Win' using the game_outcome dictionary
        retrosheet_df['Home_Win'] = retrosheet_df['Game_PK'].map(game_outcome).fillna(0).astype(int)

        return retrosheet_df

    def update_cumulative_stats_after_game(self, game_json, home_lineup, away_lineup,
                                           cumulative_player_stats, cumulative_team_stats):
        """
        Updates cumulative statistics for all players and teams after processing a game.

        Parameters:
            game_json (dict): JSON data for the game.
            home_lineup (list of int): List of player IDs for the home team.
            away_lineup (list of int): List of player IDs for the away team.
            cumulative_player_stats (dict): Cumulative player statistics.
            cumulative_team_stats (dict): Cumulative team statistics.
        """
        for team_type, lineup in [('home', home_lineup), ('away', away_lineup)]:
            for player_id in lineup:
                player_info = self.get_player_info(player_id, game_json, team_type)
                if player_info is None:
                    continue  # Skip if player not found

                # Extract current game stats
                batting_stats = player_info.get('seasonStats', {}).get('batting', {})
                pitching_stats = player_info.get('seasonStats', {}).get('pitching', {})
                fielding_stats = player_info.get('seasonStats', {}).get('fielding', {})
                position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()

                # Update batting stats
                for key in BATTING_STAT_MAP.keys():
                    cumulative_player_stats[player_id][key] += batting_stats.get(key, 0)

                # Update pitching stats
                for key in PITCHING_STAT_MAP.keys():
                    if key == 'inningsPitched':
                        innings = pitching_stats.get(key, 0.0)
                        try:
                            whole_innings = int(innings)
                            fractional = innings - whole_innings
                            outs_recorded = whole_innings * 3 + int(round(fractional * 10 / 3))
                            cumulative_player_stats[player_id][key] += outs_recorded / 3
                        except:
                            cumulative_player_stats[player_id][key] += 0
                    else:
                        cumulative_player_stats[player_id][key] += pitching_stats.get(key, 0)

                # Update fielding stats
                if position_abbr in FIELDING_STAT_MAP:
                    for key in FIELDING_STAT_MAP[position_abbr].keys():
                        cumulative_player_stats[player_id][f"{position_abbr}_{key}"] += fielding_stats.get(key, 0)

                # Update batting order and sequence as averages
                ab = cumulative_player_stats[player_id].get('atBats', 0)
                pa = cumulative_player_stats[player_id].get('plateAppearances', 0)
                if ab > 0:
                    cumulative_player_stats[player_id]['batting_order'] = pa / ab
                    cumulative_player_stats[player_id]['batting_order_sequence'] = pa / ab
                else:
                    cumulative_player_stats[player_id]['batting_order'] = 0
                    cumulative_player_stats[player_id]['batting_order_sequence'] = 0

    def update_cumulative_team_stats(self, team_id, team_retrosheet_row, cumulative_team_stats):
        """
        Updates the cumulative team statistics with the aggregated team Retrosheet row.

        Parameters:
            team_id (int): Unique identifier for the team.
            team_retrosheet_row (list): Aggregated Retrosheet statistics for the team.
            cumulative_team_stats (dict): Cumulative team statistics.
        """
        # Iterate through team_retrosheet_row and update cumulative_team_stats
        for idx, stat_value in enumerate(team_retrosheet_row):
            stat_name = self.retrosheet_field_names[idx]
            if isinstance(stat_value, (int, float)):
                cumulative_team_stats[team_id][stat_name] += stat_value

    def load_json(self, filepath):
        """
        Loads JSON data from a file.

        Parameters:
            filepath (str): Path to the JSON file.

        Returns:
            list: List of game JSON dictionaries.
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_player_info(self, player_id, game_json, team_type):
        """
        Retrieves the player information dictionary from the game JSON.

        Parameters:
            player_id (int): Unique identifier for the player.
            game_json (dict): JSON data for the game.
            team_type (str): 'home' or 'away'.

        Returns:
            dict or None: Player information dictionary or None if not found.
        """
        team_players = game_json['boxscore']['teams'][team_type]['players']
        for player_key, info in team_players.items():
            if info.get('person', {}).get('id') == player_id:
                return info
        print('player stats not found')
        return None

    @staticmethod
    def has_cumulative_stats(player_cumulative):
        """
        Determines if a player has any cumulative statistics.

        Parameters:
            player_cumulative (defaultdict): Cumulative statistics for the player.

        Returns:
            bool: True if the player has at least one non-zero statistic, False otherwise.
        """
        return any(value != 0 for value in player_cumulative.values())

    @staticmethod
    def get_team_info(game_json, team_type):
        """
        Retrieves the team-level statistics dictionary from the game JSON.

        Parameters:
            game_json (dict): JSON data for the game.
            team_type (str): 'home' or 'away'.

        Returns:
            dict or None: Team information dictionary or None if not found.
        """
        # TODO: Fill in for edge case handling
        return None

    def fetch_game_pks(self):
        """
        Fetches all gamePk's from the start date up to the end date by splitting the date range into monthly chunks.

        Returns:
            list: List of gamePk integers.
        """
        game_pks = []
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        delta = timedelta(days=30)  # Approximate one-month chunks

        current_start = start
        while current_start <= end:
            current_end = min(current_start + delta, end)
            formatted_start = current_start.strftime("%Y-%m-%d")
            formatted_end = current_end.strftime("%Y-%m-%d")
            url = self.base_schedule_url.format(start_date=formatted_start, end_date=formatted_end)

            try:
                response = requests.get(url)
                if response.status_code != 200:
                    logging.error(
                        f"Failed to fetch gamePk's from {formatted_start} to {formatted_end}: {response.status_code}")
                    current_start = current_end + timedelta(days=1)
                    continue

                data = response.json()
                if 'dates' in data:
                    for date_info in data['dates']:
                        for game in date_info.get('games', []):
                            game_pks.append(game['gamePk'])
                logging.info(f"Fetched {len(data.get('dates', []))} dates from {formatted_start} to {formatted_end}")
            except Exception as e:
                logging.error(f"Exception while fetching gamePk's from {formatted_start} to {formatted_end}: {e}")

            current_start = current_end + timedelta(days=1)

        logging.info(f"Total gamePk's fetched: {len(game_pks)}")
        return game_pks

    async def fetch_gamelog_async(self, session, game_pk, pbar):
        """
        Asynchronously fetches game log JSON for a specific gamePk.

        Parameters:
            session (aiohttp.ClientSession): The aiohttp session object.
            game_pk (int): Unique identifier for the game.
            pbar (tqdm): Progress bar object to update.

        Returns:
            dict: Game log JSON data.
        """
        url = self.base_gamelog_url.format(game_pk=game_pk)
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch gamelog for game_pk {game_pk}: {response.status}")
                    data = {}
                else:
                    data = await response.json()
        except Exception as e:
            logger.error(f"Exception occurred while fetching game_pk {game_pk}: {e}")
            data = {}
        finally:
            pbar.update(1)  # Update progress bar
        return data

    async def get_game_jsons_async(self, game_pks):
        """
        Asynchronously retrieves game JSONs for the provided game_pks.

        Parameters:
            game_pks (list): List of gamePk integers.

        Returns:
            list: List of game JSON dictionaries.
        """
        tasks = []
        connector = aiohttp.TCPConnector(limit=10)  # Limit the number of concurrent connections
        timeout = aiohttp.ClientTimeout(total=60)  # Set total timeout for all requests
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            with tqdm_asyncio(total=len(game_pks), desc="Fetching game logs") as pbar:
                for game_pk in game_pks:
                    task = asyncio.ensure_future(self.fetch_gamelog_async(session, game_pk, pbar))
                    tasks.append(task)
                game_jsons = await asyncio.gather(*tasks)
        return game_jsons

    def get_unique_game_jsons(self):
        """
        Retrieves a list of unique game JSONs for the date range from start_date to end_date.

        Returns:
            list: List of game JSON dictionaries.
        """
        game_pks = self.fetch_game_pks()
        # Use asyncio.run() if Python >= 3.7
        try:
            game_jsons = asyncio.run(self.get_game_jsons_async(game_pks))
        except RuntimeError:
            # If an event loop is already running (e.g., in Jupyter), create a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            game_jsons = loop.run_until_complete(self.get_game_jsons_async(game_pks))
        # Filter out empty results
        game_jsons = [game for game in game_jsons if game]
        logging.info(f"Total game JSONs fetched: {len(game_jsons)}")
        return game_jsons