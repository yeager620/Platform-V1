import json
import datetime
from datetime import datetime, timedelta
import requests
import pandas as pd
from collections import defaultdict

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
    "grandSlams": "B_HR4",
    "rbi": "B_RBI",
    "gameWinningRbi": "B_GW",
    "baseOnBalls": "B_BB",
    "intentionalWalks": "B_IBB",
    "strikeOuts": "B_SO",
    "groundIntoDoublePlay": "B_GDP",
    "hitByPitch": "B_HP",
    "sacBunts": "B_SH",
    "sacFlies": "B_SF",
    "stolenBases": "B_SB",
    "caughtStealing": "B_CS",
    "catchersInterference": "B_XI",
    "gamesAsPH": "B_G_PH",
    "gamesAsPR": "B_G_PR"
}

# Mapping for Pitching Stats
PITCHING_STAT_MAP = {
    "gamesPlayed": "P_G",
    "gamesStarted": "P_GS",
    "completeGames": "P_CG",
    "shutouts": "P_SHO",
    "gamesFinished": "P_GF",
    "wins": "P_W",
    "losses": "P_L",
    "saves": "P_SV",
    "battersFaced": "P_TBF",
    "atBats": "P_AB",
    "runs": "P_R",
    "earnedRuns": "P_ER",
    "hits": "P_H",
    "totalBases": "P_TB",
    "doubles": "P_2B",
    "triples": "P_3B",
    "homeRuns": "P_HR",
    "grandSlamsAllowed": "P_HR4",
    "walks": "P_BB",
    "intentionalWalks": "P_IBB",
    "strikeOuts": "P_SO",
    "groundIntoDoublePlay": "P_GDP",
    "hitBatsmen": "P_HP",
    "sacHitsAgainst": "P_SH",
    "sacFliesAgainst": "P_SF",
    "reachedOnInterference": "P_XI",
    "wildPitches": "P_WP",
    "balks": "P_BK",
    "inheritedRunners": "P_IR",
    "inheritedRunnersScored": "P_IRS",
    "groundOuts": "P_GO",
    "airOuts": "P_AO",
    "numberOfPitches": "P_PITCH",
    "strikes": "P_STRIKE",
    "gamesAtP": "F_P_G",
    "gamesStartedAtP": "F_P_GS",
    "inningsPitched": "P_OUT"  # This will be converted to outs
}

# Mapping for Fielding Stats based on Position Abbreviation
FIELDING_STAT_MAP = {
    'P': {
        "games": "F_P_G",
        "gamesStarted": "F_P_GS",
        "outsRecorded": "F_P_OUT",
        "totalChances": "F_P_TC",
        "putOuts": "F_P_PO",
        "assists": "F_P_A",
        "errors": "F_P_E",
        "doublePlays": "F_P_DP",
        "triplePlays": "F_P_TP"
    },
    'C': {
        "games": "F_C_G",
        "gamesStarted": "F_C_GS",
        "outsRecorded": "F_C_OUT",
        "totalChances": "F_C_TC",
        "putOuts": "F_C_PO",
        "assists": "F_C_A",
        "errors": "F_C_E",
        "doublePlays": "F_C_DP",
        "triplePlays": "F_C_TP",
        "passedBalls": "F_C_PB",
        "catchersInterference": "F_C_IX"
    },
    '1B': {
        "games": "F_1B_G",
        "gamesStarted": "F_1B_GS",
        "outsRecorded": "F_1B_OUT",
        "totalChances": "F_1B_TC",
        "putOuts": "F_1B_PO",
        "assists": "F_1B_A",
        "errors": "F_1B_E",
        "doublePlays": "F_1B_DP",
        "triplePlays": "F_1B_TP"
    },
    '2B': {
        "games": "F_2B_G",
        "gamesStarted": "F_2B_GS",
        "outsRecorded": "F_2B_OUT",
        "totalChances": "F_2B_TC",
        "putOuts": "F_2B_PO",
        "assists": "F_2B_A",
        "errors": "F_2B_E",
        "doublePlays": "F_2B_DP",
        "triplePlays": "F_2B_TP"
    },
    '3B': {
        "games": "F_3B_G",
        "gamesStarted": "F_3B_GS",
        "outsRecorded": "F_3B_OUT",
        "totalChances": "F_3B_TC",
        "putOuts": "F_3B_PO",
        "assists": "F_3B_A",
        "errors": "F_3B_E",
        "doublePlays": "F_3B_DP",
        "triplePlays": "F_3B_TP"
    },
    'SS': {
        "games": "F_SS_G",
        "gamesStarted": "F_SS_GS",
        "outsRecorded": "F_SS_OUT",
        "totalChances": "F_SS_TC",
        "putOuts": "F_SS_PO",
        "assists": "F_SS_A",
        "errors": "F_SS_E",
        "doublePlays": "F_SS_DP",
        "triplePlays": "F_SS_TP"
    },
    'LF': {
        "games": "F_LF_G",
        "gamesStarted": "F_LF_GS",
        "outsRecorded": "F_LF_OUT",
        "totalChances": "F_LF_TC",
        "putOuts": "F_LF_PO",
        "assists": "F_LF_A",
        "errors": "F_LF_E",
        "doublePlays": "F_LF_DP",
        "triplePlays": "F_LF_TP"
    },
    'CF': {
        "games": "F_CF_G",
        "gamesStarted": "F_CF_GS",
        "outsRecorded": "F_CF_OUT",
        "totalChances": "F_CF_TC",
        "putOuts": "F_CF_PO",
        "assists": "F_CF_A",
        "errors": "F_CF_E",
        "doublePlays": "F_CF_DP",
        "triplePlays": "F_CF_TP"
    },
    'RF': {
        "games": "F_RF_G",
        "gamesStarted": "F_RF_GS",
        "outsRecorded": "F_RF_OUT",
        "totalChances": "F_RF_TC",
        "putOuts": "F_RF_PO",
        "assists": "F_RF_A",
        "errors": "F_RF_E",
        "doublePlays": "F_RF_DP",
        "triplePlays": "F_RF_TP"
    }
}


def get_current_date():
    """
    Retrieves the current date in "MM/DD/YYYY" format.

    Returns:
        str: Current date as a string.
    """
    return datetime.now().strftime("%m/%d/%Y")


class SavantRetrosheetConverter:
    def __init__(self, start_date, end_date=get_current_date()):
        """
        Initializes the SavantRetrosheetConverter with a start date.

        Parameters:
            start_date (str): The start date in "MM/DD/YYYY" format.
        """
        self.start_date = start_date  # Expected format: "MM/DD/YYYY"
        self.end_date = end_date  # Expected format: "MM/DD/YYYY"
        self.base_schedule_url = (
            "https://statsapi.mlb.com/api/v1/schedule?"
            "sportId=1&startDate={start_date}&endDate={end_date}"
            "&fields=dates,date,games,gamePk"
        )
        self.base_gamelog_url = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
        self.retrosheet_field_names = [
            "game_id", "date", "game_number", "appearance_date", "team_id", "player_id",
            "batting_order", "batting_order_sequence", "home_flag", "opponent_id",
            "park_id",
            # Batting stats
            "B_G", "B_PA", "B_AB", "B_R", "B_H", "B_TB", "B_2B", "B_3B",
            "B_HR", "B_HR4", "B_RBI", "B_GW", "B_BB", "B_IBB", "B_SO",
            "B_GDP", "B_HP", "B_SH", "B_SF", "B_SB", "B_CS", "B_XI",
            "B_G_DH", "B_G_PH", "B_G_PR",
            # Pitching stats
            "P_G", "P_GS", "P_CG", "P_SHO", "P_GF", "P_W", "P_L",
            "P_SV", "P_OUT", "P_TBF", "P_AB", "P_R", "P_ER", "P_H",
            "P_TB", "P_2B", "P_3B", "P_HR", "P_HR4", "P_BB",
            "P_IBB", "P_SO", "P_GDP", "P_HP", "P_SH", "P_SF",
            "P_XI", "P_WP", "P_BK", "P_IR", "P_IRS", "P_GO",
            "P_AO", "P_PITCH", "P_STRIKE",
            # Fielding stats for P, C, 1B, 2B, 3B, SS, LF, CF, RF
            # Pitcher
            "F_P_G", "F_P_GS", "F_P_OUT", "F_P_TC", "F_P_PO",
            "F_P_A", "F_P_E", "F_P_DP", "F_P_TP",
            # Catcher
            "F_C_G", "F_C_GS", "F_C_OUT", "F_C_TC", "F_C_PO",
            "F_C_A", "F_C_E", "F_C_DP", "F_C_TP", "F_C_PB", "F_C_IX",
            # First Baseman
            "F_1B_G", "F_1B_GS", "F_1B_OUT", "F_1B_TC", "F_1B_PO",
            "F_1B_A", "F_1B_E", "F_1B_DP", "F_1B_TP",
            # Second Baseman
            "F_2B_G", "F_2B_GS", "F_2B_OUT", "F_2B_TC", "F_2B_PO",
            "F_2B_A", "F_2B_E", "F_2B_DP", "F_2B_TP",
            # Third Baseman
            "F_3B_G", "F_3B_GS", "F_3B_OUT", "F_3B_TC", "F_3B_PO",
            "F_3B_A", "F_3B_E", "F_3B_DP", "F_3B_TP",
            # Shortstop
            "F_SS_G", "F_SS_GS", "F_SS_OUT", "F_SS_TC", "F_SS_PO",
            "F_SS_A", "F_SS_E", "F_SS_DP", "F_SS_TP",
            # Left Fielder
            "F_LF_G", "F_LF_GS", "F_LF_OUT", "F_LF_TC", "F_LF_PO",
            "F_LF_A", "F_LF_E", "F_LF_DP", "F_LF_TP",
            # Center Fielder
            "F_CF_G", "F_CF_GS", "F_CF_OUT", "F_CF_TC", "F_CF_PO",
            "F_CF_A", "F_CF_E", "F_CF_DP", "F_CF_TP",
            # Right Fielder
            "F_RF_G", "F_RF_GS", "F_RF_OUT", "F_RF_TC", "F_RF_PO",
            "F_RF_A", "F_RF_E", "F_RF_DP", "F_RF_TP"
        ]
        self.gamelogs = self.get_unique_game_jsons()

    def reconstruct_retrosheet_row_from_cumulative(self, game_id, game_json, player_key, player_info, is_home,
                                                   cumulative_stats):
        """
        Reconstructs a Retrosheet statistics row for a single player based on cumulative stats.

        Parameters:
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            player_key (str): Player identifier (e.g., "ID451594").
            player_info (dict): Player information from the JSON.
            is_home (bool): Flag indicating if the player is on the home team.
            cumulative_stats (dict): Cumulative statistics for the player up to previous games.

        Returns:
            list: A list representing the Retrosheet statistics row with numeric values based on cumulative stats.
        """
        # Initialize a list with 154 default numeric values (zeros)
        retrosheet_row = [0] * 154

        # Field 0: Game ID
        retrosheet_row[0] = game_id

        # Field 1: Date (YYYY-MM-DD)
        game_date = game_json.get('gameDate', '')
        try:
            # Convert from "MM/DD/YYYY" to "YYYY-MM-DD"
            parsed_date = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
            retrosheet_row[1] = parsed_date
        except:
            retrosheet_row[1] = '1970-01-01'  # Default to a valid date string if parsing fails

        # Field 2: Game number (0 = no double header)
        retrosheet_row[2] = game_json.get('gameNumber', 0)

        # Field 3: Appearance date (same as date)
        retrosheet_row[3] = retrosheet_row[1]

        # Field 4: Team ID
        retrosheet_row[4] = player_info.get('parentTeamId', 0)

        # Field 5: Player ID
        retrosheet_row[5] = player_info.get('person', {}).get('id', 0)

        # Field 6: Player slot in batting order (use cumulative average if available)
        retrosheet_row[6] = cumulative_stats.get('batting_order', 0)

        # Field 7: Sequence in batting order slot (use cumulative average if available)
        retrosheet_row[7] = cumulative_stats.get('batting_order_sequence', 0)

        # Field 8: Home flag (1 if home, 0 if away)
        retrosheet_row[8] = 1 if is_home else 0

        # Field 9: Opponent ID
        home_team_id = game_json['boxscore']['teams']['home']['team']['id']
        away_team_id = game_json['boxscore']['teams']['away']['team']['id']
        opponent_id = away_team_id if is_home else home_team_id
        retrosheet_row[9] = opponent_id

        # Field 10: Park ID (use cumulative or default if not available)
        retrosheet_row[10] = game_json.get('park', {}).get('id', 0)

        # --- Batting Stats (Fields 11-35) ---
        batting_stats = player_info.get('seasonStats', {}).get('batting', {})
        for stat_key, retrosheet_field in BATTING_STAT_MAP.items():
            if retrosheet_field in self.retrosheet_field_names:
                field_index = self.retrosheet_field_names.index(retrosheet_field)
                value = batting_stats.get(stat_key, 0)
                # Convert to appropriate type
                if isinstance(value, str):
                    try:
                        # Handle average and percentage fields
                        if value.startswith('.'):
                            value = float('0' + value)
                        else:
                            value = float(value)
                    except:
                        value = 0
                elif isinstance(value, (int, float)):
                    pass
                else:
                    value = 0
                retrosheet_row[field_index] = value

        # Handle B_G_DH (Field 33)
        # Assuming 'battingOrder' indicates 'DH' usage
        batting_order_field = player_info.get('battingOrder', '')
        if isinstance(batting_order_field, str) and batting_order_field.upper() == 'DH':
            retrosheet_row[33] = 1
        else:
            retrosheet_row[33] = 0

        # --- Pitching Stats (Fields 36-74) ---
        pitching_stats = player_info.get('seasonStats', {}).get('pitching', {})
        for stat_key, retrosheet_field in PITCHING_STAT_MAP.items():
            if retrosheet_field in self.retrosheet_field_names:
                field_index = self.retrosheet_field_names.index(retrosheet_field)
                value = pitching_stats.get(stat_key, 0)
                # Special handling for inningsPitched -> P_OUT (Field 44)
                if stat_key == 'inningsPitched' and 'P_OUT' in self.retrosheet_field_names:
                    try:
                        innings = float(value)
                        whole_innings = int(innings)
                        fractional = innings - whole_innings
                        outs_recorded = whole_innings * 3 + int(round(fractional * 10 / 3))
                    except:
                        outs_recorded = 0
                    retrosheet_row[field_index] = outs_recorded
                    continue  # Skip default assignment

                # Convert to appropriate type
                if isinstance(value, str):
                    try:
                        # Handle average and percentage fields
                        if value.startswith('.'):
                            value = float('0' + value)
                        else:
                            value = float(value)
                    except:
                        value = 0
                elif isinstance(value, (int, float)):
                    pass
                else:
                    value = 0
                retrosheet_row[field_index] = value

        # Handle P_OUT (Field 44) if not already set
        if 'P_OUT' in PITCHING_STAT_MAP.values():
            field_index = self.retrosheet_field_names.index('P_OUT')
            if retrosheet_row[field_index] == 0:
                innings_pitched = pitching_stats.get('inningsPitched', '0.0')
                try:
                    innings = float(innings_pitched)
                    whole_innings = int(innings)
                    fractional = innings - whole_innings
                    outs_recorded = whole_innings * 3 + int(round(fractional * 10 / 3))
                except:
                    outs_recorded = 0
                retrosheet_row[field_index] = outs_recorded

        # --- Fielding Stats (Fields 75-153) ---
        fielding_stats = player_info.get('seasonStats', {}).get('fielding', {})
        position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
        if position_abbr in FIELDING_STAT_MAP:
            position_map = FIELDING_STAT_MAP[position_abbr]
            for stat_key, retrosheet_field in position_map.items():
                if retrosheet_field in self.retrosheet_field_names:
                    field_index = self.retrosheet_field_names.index(retrosheet_field)
                    value = fielding_stats.get(stat_key, 0)
                    # Convert to appropriate type
                    if isinstance(value, (int, float)):
                        pass
                    else:
                        value = 0
                    retrosheet_row[field_index] = value
        else:
            # If position is not recognized, skip fielding stats
            pass

        return retrosheet_row

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

    def aggregate_team_retrosheet_stats(self, lineup, game_id, game_json, team_type,
                                        cumulative_player_stats, cumulative_team_stats):
        """
        Aggregates Retrosheet statistics for an entire team based on individual player stats using cumulative stats.

        Parameters:
            lineup (list of int): List of player IDs in the team's lineup.
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            team_type (str): 'home' or 'away'.
            cumulative_player_stats (dict): Cumulative player statistics up to previous games.
            cumulative_team_stats (dict): Cumulative team statistics up to previous games.

        Returns:
            dict: A dictionary representing the aggregated Retrosheet statistics for the team.
        """
        aggregated_stats = defaultdict(float)

        for player_id in lineup:
            player_info = self.get_player_info(player_id, game_json, team_type)
            if player_info is None:
                print(f"Player info not found for player_id {player_id} in team {team_type}")
                continue

            player_stats = self.reconstruct_player_stats(
                game_id=game_id,
                game_json=game_json,
                player_info=player_info,
                is_home=(team_type == 'home'),
                cumulative_stats=cumulative_player_stats[player_id]
            )

            # Aggregate Batting Stats
            for stat in BATTING_STAT_MAP.keys():
                aggregated_stats[stat] += player_stats.get(stat, 0)

            # Aggregate Pitching Stats
            for stat in PITCHING_STAT_MAP.keys():
                aggregated_stats[stat] += player_stats.get(stat, 0)

            # Aggregate Fielding Stats
            position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
            if position_abbr in FIELDING_STAT_MAP:
                for stat in FIELDING_STAT_MAP[position_abbr].keys():
                    aggregated_stats[stat] += player_stats.get(stat, 0)
            else:
                print(f"Unrecognized position {position_abbr} for player_id {player_id}")

        # Convert defaultdict to regular dict
        aggregated_stats = dict(aggregated_stats)

        return aggregated_stats

    @staticmethod
    def reconstruct_player_stats(game_id, game_json, player_info, is_home, cumulative_stats):
        """
        Reconstructs a player's statistics based on cumulative stats.

        Parameters:
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            player_info (dict): Player information from the JSON.
            is_home (bool): Indicates if the player is on the home team.
            cumulative_stats (dict): Cumulative statistics for the player up to previous games.

        Returns:
            dict: A dictionary containing the player's statistics.
        """
        player_stats = {}

        # Batting Stats
        season_batting = player_info.get('seasonStats', {}).get('batting', {})
        for key in BATTING_STAT_MAP.keys():
            player_stats[key] = season_batting.get(key, 0)

        # Pitching Stats
        season_pitching = player_info.get('seasonStats', {}).get('pitching', {})
        for key in PITCHING_STAT_MAP.keys():
            if key == 'inningsPitched':
                innings = season_pitching.get(key, 0.0)
                try:
                    whole_innings = int(innings)
                    fractional = innings - whole_innings
                    outs_recorded = whole_innings * 3 + int(round(fractional * 10 / 3))
                    player_stats[key] = outs_recorded / 3  # Convert back to innings pitched
                except:
                    player_stats[key] = 0
            else:
                player_stats[key] = season_pitching.get(key, 0)

        # Fielding Stats
        position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
        if position_abbr in FIELDING_STAT_MAP:
            season_fielding = player_info.get('seasonStats', {}).get('fielding', {})
            for key in FIELDING_STAT_MAP[position_abbr].keys():
                player_stats[key] = season_fielding.get(key, 0)
        else:
            print(
                f"Unrecognized position {position_abbr} for player_id {player_info.get('person', {}).get('id', 'Unknown')}")

        return player_stats

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

        # Initialize cumulative stats
        cumulative_player_stats, cumulative_team_stats = self.initialize_cumulative_stats()

        for game_json in gamelogs_sorted:
            game_id = game_json['scoreboard']['gamePk']
            game_date = game_json['gameDate']

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

            # Aggregate Home Team Retrosheet Stats using cumulative stats
            home_retrosheet_dict = self.aggregate_team_retrosheet_stats(
                lineup=home_lineup,
                game_id=game_id,
                game_json=game_json,
                team_type='home',
                cumulative_player_stats=cumulative_player_stats,
                cumulative_team_stats=cumulative_team_stats
            )

            # Aggregate Away Team Retrosheet Stats using cumulative stats
            away_retrosheet_dict = self.aggregate_team_retrosheet_stats(
                lineup=away_lineup,
                game_id=game_id,
                game_json=game_json,
                team_type='away',
                cumulative_player_stats=cumulative_player_stats,
                cumulative_team_stats=cumulative_team_stats
            )

            # Prefix stat names with 'Home_' and 'Away_'
            home_prefixed = {f"Home_{key}": value for key, value in home_retrosheet_dict.items()}
            away_prefixed = {f"Away_{key}": value for key, value in away_retrosheet_dict.items()}

            # Merge dictionaries correctly using dictionary unpacking (Python 3.5+)
            concatenated_row = {**home_prefixed, **away_prefixed}

            # Include Team Names and Abbreviations, and gamePk and gameDate
            concatenated_row.update({
                'Home_Team_Name': home_team_name,
                'Home_Team_Abbr': home_team_abbr,
                'Away_Team_Name': away_team_name,
                'Away_Team_Abbr': away_team_abbr,
                'Game_PK': game_id,
                'Game_Date': game_date
            })

            retrosheet_rows.append(concatenated_row)

            # Update cumulative stats after processing the game
            self.update_cumulative_stats_after_game(
                game_json=game_json,
                home_lineup=home_lineup,
                away_lineup=away_lineup,
                cumulative_player_stats=cumulative_player_stats,
                cumulative_team_stats=cumulative_team_stats
            )

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
                # Extract final scores from liveData
                live_data = game_json.get('liveData', {})
                boxscore = live_data.get('boxscore', {})
                teams_stats = boxscore.get('teams', {})
                home_stats = teams_stats.get('home', {}).get('teamStats', {}).get('batting', {})
                away_stats = teams_stats.get('away', {}).get('teamStats', {}).get('batting', {})
                home_runs = home_stats.get('runs', 0)
                away_runs = away_stats.get('runs', 0)
                # Determine if home team won
                home_win = 1 if home_runs > away_runs else 0
                game_outcome[game_pk] = home_win
            except Exception as e:
                print(f"Error processing game_pk {game_pk}: {e}")
                game_outcome[game_pk] = 0  # Default to loss if error occurs

        # Map the 'Game_PK' column to 'Home_Win' using the game_outcome dictionary
        retrosheet_df['Home_Win'] = retrosheet_df['Game_PK'].map(game_outcome).fillna(0).astype(int)

        return retrosheet_df

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

        return retrosheet_df

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
                print(info)
                return info
        print('player stats not found')
        return None

    def fetch_game_pks(self):
        """
        Fetches all gamePk's from the start date up to the current date.

        Utilizes the MLB Stats API to retrieve game schedules within the specified date range.

        Returns:
            list: List of gamePk integers.
        """
        url = self.base_schedule_url.format(start_date=self.start_date, end_date=self.end_date)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch gamePk's: {response.status_code}")
            return []

        data = response.json()
        game_pks = []
        if 'dates' in data:
            for date_info in data['dates']:
                for game in date_info.get('games', []):
                    game_pks.append(game['gamePk'])
        return game_pks

    def fetch_gamelog(self, game_pk):
        """
        Fetches game log JSON for a specific gamePk.

        Parameters:
            game_pk (int): Unique identifier for the game.

        Returns:
            dict: Game log JSON data.
        """
        url = self.base_gamelog_url.format(game_pk=game_pk)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch gamelog for game_pk {game_pk}: {response.status_code}")
            return {}
        return response.json()

    def get_unique_game_jsons(self):
        """
        Retrieves a list of unique game JSONs for the date range from start_date to current_date.

        Returns:
            list: List of game JSON dictionaries.
        """
        game_pks = self.fetch_game_pks()
        gamelogs = []
        for game_pk in game_pks:
            gamelog = self.fetch_gamelog(game_pk)
            if gamelog:  # Ensure gamelog is not empty
                gamelogs.append(gamelog)
        return gamelogs

    @staticmethod
    def initialize_cumulative_stats():
        """
        Initializes data structures to hold cumulative player and team statistics.

        Returns:
            dict: Cumulative player stats.
            dict: Cumulative team stats.
        """
        # Using defaultdict to automatically handle missing keys
        cumulative_player_stats = defaultdict(lambda: defaultdict(float))
        cumulative_team_stats = defaultdict(lambda: defaultdict(float))
        return cumulative_player_stats, cumulative_team_stats

    @staticmethod
    def reconstruct_retrosheet_row(game_id, game_json, player_key, player_info, is_home):
        """
        Reconstructs a Retrosheet statistics row for a single player in a game using seasonStats.

        Parameters:
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            player_key (str): Player identifier (e.g., "ID451594").
            player_info (dict): Player information from the JSON.
            is_home (bool): Flag indicating if the player is on the home team.

        Returns:
            list: A list representing the Retrosheet statistics row with numeric values.
        """
        # Initialize a list with 154 default numeric values (zeros)
        retrosheet_row = [0] * 154

        # Field 0: Game ID
        retrosheet_row[0] = game_id

        # Field 1: Date (YYYY-MM-DD)
        game_date = game_json.get('gameDate', '')
        try:
            # Convert from "MM/DD/YYYY" to "YYYY-MM-DD"
            parsed_date = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
            retrosheet_row[1] = parsed_date
        except:
            retrosheet_row[1] = 0  # Default to 0 if date is invalid

        # Field 2: Game number (0 = no double header)
        retrosheet_row[2] = game_json.get('gameNumber', 0)

        # Field 3: Appearance date (same as date)
        retrosheet_row[3] = retrosheet_row[1]

        # Field 4: Team ID
        retrosheet_row[4] = player_info.get('parentTeamId', 0)

        # Field 5: Player ID
        retrosheet_row[5] = player_info.get('person', {}).get('id', 0)

        # Field 6: Player slot in batting order
        batting_order = player_info.get('battingOrder', 0)
        try:
            retrosheet_row[6] = int(batting_order)
        except:
            retrosheet_row[6] = 0  # Default to 0 if not an integer

        # Field 7: Sequence in batting order slot
        # Assuming sequence is same as battingOrder; adjust if different
        batting_order_sequence = player_info.get('battingOrderSequence', 0)
        try:
            retrosheet_row[7] = int(batting_order_sequence)
        except:
            retrosheet_row[7] = 0  # Default to 0 if not an integer

        # Field 8: Home flag (1 if home, 0 if away)
        retrosheet_row[8] = 1 if is_home else 0

        # Field 9: Opponent ID
        home_team_id = game_json['scoreboard']['teams']['home']['team']['id']
        away_team_id = game_json['scoreboard']['teams']['away']['team']['id']
        opponent_id = away_team_id if is_home else home_team_id
        retrosheet_row[9] = opponent_id

        # Field 10: Park ID
        retrosheet_row[10] = game_json.get('park', {}).get('id', 0)

        # --- Batting Stats (Fields 11-35) ---
        batting_stats = player_info.get('stats', {}).get('batting', {})
        if batting_stats:
            # Populate batting fields
            retrosheet_row[11] = batting_stats.get('gamesPlayed', 0)
            retrosheet_row[12] = batting_stats.get('plateAppearances', 0)
            retrosheet_row[13] = batting_stats.get('atBats', 0)
            retrosheet_row[14] = batting_stats.get('runs', 0)
            retrosheet_row[15] = batting_stats.get('hits', 0)
            retrosheet_row[16] = batting_stats.get('totalBases', 0)
            retrosheet_row[17] = batting_stats.get('doubles', 0)
            retrosheet_row[18] = batting_stats.get('triples', 0)
            retrosheet_row[19] = batting_stats.get('homeRuns', 0)
            retrosheet_row[20] = batting_stats.get('grandSlams', 0)
            retrosheet_row[21] = batting_stats.get('rbi', 0)
            retrosheet_row[22] = batting_stats.get('gameWinningRbi', 0)
            retrosheet_row[23] = batting_stats.get('baseOnBalls', 0)
            retrosheet_row[24] = batting_stats.get('intentionalWalks', 0)
            retrosheet_row[25] = batting_stats.get('strikeOuts', 0)
            retrosheet_row[26] = batting_stats.get('groundIntoDoublePlay', 0)
            retrosheet_row[27] = batting_stats.get('hitByPitch', 0)
            retrosheet_row[28] = batting_stats.get('sacBunts', 0)
            retrosheet_row[29] = batting_stats.get('sacFlies', 0)
            retrosheet_row[30] = batting_stats.get('stolenBases', 0)
            retrosheet_row[31] = batting_stats.get('caughtStealing', 0)
            retrosheet_row[32] = batting_stats.get('catchersInterference', 0)
            # Field 33: B_G_DH - Games as DH
            batting_order_field = player_info.get('battingOrder', '')
            if isinstance(batting_order_field, str) and batting_order_field.upper() == 'DH':
                retrosheet_row[33] = 1
            else:
                retrosheet_row[33] = 0
            # Field 34: B_G_PH - Games as PH (Pinch Hitter)
            retrosheet_row[34] = batting_stats.get('gamesAsPH', 0)
            # Field 35: B_G_PR - Games as PR (Pinch Runner)
            retrosheet_row[35] = batting_stats.get('gamesAsPR', 0)
        else:
            # Assign default values if no batting stats
            for i in range(11, 36):
                retrosheet_row[i] = 0

        # --- Pitching Stats (Fields 36-74) ---
        pitching_stats = player_info.get('stats', {}).get('pitching', {})
        if pitching_stats:
            # Populate pitching fields
            retrosheet_row[36] = pitching_stats.get('gamesPlayed', 0)
            retrosheet_row[37] = pitching_stats.get('gamesStarted', 0)
            retrosheet_row[38] = pitching_stats.get('completeGames', 0)
            retrosheet_row[39] = pitching_stats.get('shutouts', 0)
            retrosheet_row[40] = pitching_stats.get('gamesFinished', 0)
            retrosheet_row[41] = pitching_stats.get('wins', 0)
            retrosheet_row[42] = pitching_stats.get('losses', 0)
            retrosheet_row[43] = pitching_stats.get('saves', 0)
            # Field 44: P_OUT - Outs Recorded (innings pitched * 3)
            innings_pitched = pitching_stats.get('inningsPitched', '0.0')
            try:
                innings = float(innings_pitched)
                # Handle partial innings (e.g., 4.2 means 4 innings and 2 outs)
                whole_innings = int(innings)
                fractional = innings - whole_innings
                outs_recorded = whole_innings * 3 + int(round(fractional * 10 / 3))
            except:
                outs_recorded = 0
            retrosheet_row[44] = outs_recorded
            # Continue populating pitching fields
            retrosheet_row[45] = pitching_stats.get('battersFaced', 0)
            retrosheet_row[46] = pitching_stats.get('atBats', 0)
            retrosheet_row[47] = pitching_stats.get('runs', 0)
            retrosheet_row[48] = pitching_stats.get('earnedRuns', 0)
            retrosheet_row[49] = pitching_stats.get('hits', 0)
            retrosheet_row[50] = pitching_stats.get('totalBases', 0)
            retrosheet_row[51] = pitching_stats.get('doubles', 0)
            retrosheet_row[52] = pitching_stats.get('triples', 0)
            retrosheet_row[53] = pitching_stats.get('homeRuns', 0)
            retrosheet_row[54] = pitching_stats.get('grandSlamsAllowed', 0)
            retrosheet_row[55] = pitching_stats.get('walks', 0)
            retrosheet_row[56] = pitching_stats.get('intentionalWalks', 0)
            retrosheet_row[57] = pitching_stats.get('strikeOuts', 0)
            retrosheet_row[58] = pitching_stats.get('groundIntoDoublePlay', 0)
            retrosheet_row[59] = pitching_stats.get('hitBatsmen', 0)
            retrosheet_row[60] = pitching_stats.get('sacHitsAgainst', 0)
            retrosheet_row[61] = pitching_stats.get('sacFliesAgainst', 0)
            retrosheet_row[62] = pitching_stats.get('reachedOnInterference', 0)
            retrosheet_row[63] = pitching_stats.get('wildPitches', 0)
            retrosheet_row[64] = pitching_stats.get('balks', 0)
            retrosheet_row[65] = pitching_stats.get('inheritedRunners', 0)
            retrosheet_row[66] = pitching_stats.get('inheritedRunnersScored', 0)
            retrosheet_row[67] = pitching_stats.get('groundOuts', 0)
            retrosheet_row[68] = pitching_stats.get('airOuts', 0)
            retrosheet_row[69] = pitching_stats.get('numberOfPitches', 0)
            retrosheet_row[70] = pitching_stats.get('strikes', 0)
            retrosheet_row[71] = pitching_stats.get('gamesAtP', 0)
            retrosheet_row[72] = pitching_stats.get('gamesStartedAtP', 0)
            retrosheet_row[73] = outs_recorded
            retrosheet_row[74] = pitching_stats.get('totalChancesAtP', 0)
        else:
            # Assign default values if no pitching stats
            for i in range(36, 75):
                retrosheet_row[i] = 0

        # --- Fielding Stats (Fields 75-153) ---
        fielding_stats = player_info.get('stats', {}).get('fielding', {})
        position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
        if fielding_stats:
            # Mapping fielding stats based on position
            position_field_indices = {
                'P': {
                    "F_P_G": 75, "F_P_GS": 76, "F_P_OUT": 77,
                    "F_P_TC": 78, "F_P_PO": 79, "F_P_A": 80,
                    "F_P_E": 81, "F_P_DP": 82, "F_P_TP": 83
                },
                'C': {
                    "F_C_G": 84, "F_C_GS": 85, "F_C_OUT": 86,
                    "F_C_TC": 87, "F_C_PO": 88, "F_C_A": 89,
                    "F_C_E": 90, "F_C_DP": 91, "F_C_TP": 92,
                    "F_C_PB": 93, "F_C_IX": 94
                },
                '1B': {
                    "F_1B_G": 95, "F_1B_GS": 96, "F_1B_OUT": 97,
                    "F_1B_TC": 98, "F_1B_PO": 99, "F_1B_A": 100,
                    "F_1B_E": 101, "F_1B_DP": 102, "F_1B_TP": 103
                },
                '2B': {
                    "F_2B_G": 104, "F_2B_GS": 105, "F_2B_OUT": 106,
                    "F_2B_TC": 107, "F_2B_PO": 108, "F_2B_A": 109,
                    "F_2B_E": 110, "F_2B_DP": 111, "F_2B_TP": 112
                },
                '3B': {
                    "F_3B_G": 113, "F_3B_GS": 114, "F_3B_OUT": 115,
                    "F_3B_TC": 116, "F_3B_PO": 117, "F_3B_A": 118,
                    "F_3B_E": 119, "F_3B_DP": 120, "F_3B_TP": 121
                },
                'SS': {
                    "F_SS_G": 122, "F_SS_GS": 123, "F_SS_OUT": 124,
                    "F_SS_TC": 125, "F_SS_PO": 126, "F_SS_A": 127,
                    "F_SS_E": 128, "F_SS_DP": 129, "F_SS_TP": 130
                },
                'LF': {
                    "F_LF_G": 131, "F_LF_GS": 132, "F_LF_OUT": 133,
                    "F_LF_TC": 134, "F_LF_PO": 135, "F_LF_A": 136,
                    "F_LF_E": 137, "F_LF_DP": 138, "F_LF_TP": 139
                },
                'CF': {
                    "F_CF_G": 140, "F_CF_GS": 141, "F_CF_OUT": 142,
                    "F_CF_TC": 143, "F_CF_PO": 144, "F_CF_A": 145,
                    "F_CF_E": 146, "F_CF_DP": 147, "F_CF_TP": 148
                },
                'RF': {
                    "F_RF_G": 149, "F_RF_GS": 150, "F_RF_OUT": 151,
                    "F_RF_TC": 152, "F_RF_PO": 153, "F_RF_A": 154,
                    "F_RF_E": 155, "F_RF_DP": 156, "F_RF_TP": 157
                }
            }

            # Only map if the position abbreviation exists in the dictionary
            if position_abbr in position_field_indices:
                for stat_key, field_index in position_field_indices[position_abbr].items():
                    # Map each field
                    # Assuming fielding_stats keys are in lower case and correspond to stat_key
                    value = fielding_stats.get(stat_key.lower(), 0)
                    # Ensure the field index is within 0-153
                    if 0 <= field_index < 154:
                        retrosheet_row[field_index] = value
            else:
                # If position is not recognized, skip fielding stats
                pass
        else:
            # Assign default values if no fielding stats
            for i in range(75, 154):
                retrosheet_row[i] = 0

        return retrosheet_row

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
