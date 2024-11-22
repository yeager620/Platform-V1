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
    # TODO: Fix normalization logic for player level data and full game vectors
    # TODO: Ensure lineup is constructed from starting lineup, not final lineup, to prevent lookahead bias
    # TODO: Create theoretical projection calculator for game participation factor, specifically for pitchers
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

    def reconstruct_player_stats(self, game_id, game_json, player_info, is_home, is_pitcher=False, live=False):
        """
        Reconstructs a statistics dictionary for a single player based on season stats.

        Parameters:
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            player_info (dict): Player information from the JSON.
            is_home (bool): Flag indicating if the player is on the home team.
            is_pitcher (bool): Indicated player is a pitcher
            live (bool): Indicates live prediction

        Returns:
            dict: A dictionary representing the Retrosheet statistics with keys as stat names.
        """
        # print(f"Reconstructing stats for player: {player_info.get('person', {}).get('id', 'Unknown')}")
        # print(f"Season stats: {player_info.get('seasonStats', {})}")
        # print(f"This game stats: {player_info.get('stats', {})}")
        # Initialize a dictionary to hold all stats
        stats_dict = {}

        position_abb = player_info.get('position', {}).get('abbreviation', 'None')

        # Basic Fields
        stats_dict['game_id'] = game_id
        game_date = game_json.get('gameDate', '')
        try:
            # Convert from "MM/DD/YYYY" to "YYYY-MM-DD"
            parsed_date = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
        except:
            parsed_date = '1970-01-01'  # Default date if parsing fails
        stats_dict['date'] = parsed_date
        stats_dict['game_number'] = game_json.get('gameNumber', 0)
        stats_dict['appearance_date'] = parsed_date
        stats_dict['team_id'] = player_info.get('parentTeamId', 0)
        stats_dict['player_id'] = player_info.get('person', {}).get('id', 0)
        stats_dict['batting_order'] = 0  # Placeholder
        stats_dict['batting_order_sequence'] = 0  # Placeholder
        stats_dict['home_flag'] = 1 if is_home else 0

        # Opponent ID
        home_team_id = game_json['boxscore']['teams']['home']['team']['id']
        away_team_id = game_json['boxscore']['teams']['away']['team']['id']
        opponent_id = away_team_id if is_home else home_team_id
        stats_dict['opponent_id'] = opponent_id

        # Park ID
        stats_dict['park_id'] = game_json.get('park', {}).get('id', 0)

        # Helper function to adjust stats
        def get_adjusted_stats(stat_map, season_stats, game_stats):
            """
            Adjusts cumulative stats by subtracting game stats from season stats.

            Parameters:
                stat_map (dict): Mapping of stat keys to their field names.
                season_stats (dict): Season-level statistics for the player.
                game_stats (dict): Game-level statistics for the player.

            Returns:
                dict: Adjusted statistics.
            """
            adjusted_stats = {}
            for stat_key in stat_map.keys():
                # Retrieve the stats, defaulting to 0 if not found
                season_stat = season_stats.get(stat_key, 0)
                game_stat = game_stats.get(stat_key, 0)

                # Convert to numeric values, defaulting to 0 for invalid entries
                try:
                    season_stat = float(season_stat)
                except (ValueError, TypeError):
                    season_stat = 0

                try:
                    game_stat = float(game_stat)
                except (ValueError, TypeError):
                    game_stat = 0

                # Perform subtraction
                adjusted_stats[stat_key] = season_stat - game_stat

            return adjusted_stats

        # Batting Stats
        season_batting = player_info.get('seasonStats', {}).get('batting', {})
        game_batting = player_info.get('stats', {}).get('batting', {})
        adjusted_batting_stats = get_adjusted_stats(BATTING_STAT_MAP, season_batting, game_batting)

        # Populate Batting Stats
        for stat_key, retrosheet_field in BATTING_STAT_MAP.items():
            value = adjusted_batting_stats.get(stat_key, 0)
            stats_dict[f"B_{stat_key}"] = value

        # Pitching Stats
        season_pitching = player_info.get('seasonStats', {}).get('pitching', {})
        game_pitching = player_info.get('stats', {}).get('pitching', {})
        adjusted_pitching_stats = get_adjusted_stats(PITCHING_STAT_MAP, season_pitching, game_pitching)

        # Populate Pitching Stats
        for stat_key, retrosheet_field in PITCHING_STAT_MAP.items():
            value = adjusted_pitching_stats.get(stat_key, 0)
            # Special handling for 'inningsPitched' -> 'P_OUT'
            stats_dict[f"P_{stat_key}"] = value

        # Fielding Stats
        position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
        season_fielding = player_info.get('seasonStats', {}).get('fielding', {})
        game_fielding = player_info.get('stats', {}).get('fielding', {})
        fielding_map = FIELDING_STAT_MAP.get(position_abbr, {})
        adjusted_fielding_stats = get_adjusted_stats(fielding_map, season_fielding, game_fielding)

        # Populate Fielding Stats
        for stat_key, retrosheet_field in fielding_map.items():
            value = adjusted_fielding_stats.get(stat_key, 0)
            stats_dict[f"F_{stat_key}"] = value

        # Normalize stats by games played to get per-game averages
        if position_abb == 'P':
            try:
                games_played = int(player_info.get('seasonStats', {}).get('pitching', {}).get('gamesPlayed', 1)) - 1
            except (ValueError, TypeError):
                games_played = 1  # Fallback to avoid division by zero or invalid value

            try:
                season_innings_pitched = float(
                    player_info.get('seasonStats', {}).get('pitching', {}).get('inningsPitched', 0))
                game_innings_pitched = float(player_info.get('stats', {}).get('pitching', {}).get('inningsPitched', 0))
                innings_pitched = season_innings_pitched - game_innings_pitched
            except (ValueError, TypeError):
                innings_pitched = 0  # Fallback for invalid innings values
        else:
            try:
                games_played = int(player_info.get('seasonStats', {}).get('batting', {}).get('gamesPlayed', 1)) - 1
            except (ValueError, TypeError):
                games_played = 1  # Fallback to avoid division by zero or invalid value

            innings_pitched = 0  # Not applicable for non-pitchers

        if games_played <= 0:
            print(f"Warning: gamesPlayed is {games_played} for player {stats_dict['player_id']}. Defaulting to 1.")

            games_played = 1  # Avoid division by zero

        # Normalize Batting Stats
        for stat_key in BATTING_STAT_MAP.keys():
            stats_dict[f"B_{stat_key}"] /= games_played

        # Normalize Pitching Stats
        for stat_key in PITCHING_STAT_MAP.keys():
            stats_dict[f"P_{stat_key}"] /= games_played

        # Normalize Fielding Stats
        for stat_key in fielding_map.keys():
            stats_dict[f"F_{stat_key}"] /= games_played

        print(f"Reconstructed stats for player {stats_dict['player_id']}: {stats_dict}")

        return stats_dict

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

    def aggregate_team_retrosheet_stats(self, starting_batters, starting_pitcher, game_id, game_json, team_type,
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
        num_batters = 0
        num_pitchers = 0

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
                for stat in BATTING_STAT_MAP.keys():
                    aggregated_stats[f"B_{stat}"] += player_stats.get(f"B_{stat}", 0)

                # Aggregate Fielding Stats
                if position_abbr in FIELDING_STAT_MAP:
                    for stat in FIELDING_STAT_MAP[position_abbr].keys():
                        aggregated_stats[f"F_{stat}"] += player_stats.get(f"F_{stat}", 0)
                else:
                    if position_abbr != 'DH':  # DH doesn't have fielding stats
                        print(f"Unrecognized position {position_abbr} for player_id {player_id}")

            # Aggregate Pitching Stats
            if is_pitcher:
                num_pitchers += 1
                for stat in PITCHING_STAT_MAP.keys():
                    aggregated_stats[f"P_{stat}"] += player_stats.get(f"P_{stat}", 0)

        # Normalize Batting Stats
        if num_batters > 0:
            for stat in BATTING_STAT_MAP.keys():
                aggregated_stats[f"B_{stat}"] /= num_batters

        # Normalize Pitching Stats
        if num_pitchers > 0:
            for stat in PITCHING_STAT_MAP.keys():
                aggregated_stats[f"P_{stat}"] /= num_pitchers

        # Normalize Fielding Stats
        num_fielders = num_batters  # Fielders have the same number as batters (excluding DH)
        if num_fielders > 0:
            for position in FIELDING_STAT_MAP.keys():
                for stat in FIELDING_STAT_MAP[position].keys():
                    if f"F_{stat}" in aggregated_stats:
                        aggregated_stats[f"F_{stat}"] /= num_fielders

        # Convert defaultdict to regular dict
        aggregated_stats = dict(aggregated_stats)

        return aggregated_stats

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
    def has_cumulative_stats(player_cumulative):
        """
        Determines if a player has any cumulative statistics.

        Parameters:
            player_cumulative (defaultdict): Cumulative statistics for the player.

        Returns:
            bool: True if the player has at least one non-zero statistic, False otherwise.
        """
        return any(value != 0 for value in player_cumulative.values())

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

            home_batters = game_json['boxscore']['teams']['home']['batters']
            away_batters = game_json['boxscore']['teams']['away']['batters']

            def filter_batters(batter_ids, game_json, team):
                return [
                    batter_id
                    for batter_id in batter_ids
                    if game_json['boxscore']['teams'][team]['players'][f"ID{batter_id}"]['stats']['fielding'].get(
                        'gamesStarted', 0) >= 1
                ]

            # Filter home and away batters
            starting_home_batters = filter_batters(
                game_json['boxscore']['teams']['home']['batters'],
                game_json,
                'home'
            )

            starting_away_batters = filter_batters(
                game_json['boxscore']['teams']['away']['batters'],
                game_json,
                'away'
            )

            home_batting_order = game_json['boxscore']['teams']['home']['battingOrder']
            away_batting_order = game_json['boxscore']['teams']['away']['battingOrder']

            home_bench = game_json['boxscore']['teams']['home']['bench']
            away_bench = game_json['boxscore']['teams']['away']['bench']

            # Get home and away probable pitchers: Use as starting pitcher
            # TODO: Implement edge case handling for no available probable pitchers
            home_probable_pitcher = game_json['scoreboard']['probablePitchers']['home'].get('id', None)
            away_probable_pitcher = game_json['scoreboard']['probablePitchers']['away'].get('id', None)

            home_pitchers = game_json['boxscore']['teams']['home']['pitchers']
            away_pitchers = game_json['boxscore']['teams']['away']['pitchers']

            home_bullpen_pitchers = game_json['boxscore']['teams']['home']['bullpen']
            away_bullpen_pitchers = game_json['boxscore']['teams']['away']['bullpen']

            # Aggregate Home Team Retrosheet Stats using cumulative stats
            home_retrosheet_dict = self.aggregate_team_retrosheet_stats(
                starting_batters=starting_home_batters,
                starting_pitcher=home_probable_pitcher,
                game_id=game_id,
                game_json=game_json,
                team_type='home',
                cumulative_player_stats=cumulative_player_stats,
                cumulative_team_stats=cumulative_team_stats
            )

            # Aggregate Away Team Retrosheet Stats using cumulative stats
            away_retrosheet_dict = self.aggregate_team_retrosheet_stats(
                starting_batters=starting_away_batters,
                starting_pitcher=away_probable_pitcher,
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
            # Not applicable for backtesting
            '''
            self.update_cumulative_stats_after_game(
                game_json=game_json,
                home_lineup=home_lineup,
                away_lineup=away_lineup,
                cumulative_player_stats=cumulative_player_stats,
                cumulative_team_stats=cumulative_team_stats
            )
            '''
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
                return info
        print('player stats not found')
        return None

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
