import json
import datetime
from datetime import datetime
import requests
import pandas as pd


class SavantRetrosheetConverter:
    def __init__(self):
        self.base_schedule_url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={date}&endDate={date}&fields=dates,date,games,gamePk"
        self.base_gamelog_url = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"

    @staticmethod
    def get_current_date():
        return datetime.now().strftime("%m/%d/%Y")

    def fetch_game_pks(self, date):
        """
        Fetches game Pks for a specific date.

        Parameters:
            date (str): Date in "MM/DD/YYYY" format.

        Returns:
            list: List of gamePk integers.
        """
        url = self.base_schedule_url.format(date=date)
        response = requests.get(url)
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
        return response.json()

    def get_unique_game_jsons(self):
        """
        Retrieves a list of unique game JSONs for the current date.

        Returns:
            list: List of game JSON dictionaries.
        """
        current_date = self.get_current_date()
        game_pks = self.fetch_game_pks(current_date)
        gamelogs = []
        for game_pk in game_pks:
            gamelog = self.fetch_gamelog(game_pk)
            gamelogs.append(gamelog)
        return gamelogs

    def reconstruct_retrosheet_row(self, game_id, game_json, player_key, player_info, is_home):
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
            retrosheet_row[1] = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
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
        # Assuming park ID is available in `game_json['park']['id']`; else set to 0
        retrosheet_row[10] = game_json.get('park', {}).get('id', 0)

        # --- Batting Stats (Fields 11-35) ---
        batting_stats = player_info.get('stats', {}).get('batting', {})
        if batting_stats:
            # Field 11: B_G - Games Played
            retrosheet_row[11] = batting_stats.get('gamesPlayed', 0)
            # Field 12: B_PA - Plate Appearances
            retrosheet_row[12] = batting_stats.get('plateAppearances', 0)
            # Field 13: B_AB - At Bats
            retrosheet_row[13] = batting_stats.get('atBats', 0)
            # Field 14: B_R - Runs
            retrosheet_row[14] = batting_stats.get('runs', 0)
            # Field 15: B_H - Hits
            retrosheet_row[15] = batting_stats.get('hits', 0)
            # Field 16: B_TB - Total Bases
            retrosheet_row[16] = batting_stats.get('totalBases', 0)
            # Field 17: B_2B - Doubles
            retrosheet_row[17] = batting_stats.get('doubles', 0)
            # Field 18: B_3B - Triples
            retrosheet_row[18] = batting_stats.get('triples', 0)
            # Field 19: B_HR - Home Runs
            retrosheet_row[19] = batting_stats.get('homeRuns', 0)
            # Field 20: B_HR4 - Grand Slams
            retrosheet_row[20] = batting_stats.get('grandSlams', 0)  # Update based on actual data
            # Field 21: B_RBI - Runs Batted In
            retrosheet_row[21] = batting_stats.get('rbi', 0)
            # Field 22: B_GW - Game Winning RBI
            retrosheet_row[22] = batting_stats.get('gameWinningRbi', 0)  # Update based on actual data
            # Field 23: B_BB - Walks
            retrosheet_row[23] = batting_stats.get('baseOnBalls', 0)
            # Field 24: B_IBB - Intentional Walks
            retrosheet_row[24] = batting_stats.get('intentionalWalks', 0)
            # Field 25: B_SO - Strikeouts
            retrosheet_row[25] = batting_stats.get('strikeOuts', 0)
            # Field 26: B_GDP - Grounded Into Double Play
            retrosheet_row[26] = batting_stats.get('groundIntoDoublePlay', 0)
            # Field 27: B_HP - Hit By Pitch
            retrosheet_row[27] = batting_stats.get('hitByPitch', 0)
            # Field 28: B_SH - Sacrifice Hits
            retrosheet_row[28] = batting_stats.get('sacBunts', 0)
            # Field 29: B_SF - Sacrifice Flies
            retrosheet_row[29] = batting_stats.get('sacFlies', 0)
            # Field 30: B_SB - Stolen Bases
            retrosheet_row[30] = batting_stats.get('stolenBases', 0)
            # Field 31: B_CS - Caught Stealing
            retrosheet_row[31] = batting_stats.get('caughtStealing', 0)
            # Field 32: B_XI - Reached on Interference
            retrosheet_row[32] = batting_stats.get('catchersInterference', 0)
            # Field 33: B_G_DH - Games as DH
            # Assuming 'battingOrder' or another field indicates DH; adjust as needed
            batting_order = player_info.get('battingOrder', '')
            if isinstance(batting_order, str) and batting_order.upper() == 'DH':
                retrosheet_row[33] = 1
            else:
                retrosheet_row[33] = 0
            # Field 34: B_G_PH - Games as PH (Pinch Hitter)
            # Adjust based on actual data
            retrosheet_row[34] = batting_stats.get('gamesAsPH', 0)  # Update based on actual data
            # Field 35: B_G_PR - Games as PR (Pinch Runner)
            retrosheet_row[35] = batting_stats.get('gamesAsPR', 0)  # Update based on actual data
        else:
            # Assign default values if no batting stats
            for i in range(11, 36):
                retrosheet_row[i] = 0

        # --- Pitching Stats (Fields 36-74) ---
        pitching_stats = player_info.get('stats', {}).get('pitching', {})
        if pitching_stats:
            # Field 36: P_G - Games Pitched
            retrosheet_row[36] = pitching_stats.get('gamesPlayed', 0)
            # Field 37: P_GS - Games Started
            retrosheet_row[37] = pitching_stats.get('gamesStarted', 0)
            # Field 38: P_CG - Complete Games
            retrosheet_row[38] = pitching_stats.get('completeGames', 0)
            # Field 39: P_SHO - Shutouts
            retrosheet_row[39] = pitching_stats.get('shutouts', 0)
            # Field 40: P_GF - Games Finished
            retrosheet_row[40] = pitching_stats.get('gamesFinished', 0)
            # Field 41: P_W - Wins
            retrosheet_row[41] = pitching_stats.get('wins', 0)
            # Field 42: P_L - Losses
            retrosheet_row[42] = pitching_stats.get('losses', 0)
            # Field 43: P_SV - Saves
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
            # Field 45: P_TBF - Batters Faced
            retrosheet_row[45] = pitching_stats.get('battersFaced', 0)
            # Field 46: P_AB - At Bats (against)
            retrosheet_row[46] = pitching_stats.get('atBats', 0)
            # Field 47: P_R - Runs Allowed
            retrosheet_row[47] = pitching_stats.get('runs', 0)
            # Field 48: P_ER - Earned Runs Allowed
            retrosheet_row[48] = pitching_stats.get('earnedRuns', 0)
            # Field 49: P_H - Hits Allowed
            retrosheet_row[49] = pitching_stats.get('hits', 0)
            # Field 50: P_TB - Total Bases Allowed
            retrosheet_row[50] = pitching_stats.get('totalBases', 0)
            # Field 51: P_2B - Doubles Allowed
            retrosheet_row[51] = pitching_stats.get('doubles', 0)
            # Field 52: P_3B - Triples Allowed
            retrosheet_row[52] = pitching_stats.get('triples', 0)
            # Field 53: P_HR - Home Runs Allowed
            retrosheet_row[53] = pitching_stats.get('homeRuns', 0)
            # Field 54: P_HR4 - Grand Slams Allowed
            retrosheet_row[54] = pitching_stats.get('grandSlamsAllowed', 0)  # Update based on actual data
            # Field 55: P_BB - Walks Allowed
            retrosheet_row[55] = pitching_stats.get('walks', 0)
            # Field 56: P_IBB - Intentional Walks Allowed
            retrosheet_row[56] = pitching_stats.get('intentionalWalks', 0)
            # Field 57: P_SO - Strikeouts
            retrosheet_row[57] = pitching_stats.get('strikeOuts', 0)
            # Field 58: P_GDP - Grounded Into Double Play
            retrosheet_row[58] = pitching_stats.get('groundIntoDoublePlay', 0)
            # Field 59: P_HP - Hit Batsmen
            retrosheet_row[59] = pitching_stats.get('hitBatsmen', 0)
            # Field 60: P_SH - Sacrifice Hits Against
            retrosheet_row[60] = pitching_stats.get('sacHitsAgainst', 0)  # Update based on actual data
            # Field 61: P_SF - Sacrifice Flies Against
            retrosheet_row[61] = pitching_stats.get('sacFliesAgainst', 0)  # Update based on actual data
            # Field 62: P_XI - Reached on Interference
            retrosheet_row[62] = pitching_stats.get('reachedOnInterference', 0)  # Update based on actual data
            # Field 63: P_WP - Wild Pitches
            retrosheet_row[63] = pitching_stats.get('wildPitches', 0)
            # Field 64: P_BK - Balks
            retrosheet_row[64] = pitching_stats.get('balks', 0)
            # Field 65: P_IR - Inherited Runners
            retrosheet_row[65] = pitching_stats.get('inheritedRunners', 0)
            # Field 66: P_IRS - Inherited Runners Scored
            retrosheet_row[66] = pitching_stats.get('inheritedRunnersScored', 0)
            # Field 67: P_GO - Ground Outs
            retrosheet_row[67] = pitching_stats.get('groundOuts', 0)
            # Field 68: P_AO - Air Outs
            retrosheet_row[68] = pitching_stats.get('airOuts', 0)
            # Field 69: P_PITCH - Pitches
            retrosheet_row[69] = pitching_stats.get('numberOfPitches', 0)
            # Field 70: P_STRIKE - Strikes
            retrosheet_row[70] = pitching_stats.get('strikes', 0)
            # Field 71: F_P_G - Games at P (Pitcher)
            retrosheet_row[71] = pitching_stats.get('gamesAtP', 0)  # Update based on actual data
            # Field 72: F_P_GS - Games Started at P
            retrosheet_row[72] = pitching_stats.get('gamesStartedAtP', 0)  # Update based on actual data
            # Field 73: F_P_OUT - Outs Recorded at P (innings pitched * 3)
            retrosheet_row[73] = outs_recorded
            # Field 74: F_P_TC - Total Chances at P
            retrosheet_row[74] = pitching_stats.get('totalChancesAtP', 0)  # Update based on actual data
        else:
            # Assign default values if no pitching stats
            for i in range(36, 75):
                retrosheet_row[i] = 0

        # --- Fielding Stats (Fields 75-153) ---
        fielding_stats = player_info.get('stats', {}).get('fielding', {})
        position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
        if fielding_stats:
            # Mapping fielding stats based on position
            # Define a helper dictionary to map position abbreviations to field indices
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

    def aggregate_team_retrosheet_stats(self, lineup, game_id, game_json, team_type):
        """
        Aggregates Retrosheet statistics for an entire team based on individual player stats using seasonStats.

        Parameters:
            lineup (list of int): List of player IDs in the team's lineup.
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            team_type (str): 'home' or 'away'.

        Returns:
            list: A list representing the aggregated Retrosheet statistics row for the team.
        """

        # Initialize accumulators for batting, pitching, and fielding
        batting_sum = [0] * 25  # Fields 11-35 (25 fields)
        batting_weight = 0

        pitching_sum = [0] * 39  # Fields 36-74 (39 fields)
        pitching_weight = 0

        fielding_sum = [0] * 79  # Fields 75-153 (79 fields)
        fielding_weight = 0

        # Initialize the aggregated team Retrosheet row with numeric zeros
        team_retrosheet_row = [0] * 154

        # Assign game info fields (0-10)
        team_retrosheet_row[0] = game_id

        # Field 1: Date (YYYY-MM-DD)
        game_date = game_json.get('gameDate', '')
        try:
            team_retrosheet_row[1] = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
        except:
            team_retrosheet_row[1] = 0  # Default to 0 if date is invalid

        # Field 2: Game number (0 = no double header)
        team_retrosheet_row[2] = game_json.get('gameNumber', 0)

        # Field 3: Appearance date (same as date)
        team_retrosheet_row[3] = team_retrosheet_row[1]

        # Field 4: Team ID
        team_id = game_json['scoreboard']['teams'][team_type]['team']['id']
        team_retrosheet_row[4] = team_id

        # Field 5: Player ID (Using team_id as a placeholder)
        team_retrosheet_row[5] = team_id  # Alternatively, use a special identifier

        # Fields 6-7: Batting Order Slot and Sequence (set to 0 for team-level)
        team_retrosheet_row[6] = 0
        team_retrosheet_row[7] = 0

        # Field 8: Home flag (1 if home, 0 if away)
        team_retrosheet_row[8] = 1 if team_type == 'home' else 0

        # Field 9: Opponent ID
        opponent_team_id = game_json['scoreboard']['teams']['away']['team']['id'] if team_type == 'home' else \
        game_json['scoreboard']['teams']['home']['team']['id']
        team_retrosheet_row[9] = opponent_team_id

        # Field 10: Park ID
        team_retrosheet_row[10] = game_json.get('park', {}).get('id', 0)

        # Iterate through each player in the lineup
        for player_id in lineup:
            # Retrieve player_info
            player_info = self.get_player_info(player_id, game_json, team_type)
            if player_info is None:
                continue  # Skip if player not found

            # Replace 'stats' with 'seasonStats' to use season statistics
            if 'seasonStats' in player_info:
                player_info = player_info.copy()  # To avoid mutating the original data
                player_info['stats'] = player_info['seasonStats']
            else:
                # If 'seasonStats' not available, use existing 'stats'
                pass

            # Determine if the player is on the home team
            is_home = 1 if team_type == 'home' else 0

            # Reconstruct the player's Retrosheet row
            player_retrosheet_row = self.reconstruct_retrosheet_row(game_id, game_json, None, player_info, is_home)

            # --- Participation Metrics ---

            # Batting Participation: atBats (Field 13)
            at_bats = player_retrosheet_row[13]
            if at_bats > 0:
                batting_weight += at_bats
                for i in range(11, 36):
                    batting_sum[i - 11] += player_retrosheet_row[i] * at_bats

            # Pitching Participation: inningsPitched = outsRecorded / 3 (Field 44)
            outs_recorded = player_retrosheet_row[44]
            innings_pitched = outs_recorded / 3 if outs_recorded > 0 else 0
            if innings_pitched > 0:
                pitching_weight += innings_pitched
                for i in range(36, 75):
                    pitching_sum[i - 36] += player_retrosheet_row[i] * innings_pitched

            # Fielding Participation: chances (Field 75)
            chances = player_retrosheet_row[75]
            if chances > 0:
                fielding_weight += chances
                for i in range(75, 154):
                    fielding_sum[i - 75] += player_retrosheet_row[i] * chances

        # --- Calculate Weighted Averages ---

        # Batting Stats (Fields 11-35)
        for i in range(11, 36):
            if batting_weight > 0:
                team_retrosheet_row[i] = batting_sum[i - 11] / batting_weight
            else:
                team_retrosheet_row[i] = 0

        # Pitching Stats (Fields 36-74)
        for i in range(36, 75):
            if pitching_weight > 0:
                team_retrosheet_row[i] = pitching_sum[i - 36] / pitching_weight
            else:
                team_retrosheet_row[i] = 0

        # Fielding Stats (Fields 75-153)
        for i in range(75, 154):
            if fielding_weight > 0:
                team_retrosheet_row[i] = fielding_sum[i - 75] / fielding_weight
            else:
                team_retrosheet_row[i] = 0

        return team_retrosheet_row

    def reconstruct_retrosheet_row(self, game_id, game_json, player_key, player_info, is_home):
        """
        Reconstructs a Retrosheet statistics row for a single player in a game using seasonStats.

        Parameters:
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            player_key (str): Player identifier (e.g., "ID451594"). (Unused in this context)
            player_info (dict): Player information from the JSON.
            is_home (bool): Flag indicating if the player is on the home team.

        Returns:
            list: A list representing the Retrosheet statistics row with numeric values.
        """
        # Reuse the earlier reconstruct_retrosheet_row function
        # Since player_key is not used in aggregation, it's set to None when called from aggregate_team_retrosheet_stats
        return self._reconstruct_retrosheet_row(game_id, game_json, player_key, player_info, is_home)

    def _reconstruct_retrosheet_row(self, game_id, game_json, player_key, player_info, is_home):
        """
        Internal method to reconstruct Retrosheet row. Separated for clarity.
        """
        # Initialize a list with 154 default numeric values (zeros)
        retrosheet_row = [0] * 154

        # Field 0: Game ID
        retrosheet_row[0] = game_id

        # Field 1: Date (YYYY-MM-DD)
        game_date = game_json.get('gameDate', '')
        try:
            retrosheet_row[1] = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
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
        # Assuming park ID is available in `game_json['park']['id']`; else set to 0
        retrosheet_row[10] = game_json.get('park', {}).get('id', 0)

        # --- Batting Stats (Fields 11-35) ---
        batting_stats = player_info.get('stats', {}).get('batting', {})
        if batting_stats:
            # Field 11: B_G - Games Played
            retrosheet_row[11] = batting_stats.get('gamesPlayed', 0)
            # Field 12: B_PA - Plate Appearances
            retrosheet_row[12] = batting_stats.get('plateAppearances', 0)
            # Field 13: B_AB - At Bats
            retrosheet_row[13] = batting_stats.get('atBats', 0)
            # Field 14: B_R - Runs
            retrosheet_row[14] = batting_stats.get('runs', 0)
            # Field 15: B_H - Hits
            retrosheet_row[15] = batting_stats.get('hits', 0)
            # Field 16: B_TB - Total Bases
            retrosheet_row[16] = batting_stats.get('totalBases', 0)
            # Field 17: B_2B - Doubles
            retrosheet_row[17] = batting_stats.get('doubles', 0)
            # Field 18: B_3B - Triples
            retrosheet_row[18] = batting_stats.get('triples', 0)
            # Field 19: B_HR - Home Runs
            retrosheet_row[19] = batting_stats.get('homeRuns', 0)
            # Field 20: B_HR4 - Grand Slams
            retrosheet_row[20] = batting_stats.get('grandSlams', 0)  # Update based on actual data
            # Field 21: B_RBI - Runs Batted In
            retrosheet_row[21] = batting_stats.get('rbi', 0)
            # Field 22: B_GW - Game Winning RBI
            retrosheet_row[22] = batting_stats.get('gameWinningRbi', 0)  # Update based on actual data
            # Field 23: B_BB - Walks
            retrosheet_row[23] = batting_stats.get('baseOnBalls', 0)
            # Field 24: B_IBB - Intentional Walks
            retrosheet_row[24] = batting_stats.get('intentionalWalks', 0)
            # Field 25: B_SO - Strikeouts
            retrosheet_row[25] = batting_stats.get('strikeOuts', 0)
            # Field 26: B_GDP - Grounded Into Double Play
            retrosheet_row[26] = batting_stats.get('groundIntoDoublePlay', 0)
            # Field 27: B_HP - Hit By Pitch
            retrosheet_row[27] = batting_stats.get('hitByPitch', 0)
            # Field 28: B_SH - Sacrifice Hits
            retrosheet_row[28] = batting_stats.get('sacBunts', 0)
            # Field 29: B_SF - Sacrifice Flies
            retrosheet_row[29] = batting_stats.get('sacFlies', 0)
            # Field 30: B_SB - Stolen Bases
            retrosheet_row[30] = batting_stats.get('stolenBases', 0)
            # Field 31: B_CS - Caught Stealing
            retrosheet_row[31] = batting_stats.get('caughtStealing', 0)
            # Field 32: B_XI - Reached on Interference
            retrosheet_row[32] = batting_stats.get('catchersInterference', 0)
            # Field 33: B_G_DH - Games as DH
            # Assuming 'battingOrder' or another field indicates DH; adjust as needed
            batting_order = player_info.get('battingOrder', '')
            if isinstance(batting_order, str) and batting_order.upper() == 'DH':
                retrosheet_row[33] = 1
            else:
                retrosheet_row[33] = 0
            # Field 34: B_G_PH - Games as PH (Pinch Hitter)
            # Adjust based on actual data
            retrosheet_row[34] = batting_stats.get('gamesAsPH', 0)  # Update based on actual data
            # Field 35: B_G_PR - Games as PR (Pinch Runner)
            retrosheet_row[35] = batting_stats.get('gamesAsPR', 0)  # Update based on actual data
        else:
            # Assign default values if no batting stats
            for i in range(11, 36):
                retrosheet_row[i] = 0

        # --- Pitching Stats (Fields 36-74) ---
        pitching_stats = player_info.get('stats', {}).get('pitching', {})
        if pitching_stats:
            # Field 36: P_G - Games Pitched
            retrosheet_row[36] = pitching_stats.get('gamesPlayed', 0)
            # Field 37: P_GS - Games Started
            retrosheet_row[37] = pitching_stats.get('gamesStarted', 0)
            # Field 38: P_CG - Complete Games
            retrosheet_row[38] = pitching_stats.get('completeGames', 0)
            # Field 39: P_SHO - Shutouts
            retrosheet_row[39] = pitching_stats.get('shutouts', 0)
            # Field 40: P_GF - Games Finished
            retrosheet_row[40] = pitching_stats.get('gamesFinished', 0)
            # Field 41: P_W - Wins
            retrosheet_row[41] = pitching_stats.get('wins', 0)
            # Field 42: P_L - Losses
            retrosheet_row[42] = pitching_stats.get('losses', 0)
            # Field 43: P_SV - Saves
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
            # Field 45: P_TBF - Batters Faced
            retrosheet_row[45] = pitching_stats.get('battersFaced', 0)
            # Field 46: P_AB - At Bats (against)
            retrosheet_row[46] = pitching_stats.get('atBats', 0)
            # Field 47: P_R - Runs Allowed
            retrosheet_row[47] = pitching_stats.get('runs', 0)
            # Field 48: P_ER - Earned Runs Allowed
            retrosheet_row[48] = pitching_stats.get('earnedRuns', 0)
            # Field 49: P_H - Hits Allowed
            retrosheet_row[49] = pitching_stats.get('hits', 0)
            # Field 50: P_TB - Total Bases Allowed
            retrosheet_row[50] = pitching_stats.get('totalBases', 0)
            # Field 51: P_2B - Doubles Allowed
            retrosheet_row[51] = pitching_stats.get('doubles', 0)
            # Field 52: P_3B - Triples Allowed
            retrosheet_row[52] = pitching_stats.get('triples', 0)
            # Field 53: P_HR - Home Runs Allowed
            retrosheet_row[53] = pitching_stats.get('homeRuns', 0)
            # Field 54: P_HR4 - Grand Slams Allowed
            retrosheet_row[54] = pitching_stats.get('grandSlamsAllowed', 0)  # Update based on actual data
            # Field 55: P_BB - Walks Allowed
            retrosheet_row[55] = pitching_stats.get('walks', 0)
            # Field 56: P_IBB - Intentional Walks Allowed
            retrosheet_row[56] = pitching_stats.get('intentionalWalks', 0)
            # Field 57: P_SO - Strikeouts
            retrosheet_row[57] = pitching_stats.get('strikeOuts', 0)
            # Field 58: P_GDP - Grounded Into Double Play
            retrosheet_row[58] = pitching_stats.get('groundIntoDoublePlay', 0)
            # Field 59: P_HP - Hit Batsmen
            retrosheet_row[59] = pitching_stats.get('hitBatsmen', 0)
            # Field 60: P_SH - Sacrifice Hits Against
            retrosheet_row[60] = pitching_stats.get('sacHitsAgainst', 0)  # Update based on actual data
            # Field 61: P_SF - Sacrifice Flies Against
            retrosheet_row[61] = pitching_stats.get('sacFliesAgainst', 0)  # Update based on actual data
            # Field 62: P_XI - Reached on Interference
            retrosheet_row[62] = pitching_stats.get('reachedOnInterference', 0)  # Update based on actual data
            # Field 63: P_WP - Wild Pitches
            retrosheet_row[63] = pitching_stats.get('wildPitches', 0)
            # Field 64: P_BK - Balks
            retrosheet_row[64] = pitching_stats.get('balks', 0)
            # Field 65: P_IR - Inherited Runners
            retrosheet_row[65] = pitching_stats.get('inheritedRunners', 0)
            # Field 66: P_IRS - Inherited Runners Scored
            retrosheet_row[66] = pitching_stats.get('inheritedRunnersScored', 0)
            # Field 67: P_GO - Ground Outs
            retrosheet_row[67] = pitching_stats.get('groundOuts', 0)
            # Field 68: P_AO - Air Outs
            retrosheet_row[68] = pitching_stats.get('airOuts', 0)
            # Field 69: P_PITCH - Pitches
            retrosheet_row[69] = pitching_stats.get('numberOfPitches', 0)
            # Field 70: P_STRIKE - Strikes
            retrosheet_row[70] = pitching_stats.get('strikes', 0)
            # Field 71: F_P_G - Games at P (Pitcher)
            retrosheet_row[71] = pitching_stats.get('gamesAtP', 0)  # Update based on actual data
            # Field 72: F_P_GS - Games Started at P
            retrosheet_row[72] = pitching_stats.get('gamesStartedAtP', 0)  # Update based on actual data
            # Field 73: F_P_OUT - Outs Recorded at P (innings pitched * 3)
            retrosheet_row[73] = outs_recorded
            # Field 74: F_P_TC - Total Chances at P
            retrosheet_row[74] = pitching_stats.get('totalChancesAtP', 0)  # Update based on actual data
        else:
            # Assign default values if no pitching stats
            for i in range(36, 75):
                retrosheet_row[i] = 0

        # --- Fielding Stats (Fields 75-153) ---
        fielding_stats = player_info.get('stats', {}).get('fielding', {})
        position_abbr = player_info.get('position', {}).get('abbreviation', '').upper()
        if fielding_stats:
            # Mapping fielding stats based on position
            # Define a helper dictionary to map position abbreviations to field indices
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

    def aggregate_team_retrosheet_stats(self, lineup, game_id, game_json, team_type):
        """
        Aggregates Retrosheet statistics for an entire team based on individual player stats using seasonStats.

        Parameters:
            lineup (list of int): List of player IDs in the team's lineup.
            game_id (str): Unique identifier for the game.
            game_json (dict): JSON data for the game.
            team_type (str): 'home' or 'away'.

        Returns:
            list: A list representing the aggregated Retrosheet statistics row for the team.
        """

        # Initialize accumulators for batting, pitching, and fielding
        batting_sum = [0] * 25  # Fields 11-35 (25 fields)
        batting_weight = 0

        pitching_sum = [0] * 39  # Fields 36-74 (39 fields)
        pitching_weight = 0

        fielding_sum = [0] * 79  # Fields 75-153 (79 fields)
        fielding_weight = 0

        # Initialize the aggregated team Retrosheet row with numeric zeros
        team_retrosheet_row = [0] * 154

        # Assign game info fields (0-10)
        team_retrosheet_row[0] = game_id

        # Field 1: Date (YYYY-MM-DD)
        game_date = game_json.get('gameDate', '')
        try:
            team_retrosheet_row[1] = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
        except:
            team_retrosheet_row[1] = 0  # Default to 0 if date is invalid

        # Field 2: Game number (0 = no double header)
        team_retrosheet_row[2] = game_json.get('gameNumber', 0)

        # Field 3: Appearance date (same as date)
        team_retrosheet_row[3] = team_retrosheet_row[1]

        # Field 4: Team ID
        team_id = game_json['scoreboard']['teams'][team_type]['team']['id']
        team_retrosheet_row[4] = team_id

        # Field 5: Player ID (Using team_id as a placeholder)
        team_retrosheet_row[5] = team_id  # Alternatively, use a special identifier

        # Fields 6-7: Batting Order Slot and Sequence (set to 0 for team-level)
        team_retrosheet_row[6] = 0
        team_retrosheet_row[7] = 0

        # Field 8: Home flag (1 if home, 0 if away)
        team_retrosheet_row[8] = 1 if team_type == 'home' else 0

        # Field 9: Opponent ID
        opponent_team_id = game_json['scoreboard']['teams']['away']['team']['id'] if team_type == 'home' else \
        game_json['scoreboard']['teams']['home']['team']['id']
        team_retrosheet_row[9] = opponent_team_id

        # Field 10: Park ID
        team_retrosheet_row[10] = game_json.get('park', {}).get('id', 0)

        # Iterate through each player in the lineup
        for player_id in lineup:
            # Retrieve player_info
            player_info = self.get_player_info(player_id, game_json, team_type)
            if player_info is None:
                continue  # Skip if player not found

            # Replace 'stats' with 'seasonStats' to use season statistics
            if 'seasonStats' in player_info:
                player_info = player_info.copy()  # To avoid mutating the original data
                player_info['stats'] = player_info['seasonStats']
            else:
                # If 'seasonStats' not available, use existing 'stats'
                pass

            # Determine if the player is on the home team
            is_home = 1 if team_type == 'home' else 0

            # Reconstruct the player's Retrosheet row
            player_retrosheet_row = self.reconstruct_retrosheet_row(game_id, game_json, None, player_info, is_home)

            # --- Participation Metrics ---

            # Batting Participation: atBats (Field 13)
            at_bats = player_retrosheet_row[13]
            if at_bats > 0:
                batting_weight += at_bats
                for i in range(11, 36):
                    batting_sum[i - 11] += player_retrosheet_row[i] * at_bats

            # Pitching Participation: inningsPitched = outsRecorded / 3 (Field 44)
            outs_recorded = player_retrosheet_row[44]
            innings_pitched = outs_recorded / 3 if outs_recorded > 0 else 0
            if innings_pitched > 0:
                pitching_weight += innings_pitched
                for i in range(36, 75):
                    pitching_sum[i - 36] += player_retrosheet_row[i] * innings_pitched

            # Fielding Participation: chances (Field 75)
            chances = player_retrosheet_row[75]
            if chances > 0:
                fielding_weight += chances
                for i in range(75, 154):
                    fielding_sum[i - 75] += player_retrosheet_row[i] * chances

        # --- Calculate Weighted Averages ---

        # Batting Stats (Fields 11-35)
        for i in range(11, 36):
            if batting_weight > 0:
                team_retrosheet_row[i] = batting_sum[i - 11] / batting_weight
            else:
                team_retrosheet_row[i] = 0

        # Pitching Stats (Fields 36-74)
        for i in range(36, 75):
            if pitching_weight > 0:
                team_retrosheet_row[i] = pitching_sum[i - 36] / pitching_weight
            else:
                team_retrosheet_row[i] = 0

        # Fielding Stats (Fields 75-153)
        for i in range(75, 154):
            if fielding_weight > 0:
                team_retrosheet_row[i] = fielding_sum[i - 75] / fielding_weight
            else:
                team_retrosheet_row[i] = 0

        return team_retrosheet_row

    def process_games_retrosheet(self):
        """
        Processes all games and returns a DataFrame where each row represents a game.
        Each row concatenates the Retrosheet rows of the home and away teams,
        with the home team on the left and the away team on the right.
        Includes team names and abbreviations for both teams.

        Returns:
            pd.DataFrame: DataFrame containing concatenated Retrosheet rows for all games.
        """
        gamelogs = self.get_unique_game_jsons()
        retrosheet_rows = []

        for game_json in gamelogs:
            game_id = game_json['scoreboard']['gamePk']
            game_date = game_json['gameDate']

            # Extract Home Team Information
            home_team_info = game_json['boxscore']['teams']['home']['team']
            home_team_id = home_team_info['id']
            home_team_name = home_team_info['name']
            home_team_abbr = home_team_info.get('abbreviation', '')

            # Extract Away Team Information
            away_team_info = game_json['boxscore']['teams']['away']['team']
            away_team_id = away_team_info['id']
            away_team_name = away_team_info['name']
            away_team_abbr = away_team_info.get('abbreviation', '')

            # Get Home and Away Lineups (List of player_ids)
            home_players = game_json['boxscore']['teams']['home'].get('players', {})
            home_lineup = [info['person']['id'] for key, info in home_players.items()]

            away_players = game_json['boxscore']['teams']['away'].get('players', {})
            away_lineup = [info['person']['id'] for key, info in away_players.items()]

            # Aggregate Home Team Retrosheet Stats using seasonStats
            home_retrosheet_row = self.aggregate_team_retrosheet_stats(
                lineup=home_lineup,
                game_id=game_id,
                game_json=game_json,
                team_type='home'
            )

            # Aggregate Away Team Retrosheet Stats using seasonStats
            away_retrosheet_row = self.aggregate_team_retrosheet_stats(
                lineup=away_lineup,
                game_id=game_id,
                game_json=game_json,
                team_type='away'
            )

            # Concatenate Home and Away Retrosheet Rows
            concatenated_row = home_retrosheet_row + away_retrosheet_row

            # Include Team Names and Abbreviations
            # Append them at the end of the row
            concatenated_row += [home_team_name, home_team_abbr, away_team_name, away_team_abbr, game_id]

            retrosheet_rows.append(concatenated_row)

        # Define column names
        # Retrosheet fields 0-153 for Home Team, 154-307 for Away Team
        # Plus 308-312 for Team Names, Abbreviations, and gamePk
        home_columns = [f"Home_Field_{i}" for i in range(154)]
        away_columns = [f"Away_Field_{i}" for i in range(154)]
        team_info_columns = ['Home_Team_Name', 'Home_Team_Abbr', 'Away_Team_Name', 'Away_Team_Abbr', 'Game_PK']
        all_columns = home_columns + away_columns + team_info_columns

        # Create DataFrame
        df = pd.DataFrame(retrosheet_rows, columns=all_columns)

        return df

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

    def process_games(self):
        """
        Processes all games and returns a DataFrame of extracted player data.

        Returns:
            pd.DataFrame: DataFrame containing player data for all games.
        """
        games_json = self.get_unique_game_jsons()
        all_player_data = []
        for game_json in games_json:
            player_data = self.extract_player_data(game_json)
            all_player_data.extend(player_data)

        df = pd.DataFrame(all_player_data)
        return df

    @staticmethod
    def extract_player_data(game_json):
        """
        Extracts player data from a single game JSON.

        Parameters:
            game_json (dict): JSON data for the game.

        Returns:
            list: List of player data dictionaries.
        """
        game_id = game_json['scoreboard']['gamePk']
        game_date = game_json['gameDate']
        player_data = []

        for player_id, player_info in game_json['players'].items():
            base_record = {
                'game_id': game_id,
                'date': game_date,
                'player_id': player_info['person']['id'],
                'player_name': player_info['person']['fullName'],
                'jersey_number': player_info.get('jerseyNumber'),
                'position_code': player_info['position']['code'],
                'position_name': player_info['position']['name'],
                'batting_order': player_info.get('battingOrder'),
                'status_code': player_info['status']['code'],
                'status_description': player_info['status']['description'],
                'team_id': player_info['parentTeamId'],
            }

            # Flatten stats, season_stats, game_status, all_positions
            if 'stats' in player_info:
                for stat_type, stats in player_info['stats'].items():
                    for stat_key, stat_value in stats.items():
                        base_record[f"{stat_type}_{stat_key}"] = stat_value

            if 'seasonStats' in player_info:
                for stat_type, stats in player_info['seasonStats'].items():
                    for stat_key, stat_value in stats.items():
                        base_record[f"season_{stat_type}_{stat_key}"] = stat_value

            if 'gameStatus' in player_info:
                for status_key, status_value in player_info['gameStatus'].items():
                    base_record[f"game_status_{status_key}"] = status_value

            if 'allPositions' in player_info:
                for i, position in enumerate(player_info['allPositions']):
                    for pos_key, pos_value in position.items():
                        base_record[f"position_{i}_{pos_key}"] = pos_value

            player_data.append(base_record)

        return player_data
