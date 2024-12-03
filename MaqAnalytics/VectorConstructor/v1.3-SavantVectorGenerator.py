import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from typing import List

import aiohttp
import requests
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from .player_features import BattingStats, PitchingStats, FieldingStats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_current_date():
    """
    Retrieves the current date in "MM/DD/YYYY" format.

    Returns:
        str: Current date as a string.
    """
    return datetime.now().strftime("%m/%d/%Y")


@dataclass
class StartingBatters:
    batters: List[BattingStats] = field(default_factory=lambda: [BattingStats() for _ in range(9)])


@dataclass
class BenchBatters:
    aggregated_stats: BattingStats = BattingStats()


@dataclass
class StartingPitcher:
    pitcher: PitchingStats = PitchingStats()


@dataclass
class BullpenPitchers:
    aggregated_stats: PitchingStats = PitchingStats()


@dataclass
class StartingInfielders:
    infielders: List[FieldingStats] = field(
        default_factory=lambda: [FieldingStats() for _ in range(4)])  # 1B, 2B, 3B, SS


@dataclass
class StartingOutfielders:
    outfielders: List[FieldingStats] = field(default_factory=lambda: [FieldingStats() for _ in range(3)])  # LF, CF, RF


@dataclass
class BenchedInfielders:
    aggregated_stats: FieldingStats = FieldingStats()


@dataclass
class BenchedOutfielders:
    aggregated_stats: FieldingStats = FieldingStats()


@dataclass
class TeamStats:
    starting_batters: StartingBatters = StartingBatters()
    bench_batters: BenchBatters = BenchBatters()
    starting_pitcher: StartingPitcher = StartingPitcher()
    bullpen_pitchers: BullpenPitchers = BullpenPitchers()
    starting_infielders: StartingInfielders = StartingInfielders()
    starting_outfielders: StartingOutfielders = StartingOutfielders()
    benched_infielders: BenchedInfielders = BenchedInfielders()
    benched_outfielders: BenchedOutfielders = BenchedOutfielders()


class SavantVectorGenerator:
    def __init__(self, gamePk: int):
        """
        Initializes the SavantVectorGenerator with a specific gamePk.

        Parameters:
            gamePk (int): The unique Baseball Savant game id
        """
        self.gamePk = gamePk
        self.base_gamelog_url = f"https://baseballsavant.mlb.com/gf?game_pk={gamePk}"
        self.game_json = self.get_game_json()

    def reconstruct_player_stats(self, game_id: int, player_info: dict, is_home: bool,
                                 is_pitcher: bool = False) -> dict:
        """
        Reconstructs a statistics dictionary for a single player based on season stats.

        Parameters:
            game_id (int): Unique identifier for the game.
            player_info (dict): Player information from the JSON.
            is_home (bool): Flag indicating if the player is on the home team.
            is_pitcher (bool): Indicates if the player is a pitcher.

        Returns:
            dict: A dictionary representing the player's statistics with prefixed keys.
        """
        stats_dict = {}

        position_abb = player_info.get('position', {}).get('abbreviation', 'None').upper()

        # Basic Fields
        stats_dict['game_id'] = game_id
        game_date = self.game_json.get('gameDate', '')
        try:
            # Convert from "MM/DD/YYYY" to "YYYY-MM-DD"
            parsed_date = datetime.strptime(game_date, "%m/%d/%Y").strftime('%Y-%m-%d')
        except:
            parsed_date = '1970-01-01'  # Default date if parsing fails
        stats_dict['date'] = parsed_date
        stats_dict['appearance_date'] = parsed_date
        stats_dict['team_id'] = player_info.get('parentTeamId', 0)
        stats_dict['player_id'] = player_info.get('person', {}).get('id', 0)
        stats_dict['batting_order'] = 0  # Placeholder
        stats_dict['batting_order_sequence'] = 0  # Placeholder
        stats_dict['home_flag'] = 1 if is_home else 0

        # Opponent ID
        home_team_id = self.game_json['boxscore']['teams']['home']['team']['id']
        away_team_id = self.game_json['boxscore']['teams']['away']['team']['id']
        opponent_id = away_team_id if is_home else home_team_id
        stats_dict['opponent_id'] = opponent_id

        # Park ID
        stats_dict['park_id'] = self.game_json.get('park', {}).get('id', 0)

        # Helper function to adjust stats
        def get_adjusted_stats(stat_group, season_stats, game_stats):
            adjusted_stats = {}
            for stat in stat_group.stats:
                stat_key = stat.value
                season_stat = season_stats.get(stat_key, 0)
                game_stat = game_stats.get(stat_key, 0)

                try:
                    season_stat = float(season_stat)
                except (ValueError, TypeError):
                    season_stat = 0

                try:
                    game_stat = float(game_stat)
                except (ValueError, TypeError):
                    game_stat = 0

                adjusted_stats[stat_key] = season_stat - game_stat
            return adjusted_stats

        # Batting Stats
        if not is_pitcher:
            season_batting = player_info.get('seasonStats', {}).get('batting', {})
            game_batting = player_info.get('stats', {}).get('batting', {})
            adjusted_batting = get_adjusted_stats(BattingStats(), season_batting, game_batting)
            for stat in BattingStats:
                stat_key = stat.value
                stats_dict[f"B_{stat_key}"] = adjusted_batting.get(stat_key, 0)

        # Pitching Stats
        if is_pitcher:
            season_pitching = player_info.get('seasonStats', {}).get('pitching', {})
            game_pitching = player_info.get('stats', {}).get('pitching', {})
            adjusted_pitching = get_adjusted_stats(PitchingStats(), season_pitching, game_pitching)
            for stat in PitchingStats:
                stat_key = stat.value
                stats_dict[f"P_{stat_key}"] = adjusted_pitching.get(stat_key, 0)

        # Fielding Stats
        season_fielding = player_info.get('seasonStats', {}).get('fielding', {})
        game_fielding = player_info.get('stats', {}).get('fielding', {})
        adjusted_fielding = get_adjusted_stats(FieldingStats(), season_fielding, game_fielding)
        for stat in FieldingStats:
            stat_key = stat.value
            stats_dict[f"F_{stat_key}"] = adjusted_fielding.get(stat_key, 0)

        # Normalize stats by games played to get per-game averages
        try:
            if is_pitcher:
                games_played = float(player_info.get('seasonStats', {}).get('pitching', {}).get('gamesPlayed', 1)) - 1
                innings_pitched = float(
                    player_info.get('seasonStats', {}).get('pitching', {}).get('innings_pitched', 0)) - float(
                    player_info.get('stats', {}).get('pitching', {}).get('innings_pitched', 0))
                if innings_pitched <= 0:
                    innings_pitched = 1  # Avoid division by zero
            else:
                games_played = float(player_info.get('seasonStats', {}).get('batting', {}).get('gamesPlayed', 1)) - 1
                innings_pitched = 0  # Not applicable for non-pitchers
        except (ValueError, TypeError):
            games_played = 1
            innings_pitched = 1 if is_pitcher else 0

        if games_played <= 0:
            logger.warning(
                f"gamePk {game_id}: gamesPlayed is {games_played} for player {stats_dict['player_id']}. Defaulting to 1.")
            games_played = 1  # Avoid division by zero

        # Normalize Batting Stats
        if not is_pitcher:
            for stat in BattingStats:
                stat_key = stat.value
                stats_dict[f"B_{stat_key}"] /= games_played

        # Normalize Pitching Stats
        if is_pitcher:
            for stat in PitchingStats:
                stat_key = stat.value
                # Define which stats to normalize by innings_pitched and which by games_played
                if stat_key in {
                    "groundOuts", "airOuts", "runs", "earnedRuns", "hits", "doubles",
                    "triples", "homeRuns", "strikeOuts", "baseOnBalls", "intentional_walks",
                    "hitByPitch", "wild_pitches", "balks", "batters_faced", "number_of_pitches",
                    "stolen_bases", "caught_stealing", "outs", "pitches_thrown", "balls", "strikes",
                    "hit_batsmen", "at_bats", "pickoffs", "sac_bunts", "sac_flies", "catchers_interference",
                    "inherited_runners_scored", "rbi", "passed_ball"
                }:
                    stats_dict[f"P_{stat_key}"] /= innings_pitched
                elif stat_key in {
                    "games_played", "games_started", "wins", "losses", "saves",
                    "save_opportunities", "complete_games", "shutouts", "holds",
                    "blown_saves", "inherited_runners", "inherited_runners_scored", "games_finished"
                }:
                    stats_dict[f"P_{stat_key}"] /= games_played
                else:
                    # Leave as-is or apply other normalization if needed
                    pass

        # Normalize Fielding Stats
        for stat in FieldingStats:
            stat_key = stat.value
            stats_dict[f"F_{stat_key}"] /= games_played

        return stats_dict

    def aggregate_team_stats(self, team_type: str) -> TeamStats:
        """
        Aggregates all relevant stats for a given team in the game.

        Parameters:
            team_type (str): 'home' or 'away'.

        Returns:
            TeamStats: Aggregated statistics for the team.
        """
        team_stats = TeamStats()

        team_key = 'home' if team_type == 'home' else 'away'
        lineup = self.game_json[f'{team_key}_lineup']
        bench = self.game_json['boxscore']['teams'][team_key]['bench']
        batters = self.game_json['boxscore']['teams'][team_key]['batters']
        pitchers = self.game_json['boxscore']['teams'][team_key]['pitchers']
        bullpen = self.game_json['boxscore']['teams'][team_key]['bullpen']

        # Starting Batters
        for i, batter_id in enumerate(lineup.get('batters', [])):
            player_info = self.get_player_info(batter_id, team_type)
            if player_info:
                player_stats = self.reconstruct_player_stats(
                    game_id=self.gamePk,
                    player_info=player_info,
                    is_home=(team_type == 'home'),
                    is_pitcher=False
                )
                batting_stats = BattingStats(
                    **{k.replace('B_', ''): v for k, v in player_stats.items() if k.startswith('B_')})
                team_stats.starting_batters.batters[i] = batting_stats

        # Bench Batters - Aggregated Stats
        for batter_id in bench.get('batters', []):
            player_info = self.get_player_info(batter_id, team_type)
            if player_info:
                player_stats = self.reconstruct_player_stats(
                    game_id=self.gamePk,
                    player_info=player_info,
                    is_home=(team_type == 'home'),
                    is_pitcher=False
                )
                for key, value in player_stats.items():
                    if key.startswith('B_'):
                        stat_key = key.replace('B_', '')
                        setattr(team_stats.bench_batters.aggregated_stats, stat_key,
                                getattr(team_stats.bench_batters.aggregated_stats, stat_key) + value)

        # Normalize Bench Batters by number of bench batters
        num_bench_batters = len(bench.get('batters', []))
        if num_bench_batters > 0:
            for field in vars(team_stats.bench_batters.aggregated_stats):
                setattr(team_stats.bench_batters.aggregated_stats, field,
                        getattr(team_stats.bench_batters.aggregated_stats, field) / num_bench_batters)

        # Starting Pitcher
        starting_pitcher_id = lineup.get('pitcher')
        if starting_pitcher_id:
            pitcher_info = self.get_player_info(starting_pitcher_id, team_type)
            if pitcher_info:
                pitcher_stats = self.reconstruct_player_stats(
                    game_id=self.gamePk,
                    player_info=pitcher_info,
                    is_home=(team_type == 'home'),
                    is_pitcher=True
                )
                pitching_stats = PitchingStats(
                    **{k.replace('P_', ''): v for k, v in pitcher_stats.items() if k.startswith('P_')})
                team_stats.starting_pitcher.pitcher = pitching_stats

        # Bullpen Pitchers - Aggregated Stats
        for pitcher_id in bullpen:
            pitcher_info = self.get_player_info(pitcher_id, team_type)
            if pitcher_info:
                pitcher_stats = self.reconstruct_player_stats(
                    game_id=self.gamePk,
                    player_info=pitcher_info,
                    is_home=(team_type == 'home'),
                    is_pitcher=True
                )
                for key, value in pitcher_stats.items():
                    if key.startswith('P_'):
                        stat_key = key.replace('P_', '')
                        setattr(team_stats.bullpen_pitchers.aggregated_stats, stat_key,
                                getattr(team_stats.bullpen_pitchers.aggregated_stats, stat_key) + value)

        # Normalize Bullpen Pitchers by number of bullpen pitchers
        num_bullpen_pitchers = len(bullpen)
        if num_bullpen_pitchers > 0:
            for field in vars(team_stats.bullpen_pitchers.aggregated_stats):
                setattr(team_stats.bullpen_pitchers.aggregated_stats, field,
                        getattr(team_stats.bullpen_pitchers.aggregated_stats, field) / num_bullpen_pitchers)

        # Starting Infielders
        for i, infielder_id in enumerate(lineup.get('infielders', [])):
            player_info = self.get_player_info(infielder_id, team_type)
            if player_info:
                fielding_stats = self.reconstruct_player_stats(
                    game_id=self.gamePk,
                    player_info=player_info,
                    is_home=(team_type == 'home'),
                    is_pitcher=False
                )
                f_stats = FieldingStats(
                    **{k.replace('F_', ''): v for k, v in fielding_stats.items() if k.startswith('F_')})
                team_stats.starting_infielders.infielders[i] = f_stats

        # Starting Outfielders
        for i, outfielder_id in enumerate(lineup.get('outfielders', [])):
            player_info = self.get_player_info(outfielder_id, team_type)
            if player_info:
                fielding_stats = self.reconstruct_player_stats(
                    game_id=self.gamePk,
                    player_info=player_info,
                    is_home=(team_type == 'home'),
                    is_pitcher=False
                )
                f_stats = FieldingStats(
                    **{k.replace('F_', ''): v for k, v in fielding_stats.items() if k.startswith('F_')})
                team_stats.starting_outfielders.outfielders[i] = f_stats

        # Benched Infielders - Aggregated Stats
        for infielder_id in bench.get('infielders', []):
            player_info = self.get_player_info(infielder_id, team_type)
            if player_info:
                fielding_stats = self.reconstruct_player_stats(
                    game_id=self.gamePk,
                    player_info=player_info,
                    is_home=(team_type == 'home'),
                    is_pitcher=False
                )
                for key, value in fielding_stats.items():
                    if key.startswith('F_'):
                        stat_key = key.replace('F_', '')
                        setattr(team_stats.benched_infielders.aggregated_stats, stat_key,
                                getattr(team_stats.benched_infielders.aggregated_stats, stat_key) + value)

        # Normalize Benched Infielders by number of benched infielders
        num_benched_infielders = len(bench.get('infielders', []))
        if num_benched_infielders > 0:
            for field in vars(team_stats.benched_infielders.aggregated_stats):
                setattr(team_stats.benched_infielders.aggregated_stats, field,
                        getattr(team_stats.benched_infielders.aggregated_stats, field) / num_benched_infielders)

        # Benched Outfielders - Aggregated Stats
        for outfielder_id in bench.get('outfielders', []):
            player_info = self.get_player_info(outfielder_id, team_type)
            if player_info:
                fielding_stats = self.reconstruct_player_stats(
                    game_id=self.gamePk,
                    player_info=player_info,
                    is_home=(team_type == 'home'),
                    is_pitcher=False
                )
                for key, value in fielding_stats.items():
                    if key.startswith('F_'):
                        stat_key = key.replace('F_', '')
                        setattr(team_stats.benched_outfielders.aggregated_stats, stat_key,
                                getattr(team_stats.benched_outfielders.aggregated_stats, stat_key) + value)

        # Normalize Benched Outfielders by number of benched outfielders
        num_benched_outfielders = len(bench.get('outfielders', []))
        if num_benched_outfielders > 0:
            for field in vars(team_stats.benched_outfielders.aggregated_stats):
                setattr(team_stats.benched_outfielders.aggregated_stats, field,
                        getattr(team_stats.benched_outfielders.aggregated_stats, field) / num_benched_outfielders)

        return team_stats

    def prefix_team_stats(self, team_stats: TeamStats, prefix: str) -> dict:
        """
        Prefixes all keys in TeamStats with a given prefix.

        Parameters:
            team_stats (TeamStats): The team's statistics.
            prefix (str): The prefix to add (e.g., 'Home_', 'Away_').

        Returns:
            dict: Prefixed statistics as a flat dictionary.
        """
        prefixed_stats = {}

        # Starting Batters
        for i, batter in enumerate(team_stats.starting_batters.batters, start=1):
            for field, value in batter.__dict__.items():
                key = f"{prefix}StartingBatter{i}_{field}"
                prefixed_stats[key] = value

        # Bench Batters
        for field, value in team_stats.bench_batters.aggregated_stats.__dict__.items():
            key = f"{prefix}BenchBatters_{field}"
            prefixed_stats[key] = value

        # Starting Pitcher
        pitcher = team_stats.starting_pitcher.pitcher
        for field, value in pitcher.__dict__.items():
            key = f"{prefix}StartingPitcher_{field}"
            prefixed_stats[key] = value

        # Bullpen Pitchers
        for field, value in team_stats.bullpen_pitchers.aggregated_stats.__dict__.items():
            key = f"{prefix}BullpenPitchers_{field}"
            prefixed_stats[key] = value

        # Starting Infielders
        for i, infielder in enumerate(team_stats.starting_infielders.infielders, start=1):
            for field, value in infielder.__dict__.items():
                key = f"{prefix}StartingInfielder{i}_{field}"
                prefixed_stats[key] = value

        # Starting Outfielders
        for i, outfielder in enumerate(team_stats.starting_outfielders.outfielders, start=1):
            for field, value in outfielder.__dict__.items():
                key = f"{prefix}StartingOutfielder{i}_{field}"
                prefixed_stats[key] = value

        # Benched Infielders
        for field, value in team_stats.benched_infielders.aggregated_stats.__dict__.items():
            key = f"{prefix}BenchedInfielders_{field}"
            prefixed_stats[key] = value

        # Benched Outfielders
        for field, value in team_stats.benched_outfielders.aggregated_stats.__dict__.items():
            key = f"{prefix}BenchedOutfielders_{field}"
            prefixed_stats[key] = value

        return prefixed_stats

    def process_game(self) -> pd.DataFrame:
        """
        Processes a single game and returns a DataFrame containing feature vectors for both teams.

        Returns:
            pd.DataFrame: DataFrame containing feature vectors for the game.
        """
        retrosheet_row = {}

        # Extract Team Abbreviations
        home_team_info = self.game_json.get('home_team_data', {})
        away_team_info = self.game_json.get('away_team_data', {})
        home_team_abbr = home_team_info.get('abbreviation', '')
        away_team_abbr = away_team_info.get('abbreviation', '')

        # Aggregate Home Team Stats
        home_team_stats = self.aggregate_team_stats('home')
        home_prefixed = self.prefix_team_stats(home_team_stats, 'Home_')

        # Aggregate Away Team Stats
        away_team_stats = self.aggregate_team_stats('away')
        away_prefixed = self.prefix_team_stats(away_team_stats, 'Away_')

        # Combine Stats
        retrosheet_row.update(home_prefixed)
        retrosheet_row.update(away_prefixed)

        # Add Game Metadata
        retrosheet_row.update({
            'Game_Date': self.game_json.get('gameDate', ''),
            'Game_PK': self.gamePk,
            'Home_Team_Abbr': home_team_abbr,
            'Away_Team_Abbr': away_team_abbr,
            'park_id': self.game_json.get('park', {}).get('id', 'Unknown'),
        })

        # Convert to DataFrame
        df = pd.DataFrame([retrosheet_row])

        # Add Game Outcome
        df = self.add_game_outcome(df)

        return df

    def add_game_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a target variable indicating whether the home team won (1) or lost (0).

        Parameters:
            df (pd.DataFrame): DataFrame containing Retrosheet game feature vectors.

        Returns:
            pd.DataFrame: Updated DataFrame with 'Home_Win' column added.
        """
        try:
            home_runs = self.game_json['scoreboard']['linescore']['teams']['home']['runs']
            away_runs = self.game_json['scoreboard']['linescore']['teams']['away']['runs']
            home_win = 1 if home_runs > away_runs else 0
        except Exception as e:
            logger.error(f"Error processing game outcome for gamePk {self.gamePk}: {e}")
            home_win = 0  # Default to loss if error occurs

        df['Home_Win'] = home_win

        return df

    def get_player_info(self, player_id: int, team_type: str) -> dict:
        """
        Retrieves the player information dictionary from the game JSON.

        Parameters:
            player_id (int): Unique identifier for the player.
            team_type (str): 'home' or 'away'.

        Returns:
            dict or None: Player information dictionary or None if not found.
        """
        team_players = self.game_json['boxscore']['teams'][team_type]['players']
        for player_key, info in team_players.items():
            if info.get('person', {}).get('id') == player_id:
                return info
        logger.warning(f"Player stats not found for player_id {player_id} in team {team_type}")
        return None

    def get_game_json(self) -> dict:
        """
        Retrieves the game JSON for the specific gamePk.

        Returns:
            dict: Game JSON dictionary.
        """
        try:
            response = requests.get(self.base_gamelog_url)
            if response.status_code != 200:
                logger.error(f"Failed to fetch gamelog for gamePk {self.gamePk}: {response.status_code}")
                return {}
            data = response.json()
            logger.info(f"Successfully fetched gamelog for gamePk {self.gamePk}")
            return data
        except Exception as e:
            logger.error(f"Exception occurred while fetching gamePk {self.gamePk}: {e}")
            return {}

    def reconstruct_player_stats_aggregated(self, team_stats: TeamStats, team_type: str):
        """
        Placeholder for any additional processing if needed.

        Parameters:
            team_stats (TeamStats): Aggregated team statistics.
            team_type (str): 'home' or 'away'.
        """
        # Implement any additional aggregation or processing if necessary
        pass
