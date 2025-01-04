#!/usr/bin/env python3
import os
import asyncio
import json
import logging
import requests
import pandas as pd
import aiohttp

from datetime import datetime, timedelta
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------
#  Stat Maps (same as your code)
# ----------------------------------
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
    "stolenBasePercentage": "B_SB_PCT",
    "leftOnBase": "B_LOB",
    "sacBunts": "B_SH",
    "sacFlies": "B_SF",
    "catchersInterference": "B_XI",
    "pickoffs": "B_PK",
    "atBatsPerHomeRun": "B_AB_HR",
    "flyOuts": "B_FO",
    "groundOuts": "B_GO"
}

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
    "obp": "P_OBP",
    "caughtStealing": "P_CS",
    "stolenBases": "P_SB",
    "stolenBasePercentage": "P_SB_PCT",
    "numberOfPitches": "P_PITCH",
    "era": "P_ERA",
    "inningsPitched": "P_OUT",
    "wins": "P_W",
    "losses": "P_L",
    "saves": "P_SV",
    "saveOpportunities": "P_SVO",
    "holds": "P_HOLD",
    "blownSaves": "P_BLSV",
    "earnedRuns": "P_ER",
    "whip": "P_WHIP",
    "battersFaced": "P_TBF",
    "outs": "P_OUTS",
    "completeGames": "P_CG",
    "shutouts": "P_SHO",
    "pitchesThrown": "P_PITCHES",
    "balls": "P_BALLS",
    "strikes": "P_STRIKES",
    "strikePercentage": "P_STRIKE_PCT",
    "hitBatsmen": "P_HBP",
    "balks": "P_BK",
    "wildPitches": "P_WP",
    "pickoffs": "P_PK",
    "groundOutsToAirouts": "P_GO_AO",
    "rbi": "P_RBI",
    "winPercentage": "P_W_PCT",
    "pitchesPerInning": "P_PITCHES_IP",
    "gamesFinished": "P_GF",
    "strikeoutWalkRatio": "P_SO_BB",
    "strikeoutsPer9Inn": "P_SO9",
    "walksPer9Inn": "P_BB9",
    "hitsPer9Inn": "P_H9",
    "runsScoredPer9": "P_R9",
    "homeRunsPer9": "P_HR9",
    "inheritedRunners": "P_IR",
    "inheritedRunnersScored": "P_IRS",
    "catchersInterference": "P_CI",
    "sacBunts": "P_SH",
    "sacFlies": "P_SF",
    "passedBall": "P_PB",
}

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


class SavantVectorGenerator:
    """
    Example generator that processes month-by-month:
    1) For each month, fetch game_pks
    2) async fetch+process for that month
    3) append to CSV
    """

    def __init__(
            self,
            start_date="01/01/2021",
            end_date="12/31/2024",
            batters_csv="batters.csv",
            pitchers_csv="pitchers.csv",
            concurrency=5
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.batters_csv = batters_csv
        self.pitchers_csv = pitchers_csv
        self.concurrency = concurrency  # max parallel connections

        self.base_schedule_url = (
            "https://statsapi.mlb.com/api/v1/schedule?"
            "sportId=1&startDate={start_date}&endDate={end_date}"
            "&fields=dates,date,games,gamePk"
        )
        self.base_gamelog_url = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"

        # Keep track of whether CSVs have headers
        self.batters_initialized = False
        self.pitchers_initialized = False

        # Lock for writing CSVs in parallel
        self.csv_lock = asyncio.Lock()

    @staticmethod
    def month_chunks(start_dt, end_dt):
        """
        Generator that yields (chunk_start, chunk_end) datetime pairs
        for each month between start_dt and end_dt (inclusive).
        """
        current = start_dt.replace(day=1)
        while current <= end_dt:
            # end of the current month (approx by next month's 1 minus one day)
            next_month = (current.month % 12) + 1
            next_year = current.year + (current.month // 12)
            month_end = (datetime(next_year, next_month, 1) - timedelta(days=1))

            chunk_start = current
            chunk_end = min(month_end, end_dt)
            yield (chunk_start, chunk_end)

            # move to the next month
            current = (month_end + timedelta(days=1)).replace(day=1)

    def fetch_game_pks_for_chunk(self, chunk_start, chunk_end):
        """
        Fetch all game_pks for the chunk [chunk_start, chunk_end].
        """
        game_pks = []
        url = self.base_schedule_url.format(
            start_date=chunk_start.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d")
        )
        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch schedule chunk {chunk_start} -> {chunk_end}, code={resp.status_code}")
            else:
                data = resp.json()
                for date_info in data.get('dates', []):
                    for gm in date_info.get('games', []):
                        game_pks.append(gm['gamePk'])
        except Exception as e:
            logger.error(f"Exception fetching chunk {chunk_start}->{chunk_end}: {e}")

        logger.info(
            f"  Chunk {chunk_start.strftime('%Y-%m')} -> {chunk_end.strftime('%Y-%m-%d')} has {len(game_pks)} games.")
        return game_pks

    async def fetch_one_game(self, session, game_pk, pbar, max_retries=3):
        """
        Fetch 1 game JSON from Baseball Savant, w/ small retry logic.
        """
        url = self.base_gamelog_url.format(game_pk=game_pk)
        attempt = 0
        data = None

        while attempt < max_retries:
            try:
                # short delay to avoid flooding
                await asyncio.sleep(0.15)

                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Game {game_pk} HTTP {response.status}")
                    else:
                        text = await response.text()
                        try:
                            data = json.loads(text)
                            break
                        except json.JSONDecodeError:
                            logger.error(f"Game {game_pk}: Not valid JSON (first 200 chars) => {text[:200]}")
            except Exception as e:
                logger.error(f"Game {game_pk} exception: {e}")

            attempt += 1

        pbar.update(1)
        return data

    async def process_game(self, game_json):
        """
        Convert a single game JSON to rows -> append to CSV.
        """
        if not game_json:
            return

        scoreboard = game_json.get('scoreboard', {})
        game_id = scoreboard.get('gamePk')
        if not game_id:
            return

        boxscore = game_json.get('boxscore', {})
        if 'teams' not in boxscore:
            return

        # date
        game_date_str = game_json.get('gameDate', '01/01/1970')
        try:
            date_iso = datetime.strptime(game_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
        except:
            date_iso = "1970-01-01"

        # probable pitchers
        probable = scoreboard.get('probablePitchers', {})
        home_prob_pitcher_id = probable.get('home', {}).get('id')
        away_prob_pitcher_id = probable.get('away', {}).get('id')

        batter_rows = []
        pitcher_rows = []

        for side in ['home', 'away']:
            is_home = (side == 'home')
            team_info = boxscore['teams'][side].get('team', {})
            team_id = team_info.get('id', 0)

            players_dict = boxscore['teams'][side].get('players', {})
            prob_pitch_id = home_prob_pitcher_id if is_home else away_prob_pitcher_id

            for _, player_info in players_dict.items():
                row = {}
                row['game_id'] = game_id
                row['date'] = date_iso
                row['home_flag'] = 1 if is_home else 0
                row['team_id'] = team_id

                pid = player_info.get('person', {}).get('id', 0)
                row['player_id'] = pid

                # started?
                fielding_stats = player_info.get('stats', {}).get('fielding', {})
                row['started_flag'] = 1 if fielding_stats.get('gamesStarted', 0) > 0 else 0

                # probable pitcher
                row['probable_pitcher_flag'] = 1 if pid == prob_pitch_id else 0

                # position
                position_abb = player_info.get('position', {}).get('abbreviation', 'None')
                row['position'] = position_abb

                # batting order
                bo_str = player_info.get('battingOrder', '0')
                try:
                    row['batting_order_pos'] = int(bo_str)
                except:
                    row['batting_order_pos'] = 0

                # batting stats
                game_batting = player_info.get('stats', {}).get('batting', {})
                for sk in BATTING_STAT_MAP:
                    val = game_batting.get(sk, 0)
                    row[f"B_{sk}"] = float(val) if str(val).replace('.', '', 1).isdigit() else 0.0

                # pitching stats
                if position_abb == 'P':
                    game_pitching = player_info.get('stats', {}).get('pitching', {})
                    for sk in PITCHING_STAT_MAP:
                        val = game_pitching.get(sk, 0)
                        row[f"P_{sk}"] = float(val) if str(val).replace('.', '', 1).isdigit() else 0.0

                # fielding stats
                fld_map = FIELDING_STAT_MAP.get(position_abb.upper(), {})
                game_fielding = player_info.get('stats', {}).get('fielding', {})
                for fld_key in fld_map:
                    val = game_fielding.get(fld_key, 0)
                    row[f"F_{fld_key}"] = float(val) if str(val).isdigit() else 0.0

                # separate
                if position_abb == 'P':
                    pitcher_rows.append(row)
                else:
                    batter_rows.append(row)

        df_batters = pd.DataFrame(batter_rows)
        df_pitchers = pd.DataFrame(pitcher_rows)

        # write with lock
        async with self.csv_lock:
            if not df_batters.empty:
                df_batters.to_csv(
                    self.batters_csv,
                    mode='a',
                    index=False,
                    header=(not self.batters_initialized)
                )
                self.batters_initialized = True

            if not df_pitchers.empty:
                df_pitchers.to_csv(
                    self.pitchers_csv,
                    mode='a',
                    index=False,
                    header=(not self.pitchers_initialized)
                )
                self.pitchers_initialized = True

    async def fetch_process_one(self, session, game_pk, pbar):
        """
        Single game fetch+process
        """
        game_json = await self.fetch_one_game(session, game_pk, pbar)
        if game_json:
            await self.process_game(game_json)

    async def run_parallel_fetch(self, game_pks):
        """
        Given a list of game_pks for one month,
        fetch+process them asynchronously with concurrency limit.
        """
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        timeout = aiohttp.ClientTimeout(total=120)
        tasks = []
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            with tqdm_asyncio(total=len(game_pks), desc="Fetching/processing") as pbar:
                for pk in game_pks:
                    task = asyncio.ensure_future(self.fetch_process_one(session, pk, pbar))
                    tasks.append(task)
                await asyncio.gather(*tasks)

    def run(self):
        """
        Main driver:
          1) Remove old CSV
          2) For each month chunk, fetch game pks, do concurrency-limited fetch
        """
        # Remove old CSV if exist
        if os.path.exists(self.batters_csv):
            os.remove(self.batters_csv)
        if os.path.exists(self.pitchers_csv):
            os.remove(self.pitchers_csv)

        # Convert start/end to datetime
        sdt = datetime.strptime(self.start_date, "%m/%d/%Y")
        edt = datetime.strptime(self.end_date, "%m/%d/%Y")

        # For each month chunk, get game pks, then run async
        chunk_list = list(self.month_chunks(sdt, edt))
        for (chunk_start, chunk_end) in chunk_list:
            # 1) fetch game pks for chunk
            chunk_pks = self.fetch_game_pks_for_chunk(chunk_start, chunk_end)
            if not chunk_pks:
                logger.info(f"No games in {chunk_start.strftime('%Y-%m')}")
                continue

            # 2) async fetch+process for that chunk
            logger.info(f"Processing chunk {chunk_start.strftime('%Y-%m')} with {len(chunk_pks)} games...")
            try:
                asyncio.run(self.run_parallel_fetch(chunk_pks))
            except RuntimeError:
                # If we are in an environment with an active loop, we need a new loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.run_parallel_fetch(chunk_pks))

        logger.info("All month chunks completed.")


def main():
    generator = SavantVectorGenerator(
        start_date="01/01/2021",
        end_date="12/31/2024",
        batters_csv="batters_2021_2024.csv",
        pitchers_csv="pitchers_2021_2024.csv",
        concurrency=10
    )
    generator.run()


if __name__ == "__main__":
    main()
