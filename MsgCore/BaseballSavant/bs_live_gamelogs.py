import aiohttp
import asyncio
import datetime
from typing import List, Dict, Any, Set


class LiveGamelogsFetcher:
    def __init__(self, days_ahead: int = 7, max_concurrent_requests: int = 10):
        """
        Initialize the fetcher.

        :param days_ahead: Number of days ahead to look for upcoming games.
        :param max_concurrent_requests: Maximum number of concurrent HTTP requests.
        """
        self.base_schedule_url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}&fields=dates,date,games,gamePk,teams,team"
        self.base_gamelog_url = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
        self.days_ahead = days_ahead
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)  # To limit concurrent requests

    @staticmethod
    def get_date_str(date_obj: datetime.date) -> str:
        """
        Convert a date object to a string in 'MM/DD/YYYY' format.

        :param date_obj: datetime.date object.
        :return: Formatted date string.
        """
        return date_obj.strftime("%m/%d/%Y")

    def get_upcoming_dates(self) -> List[str]:
        """
        Generate a list of upcoming dates as strings.

        :return: List of date strings.
        """
        today = datetime.date.today()
        return [self.get_date_str(today + datetime.timedelta(days=i)) for i in range(self.days_ahead)]

    async def fetch_json(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """
        Asynchronously fetch JSON data from a given URL.

        :param session: aiohttp ClientSession.
        :param url: URL to fetch data from.
        :return: Parsed JSON data.
        """
        async with self.semaphore:  # Ensure we don't exceed the max concurrent requests
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                print(f"HTTP error for URL {url}: {e}")
                return {}
            except Exception as e:
                print(f"Unexpected error for URL {url}: {e}")
                return {}

    async def fetch_game_log(self, session: aiohttp.ClientSession, game_pk: int) -> Dict[str, Any]:
        """
        Asynchronously fetch a single game log.

        :param session: aiohttp ClientSession.
        :param game_pk: GamePk integer.
        :return: Game log data.
        """
        url = self.base_gamelog_url.format(game_pk=game_pk)
        return await self.fetch_json(session, url)

    async def fetch_schedule(self, session: aiohttp.ClientSession, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Asynchronously fetch the schedule between start_date and end_date.

        :param session: aiohttp ClientSession.
        :param start_date: Start date string in 'MM/DD/YYYY'.
        :param end_date: End date string in 'MM/DD/YYYY'.
        :return: Schedule data.
        """
        url = self.base_schedule_url.format(start_date=start_date, end_date=end_date)
        return await self.fetch_json(session, url)

    async def get_next_games(self) -> List[int]:
        """
        Identify the next game for each team and return unique gamePk's.

        :return: List of unique gamePk integers.
        """
        upcoming_dates = self.get_upcoming_dates()
        team_next_game: Dict[int, int] = {}  # team_id -> gamePk

        async with aiohttp.ClientSession() as session:
            # Fetch schedules for all upcoming dates
            tasks = [self.fetch_schedule(session, date, date) for date in upcoming_dates]
            schedules = await asyncio.gather(*tasks)

            for schedule in schedules:
                if 'dates' not in schedule:
                    continue
                for date_info in schedule['dates']:
                    for game in date_info.get('games', []):
                        game_pk = game.get('gamePk')
                        teams = game.get('teams', {})
                        home_team = teams.get('home', {}).get('team', {})
                        away_team = teams.get('away', {}).get('team', {})
                        home_team_id = home_team.get('id')
                        away_team_id = away_team.get('id')

                        # Assign the gamePk as the next game for both teams if not already assigned
                        for team_id in [home_team_id, away_team_id]:
                            if team_id not in team_next_game:
                                team_next_game[team_id] = game_pk

                            # Early exit if all teams have been assigned
                            if len(team_next_game) >= 30:
                                break
                        if len(team_next_game) >= 30:
                            break
                if len(team_next_game) >= 30:
                    break

        # Extract unique gamePk's from the next games of all teams
        unique_game_pks = list(set(team_next_game.values()))
        print(f"Identified {len(unique_game_pks)} unique upcoming games.")
        return unique_game_pks

    async def fetch_game_logs(self, game_pks: List[int]) -> List[Dict[str, Any]]:
        """
        Asynchronously fetch game logs for a list of game Pk's.

        :param game_pks: List of gamePk integers.
        :return: List of game log data.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_game_log(session, game_pk) for game_pk in game_pks]
            return await asyncio.gather(*tasks)

    async def get_gamelogs_for_next_games(self) -> List[Dict[str, Any]]:
        """
        Fetch game logs for the next game of each team.

        :return: List of game log data.
        """
        next_game_pks = await self.get_next_games()
        gamelogs = await self.fetch_game_logs(next_game_pks)
        return gamelogs

    def get_gamelogs(self) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper to fetch game logs.

        :return: List of game log data.
        """
        return asyncio.run(self.get_gamelogs_for_next_games())
