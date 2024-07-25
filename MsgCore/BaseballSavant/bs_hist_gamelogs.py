import requests
import datetime
from dateutil.relativedelta import relativedelta


class bs_hist_gamelogs:
    def __init__(self):
        self.base_schedule_url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}&fields=dates,date,games,gamePk"
        self.base_gamelog_url = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"

    @staticmethod
    def get_past_date(years=0, months=0, days=0):
        current_date = datetime.datetime.now()
        past_date = current_date - relativedelta(years=years, months=months, days=days)
        return past_date.strftime("%m/%d/%Y")

    def fetch_game_pks(self, start_date, end_date):
        url = self.base_schedule_url.format(start_date=start_date, end_date=end_date)
        response = requests.get(url)
        data = response.json()
        game_pks = []
        if 'dates' in data:
            for date_info in data['dates']:
                for game in date_info['games']:
                    game_pks.append(game['gamePk'])
        return game_pks

    def fetch_gamelog(self, game_pk):
        url = self.base_gamelog_url.format(game_pk=game_pk)
        response = requests.get(url)
        return response.json()

    def get_gamelogs_for_period(self, years=0, months=0, days=0):
        start_date = self.get_past_date(years, months, days)
        end_date = datetime.datetime.now().strftime("%m/%d/%Y")
        game_pks = self.fetch_game_pks(start_date, end_date)
        gamelogs = []
        for game_pk in game_pks:
            gamelog = self.fetch_gamelog(game_pk)
            gamelogs.append(gamelog)
        return gamelogs
