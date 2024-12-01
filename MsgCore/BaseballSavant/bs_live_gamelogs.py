import requests
import datetime


class bs_live_gamelogs:
    def __init__(self):
        self.base_schedule_url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={date}&endDate={date}&fields=dates,date,games,gamePk"
        self.base_gamelog_url = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"

    @staticmethod
    def get_current_date():
        return datetime.datetime.now().strftime("%m/%d/%Y")

    def fetch_game_pks(self, date):
        url = self.base_schedule_url.format(date=date)
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

    def get_gamelogs_for_current_date(self):
        current_date = self.get_current_date()
        game_pks = self.fetch_game_pks(current_date)
        gamelogs = []
        for game_pk in game_pks:
            gamelog = self.fetch_gamelog(game_pk)
            gamelogs.append(gamelog)
        return gamelogs
