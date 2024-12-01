import httpx
from fastapi import FastAPI, HTTPException


class OddsAPI:
    def __init__(self, api_key: str):
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.api_key = api_key

    async def fetch_data(self, endpoint: str, params: dict = None):
        params = params or {}
        params['apiKey'] = self.api_key
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/{endpoint}", params=params)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()

    async def get_sports(self):
        return await self.fetch_data("")

    async def get_odds(self, sport_key: str, region: str = 'us', market: str = 'h2h'):
        return await self.fetch_data(f"{sport_key}/odds", params={'region': region, 'market': market})

    async def get_sport(self, sport_key: str):
        return await self.fetch_data(sport_key)


class GameMapper:
    def __init__(self):
        self.baseball_savant_url = "https://baseballsavant.mlb.com/gf"
        self.team_name_mapping = {
            "Chicago Cubs": "Chicago Cubs",
            "Los Angeles Dodgers": "Los Angeles Dodgers",
            # Add more mappings as needed
        }

    async def get_game_details(self, game_pk: int):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.baseball_savant_url}?game_pk={game_pk}")
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            game_data = response.json()
            home_team = game_data['scoreboard']['teams.json']['home']['team']['name']
            away_team = game_data['scoreboard']['teams.json']['away']['team']['name']
            commence_time = game_data['gameDate']

            sport_id = "baseball_mlb"  # Hardcoded for MLB games

            return {
                "sport_id": sport_id,
                "home_team": self.map_team_name(home_team),
                "away_team": self.map_team_name(away_team),
                "commence_time": commence_time
            }

    def map_team_name(self, team_name):
        return self.team_name_mapping.get(team_name, team_name)


app = FastAPI()
odds_api = OddsAPI(api_key="your_api_key_here")
game_mapper = GameMapper()


@app.get("/sports")
async def get_sports():
    return await odds_api.get_sports()


@app.get("/odds/{game_pk}")
async def get_odds(game_pk: int, region: str = 'us', market: str = 'h2h'):
    game_details = await game_mapper.get_game_details(game_pk)
    sport_key = game_details['sport_id']
    odds_data = await odds_api.get_odds(sport_key, region, market)

    filtered_odds = []
    for event in odds_data:
        if event['home_team'] == game_details['home_team'] and event['away_team'] == game_details['away_team'] and \
                event['commence_time'] == game_details['commence_time']:
            filtered_odds.append(event)

    if not filtered_odds:
        raise HTTPException(status_code=404, detail="No matching odds found")

    return filtered_odds


@app.get("/sport/{game_pk}")
async def get_sport(game_pk: int):
    game_details = await game_mapper.get_game_details(game_pk)
    sport_key = game_details['sport_id']
    return await odds_api.get_sport(sport_key)

# To run the FastAPI app, use the command:
# uvicorn your_script_name:app --reload