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


app = FastAPI()
odds_api = OddsAPI(api_key="960a7b445f8a6adf05292e951181bb07")


@app.get("/sports")
async def get_sports():
    return await odds_api.get_sports()


@app.get("/odds/{sport_key}")
async def get_odds(sport_key: str, region: str = 'us', market: str = 'h2h'):
    return await odds_api.get_odds(sport_key, region, market)


@app.get("/sport/{sport_key}")
async def get_sport(sport_key: str):
    return await odds_api.get_sport(sport_key)

# To run the FastAPI app, use the command:
# uvicorn your_script_name:app --reload
