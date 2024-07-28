import asyncio

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from rich.jupyter import display


class OddsBlazeAPI:
    def __init__(self):
        self.base_url = "https://api.oddsblaze.com/v1/odds"
        self.api_key = "Xz6U6X270rSlWjaQ97zG"

    async def fetch_data(self, params: dict):
        params['key'] = self.api_key
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, params=params)
            if response.status_code != 200:
                response.raise_for_status()
            data = response.json()
            return data

    async def get_moneyline_data(self, league: str, region: str = 'us', price_format: str = 'american'):
        params = {
            'league': league,
            'market': 'moneyline',
            'region': region,
            'price': price_format,
        }
        data = await self.fetch_data(params)
        return self.format_data(data)

    @staticmethod
    def format_data(data):
        records = []
        for game in data['games']:
            for sportsbook in game['sportsbooks']:
                for odds in sportsbook['odds']:
                    if odds['market'].lower() == 'moneyline':  # Adjust this if 'moneyline' is not the correct key
                        record = {
                            'game_id': game['id'],
                            'commence_time': game['start'],
                            'home_team': game['teams']['home']['abbreviation'],
                            'away_team': game['teams']['away']['abbreviation'],
                            'sportsbook': sportsbook['name'],
                            'market': odds['market'],
                            'name': odds['name'],  # Adjust as per your need
                            'price': odds['price'],
                        }
                        records.append(record)
        return pd.DataFrame(records)


app = FastAPI()
odds_blaze = OddsBlazeAPI()


@app.get("/odds-blaze/moneyline")
async def get_moneyline_odds(client, league: str):
    try:
        data = await client.get_moneyline_data(league=league)
        return data.to_dict(orient='records')
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


# To run the FastAPI app, use the command:
# uvicorn your_script_name:app --reload

async def main():
    pd.set_option('display.max_colwidth', None)
    odds_blaze = OddsBlazeAPI()
    data = await odds_blaze.get_moneyline_data(league="mlb")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(data)


if __name__ == "__main__":
    asyncio.run(main())