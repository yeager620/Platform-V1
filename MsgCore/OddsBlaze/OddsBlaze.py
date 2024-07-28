import asyncio

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from rich.jupyter import display


class OddsBlazeAPI:
    def __init__(self):
        self.base_url = "https://api.oddsblaze.com/v1"
        self.api_key = "Xz6U6X270rSlWjaQ97zG"
        self.app = FastAPI()
        self.app.add_api_route("/odds-blaze/odds", self.get_odds, methods=["GET"])

    async def fetch_data(self, endpoint: str, params: dict):
        params['key'] = self.api_key
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/{endpoint}", params=params)
            if response.status_code != 200:
                response.raise_for_status()
            data = response.json()
            return data

    async def get_markets(self, league: str):
        params = {
            'league': league,
        }
        data = await self.fetch_data('markets', params)
        return data

    async def get_market_odds(self, league: str, market_id: str, region: str = 'us', price_format: str = 'american'):
        params = {
            'league': league,
            'market': market_id,
            'region': region,
            'price': price_format,
        }
        data = await self.fetch_data('odds', params)
        return data

    async def get_all_market_odds(self, league: str, region: str = 'us', price_format: str = 'american'):
        markets_data = await self.get_markets(league)
        all_records = []
        for league_info in markets_data.get('leagues', []):
            if league_info['id'] == league:
                for market in league_info.get('markets', []):
                    market_id = market['id']
                    odds_data = await self.get_market_odds(league, market_id, region, price_format)
                    records = self.format_data(odds_data)
                    all_records.extend(records)
        return pd.DataFrame(all_records)

    async def get_market_data(self, league: str, market: str, region: str = 'us', price_format: str = 'american'):
        params = {
            'league': league,
            'market': market,
            'region': region,
            'price': price_format,
        }
        data = await self.fetch_data(params)
        return self.format_data(data)

    async def get_game_moneyline_data(self, league: str, region: str = 'us', price_format: str = 'american'):
        params = {
            'league': league,
            'market': 'moneyline',
            'region': region,
            'price': price_format,
        }
        data = await self.fetch_data('odds', params)
        return self.format_data(data)

    @staticmethod
    def format_data(data):
        records = []
        for game in data.get('games', []):
            for sportsbook in game.get('sportsbooks', []):
                for odds in sportsbook.get('odds', []):
                    record = {
                        'game_id': game['id'],
                        'commence_time': game['start'],
                        'home_team': game['teams']['home']['abbreviation'],
                        'away_team': game['teams']['away']['abbreviation'],
                        'sportsbook': sportsbook['name'],
                        'market': odds.get('market', ''),
                        'name': odds.get('name', ''),
                        'price': odds.get('price', ''),
                    }
                    records.append(record)
        return pd.DataFrame(records)

    async def get_odds(self, league: str, market: str = None, region: str = 'us', price_format: str = 'american'):
        try:
            if market:
                data = await self.get_market_odds(league=league, market_id=market, region=region, price_format=price_format)
                records = self.format_data(data)
                return records
            else:
                records = await self.get_all_market_odds(league=league, region=region, price_format=price_format)
                return records
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    """def run(self):
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=8000)"""