import asyncio
from http.client import HTTPException
import pandas as pd
from fastapi import FastAPI

from OddsBlaze import OddsBlazeAPI

app = FastAPI()
odds_blaze = OddsBlazeAPI()


async def main():
    pd.set_option('display.max_colwidth', None)
    data = await odds_blaze.get_odds(league="mlb", market="mlb:moneyline")
    # data = await odds_blaze.get_all_market_odds(league="mlb")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(data)


if __name__ == "__main__":
    asyncio.run(main())
