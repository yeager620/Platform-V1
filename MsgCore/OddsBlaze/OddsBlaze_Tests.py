import asyncio
from http.client import HTTPException

import pandas as pd
from fastapi import FastAPI

from OddsBlaze import OddsBlazeAPI

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