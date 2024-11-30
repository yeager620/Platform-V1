import datetime
from fastapi import FastAPI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time


class OddsTrader:
    def __init__(self):
        self.options = Options()
        self.options.headless = True
        self.base_url = "https://www.oddstrader.com/mlb/?g=game&m=money"
        self.bet_types = ['401market-types-selector', '83market-types-selector', '402market-types-selector',
                          'mergedmarket-types-selector']

    def _init_browser(self):
        service = Service()
        return webdriver.Chrome(options=self.options)

    def scrape_bet_type(self, browser, bet_type):
        browser.find_element(By.ID, 'market-types-selector').click()
        time.sleep(1)  # Wait for the dropdown to be visible
        browser.find_element(By.ID, bet_type).click()
        time.sleep(2)  # Wait for the page to load the new data

        titles = browser.find_elements(By.CLASS_NAME, 'market-header')
        outcomes = browser.find_elements(By.CLASS_NAME, 'outcomes')
        bet_prices = browser.find_elements(By.CLASS_NAME, 'bet-price')

        now = datetime.datetime.now()

        o = []
        b = []
        t = []

        for i in range(len(outcomes)):
            o.append(outcomes[i].text)
            b.append(bet_prices[i].text)
            t.append(titles[0].text)

        df = pd.DataFrame({'outcomes': o, 'bet_price': b, 'title': t})
        df['date'] = now
        df['bet_price'] = df['bet_price'].replace('EVEN', '0')
        df['bet_price'] = df['bet_price'].astype(float)

        return df

    def scrape_all_bet_types(self):
        browser = self._init_browser()
        browser.get(self.base_url)
        browser.implicitly_wait(5)

        all_data = []

        for bet_type in self.bet_types:
            df = self.scrape_bet_type(browser, bet_type)
            all_data.append(df)

        browser.quit()

        return pd.concat(all_data, ignore_index=True)


app = FastAPI()
odds_trader = OddsTrader()


@app.get("/odds-trader/all-bet-types")
def get_all_bet_types_data():
    data = odds_trader.scrape_all_bet_types()
    return data.to_dict(orient='records')

# To run the FastAPI app, use the command:
# uvicorn your_script_name:app --reload
