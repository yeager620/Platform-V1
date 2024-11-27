import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import datetime


class SportsbookReviewScraper:
    def __init__(self, start_date, end_date):
        self.base_url = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/"
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = self.generate_date_range()
        self.session = self.create_session_with_retries()  # Initialize session with retries

    def generate_date_range(self):
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        return [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]

    def create_session_with_retries(self):
        """
        Creates a requests Session with a retry strategy to handle 502, 503, and 504 HTTP errors.
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=5,  # Total number of retries
            status_forcelist=[502, 503, 504],  # HTTP status codes to retry on
            method_whitelist=["HEAD", "GET", "OPTIONS"],  # Methods to retry
            backoff_factor=1  # Exponential backoff factor (1, 2, 4, 8, 16 seconds)
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch_page(self, date):
        formatted_date = date.strftime("%Y-%m-%d")
        url = f"{self.base_url}?date={formatted_date}"
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/58.0.3029.110 Safari/537.3'
            )
        }
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            return response.text
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # HTTP error
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")  # Network problem
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")  # Request timed out
        except requests.exceptions.RequestException as req_err:
            print(f"An error occurred: {req_err}")  # Other requests exceptions
        return None  # Return None if an exception occurs

    def parse_page(self, html, date):
        if not html:
            return []  # Return empty list if no HTML content
        soup = BeautifulSoup(html, 'html.parser')
        games = soup.find_all('div', class_='GameRows_eventMarketGridContainer__GuplK')
        records = []
        for game in games:
            try:
                game_time = game.find('span', class_='fs-9').text.strip()
                game_url = game.find('a', class_='fs-9 py-2 pe-1 text-primary')['href']
                teams = game.find_all('b')
                team_1 = teams[0].text.strip()
                team_2 = teams[1].text.strip()

                # Extracting wager percentages
                wager_data = game.find('div', class_='GameRows_consensusColumn__AOd1q')
                wager_percentages = wager_data.find_all('span', class_='opener') if wager_data else []
                team_1_wager = wager_percentages[0].text.strip() if len(wager_percentages) > 0 else None
                team_2_wager = wager_percentages[1].text.strip() if len(wager_percentages) > 1 else None

                sportsbook_data = game.find_all('div', class_='OddsCells_numbersContainer__6V_XO')
                for sportsbook in sportsbook_data:
                    sportsbook_name = sportsbook.find('a')['data-aatracker'].split(' - ')[-1].strip()
                    odds_lines = sportsbook.find_all('div', class_='OddsCells_oddsNumber__u3rsp')

                    if len(odds_lines) >= 2:
                        team_1_odds = odds_lines[0].find_all('span', class_='fs-9')[-1].text.strip()
                        team_2_odds = odds_lines[1].find_all('span', class_='fs-9')[-1].text.strip()

                        # Record for team 1
                        record_team_1 = {
                            'date': date.strftime("%Y-%m-%d"),
                            'team': team_1,
                            'opponent': team_2,
                            'sportsbook': sportsbook_name,
                            'odds': team_1_odds,
                            'wager_percentage': team_1_wager,
                            'game_time': game_time,
                            'game_url': game_url
                        }
                        records.append(record_team_1)

                        # Record for team 2
                        record_team_2 = {
                            'date': date.strftime("%Y-%m-%d"),
                            'team': team_2,
                            'opponent': team_1,
                            'sportsbook': sportsbook_name,
                            'odds': team_2_odds,
                            'wager_percentage': team_2_wager,
                            'game_time': game_time,
                            'game_url': game_url
                        }
                        records.append(record_team_2)
            except Exception as e:
                print(f"Error parsing game data: {e}")
                continue
        return records

    def scrape(self):
        all_records = []
        for date in self.date_range:
            print(f"Scraping data for date: {date.strftime('%Y-%m-%d')}")
            html = self.fetch_page(date)
            records = self.parse_page(html, date)
            all_records.extend(records)
        return pd.DataFrame(all_records)
