import asyncio
import aiohttp
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientError
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio


class SportsbookReviewScraper:
    def __init__(self, start_date, end_date):
        self.base_url = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/"
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = self.generate_date_range()

    def generate_date_range(self):
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        return [start + timedelta(days=x) for x in range((end - start).days + 1)]

    async def fetch_page(self, session, date, max_retries=5):
        """
        Asynchronously fetches the HTML content of a page for a given date with retries.

        Parameters:
            session (ClientSession): The aiohttp session object.
            date (datetime): The date for which to fetch the page.
            max_retries (int): Maximum number of retries for failed requests.

        Returns:
            tuple: (date, html content) if successful, else (date, None).
        """
        formatted_date = date.strftime("%Y-%m-%d")
        url = f"{self.base_url}?date={formatted_date}"
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            )
        }
        retries = 0
        while retries < max_retries:
            try:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        return date, html
                    elif response.status in [502, 503, 504]:
                        retries += 1
                        await asyncio.sleep(2 ** retries)  # Exponential backoff
                    else:
                        print(f"HTTP error {response.status} for date {formatted_date}")
                        return date, None
            except asyncio.TimeoutError:
                retries += 1
                print(f"Timeout error on {formatted_date}, retry {retries}/{max_retries}")
                await asyncio.sleep(2 ** retries)
            except ClientError as e:
                retries += 1
                print(f"Client error on {formatted_date}: {e}, retry {retries}/{max_retries}")
                await asyncio.sleep(2 ** retries)
        print(f"Failed to fetch page for date {formatted_date} after {max_retries} retries")
        return date, None

    async def fetch_all_pages(self):
        """
        Asynchronously fetches all pages for the date range.

        Returns:
            list: List of tuples containing (date, html content).
        """
        pages = []
        connector = aiohttp.TCPConnector(limit=10)  # Limit concurrent connections
        timeout = aiohttp.ClientTimeout(total=60)    # Total timeout for requests

        async with ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self.fetch_page(session, date) for date in self.date_range]
            for f in tqdm_asyncio.as_completed(tasks, desc="Fetching pages", total=len(tasks)):
                date, html = await f
                pages.append((date, html))
        return pages

    def parse_page(self, html, date):
        """
        Parses the HTML content of a page to extract game data.

        Parameters:
            html (str): The HTML content of the page.
            date (datetime): The date corresponding to the page.

        Returns:
            list: List of dictionaries containing game data.
        """
        if not html:
            return []  # Return empty list if no HTML content
        soup = BeautifulSoup(html, 'html.parser')
        games = soup.find_all('div', class_='GameRows_eventMarketGridContainer__GuplK')
        records = []
        for game in games:
            try:
                game_time = game.find('span', class_='fs-9').text.strip()
                game_url_tag = game.find('a', class_='fs-9 py-2 pe-1 text-primary')
                game_url = game_url_tag['href'] if game_url_tag else None
                teams = game.find_all('b')
                team_1 = teams[0].text.strip() if len(teams) > 0 else None
                team_2 = teams[1].text.strip() if len(teams) > 1 else None

                # Extracting wager percentages
                wager_data = game.find('div', class_='GameRows_consensusColumn__AOd1q')
                wager_percentages = wager_data.find_all('span', class_='opener') if wager_data else []
                team_1_wager = wager_percentages[0].text.strip() if len(wager_percentages) > 0 else None
                team_2_wager = wager_percentages[1].text.strip() if len(wager_percentages) > 1 else None

                sportsbook_data = game.find_all('div', class_='OddsCells_numbersContainer__6V_XO')
                for sportsbook in sportsbook_data:
                    sportsbook_link = sportsbook.find('a')
                    sportsbook_name = sportsbook_link['data-aatracker'].split(' - ')[-1].strip() if sportsbook_link else None
                    odds_lines = sportsbook.find_all('div', class_='OddsCells_oddsNumber__u3rsp')

                    if len(odds_lines) >= 2:
                        team_1_odds_span = odds_lines[0].find_all('span', class_='fs-9')
                        team_2_odds_span = odds_lines[1].find_all('span', class_='fs-9')
                        team_1_odds = team_1_odds_span[-1].text.strip() if team_1_odds_span else None
                        team_2_odds = team_2_odds_span[-1].text.strip() if team_2_odds_span else None

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
                print(f"Error parsing game data on {date.strftime('%Y-%m-%d')}: {e}")
                continue
        return records

    def scrape(self):
        """
        Main method to scrape data and return a DataFrame.

        Returns:
            DataFrame: Pandas DataFrame containing all scraped data.
        """
        all_records = []
        # Handle event loop creation based on environment
        try:
            date_html_list = asyncio.run(self.fetch_all_pages())
        except RuntimeError:
            # If an event loop is already running (e.g., in Jupyter), create a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            date_html_list = loop.run_until_complete(self.fetch_all_pages())

        # Parse pages and collect records
        for date, html in date_html_list:
            records = self.parse_page(html, date)
            all_records.extend(records)
        return pd.DataFrame(all_records)
