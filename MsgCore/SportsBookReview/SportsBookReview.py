import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime


class SportsbookReviewScraper:
    def __init__(self, start_date, end_date):
        self.base_url = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/"
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = self.generate_date_range()

    def generate_date_range(self):
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        return [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]

    def fetch_page(self, date):
        formatted_date = date.strftime("%Y-%m-%d")
        url = f"{self.base_url}?date={formatted_date}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def parse_page(self, html, date):
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
                wager_data = game.find_all('div', class_='GameRows_consensusColumn__AOd1q')[0]
                wager_percentages = wager_data.find_all('span', class_='opener')
                team_1_wager = wager_percentages[0].text.strip() if len(wager_percentages) > 0 else None
                team_2_wager = wager_percentages[1].text.strip() if len(wager_percentages) > 1 else None

                sportsbook_data = game.find_all('div', class_='OddsCells_numbersContainer__6V_XO')
                for sportsbook in sportsbook_data:
                    sportsbook_name = sportsbook.find('a')['data-aatracker'].split(' - ')[-1].strip()
                    odds_lines = sportsbook.find_all('div', class_='OddsCells_oddsNumber__u3rsp')

                    if len(odds_lines) >= 2:
                        team_1_odds = odds_lines[0].find_all('span', class_='fs-9')[-1].text.strip()
                        team_2_odds = odds_lines[1].find_all('span', class_='fs-9')[-1].text.strip()

                        # Extracting opener (if available)
                        opener_tag = sportsbook.find_previous('span', class_='opener')
                        opener = opener_tag.text.strip() if opener_tag else None

                        # Record for team 1
                        record_team_1 = {
                            'date': date.strftime("%Y-%m-%d"),
                            'team': team_1,
                            'opponent': team_2,
                            'sportsbook': sportsbook_name,
                            'odds': team_1_odds,
                            'wager_percentage': team_1_wager,
                            'opener': opener,
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
                            'opener': opener,
                            'game_time': game_time,
                            'game_url': game_url
                        }
                        records.append(record_team_2)
            except Exception as e:
                print(f"Error parsing game data: {e}")
        return records

    def scrape(self):
        all_records = []
        for date in self.date_range:
            print(f"Scraping data for date: {date.strftime('%Y-%m-%d')}")
            html = self.fetch_page(date)
            records = self.parse_page(html, date)
            all_records.extend(records)
        return pd.DataFrame(all_records)


if __name__ == "__main__":
    scraper = SportsbookReviewScraper(start_date="2023-06-10", end_date="2023-06-15")
    df = scraper.scrape()
    print(df.head())
    df.to_csv("sportsbook_data.csv", index=False)
