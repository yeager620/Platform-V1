import pandas as pd
from SportsBookReview import SportsbookReviewScraper

if __name__ == "__main__":
    scraper = SportsbookReviewScraper(start_date="2022-04-30", end_date="2022-04-30")
    data = scraper.scrape()
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(data)