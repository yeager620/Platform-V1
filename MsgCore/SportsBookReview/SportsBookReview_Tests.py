from SportsBookReview import SportsbookReviewScraper

if __name__ == "__main__":
    scraper = SportsbookReviewScraper(start_date="2023-06-01", end_date="2023-06-02")
    df = scraper.scrape()
    print(df.head())
    df.to_csv("sportsbook_data.csv", index=False)