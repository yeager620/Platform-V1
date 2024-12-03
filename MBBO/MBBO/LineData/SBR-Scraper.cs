using HtmlAgilityPack;
using Polly;
using Polly.Retry;
using Microsoft.Extensions.Logging;

namespace MBBO.LineData
{
    public class SportsbookReviewScraper
    {
        private readonly string _baseUrl = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/";
        private readonly DateTime _startDate;
        private readonly DateTime _endDate;
        private readonly List<DateTime> _dateRange;
        private readonly HttpClient _httpClient;
        private readonly ILogger<SportsbookReviewScraper> _logger;
        private readonly AsyncRetryPolicy<HttpResponseMessage> _retryPolicy;

        public SportsbookReviewScraper(string startDate, string endDate, ILogger<SportsbookReviewScraper> logger = null)
        {
            _startDate = DateTime.ParseExact(startDate, "yyyy-MM-dd", null);
            _endDate = DateTime.ParseExact(endDate, "yyyy-MM-dd", null);
            _dateRange = GenerateDateRange(_startDate, _endDate);
            _httpClient = new HttpClient();
            _logger = logger;

            // Define a retry policy with exponential backoff
            _retryPolicy = Policy
                .HandleResult<HttpResponseMessage>(r => 
                    r.StatusCode == System.Net.HttpStatusCode.BadGateway ||
                    r.StatusCode == System.Net.HttpStatusCode.ServiceUnavailable ||
                    r.StatusCode == System.Net.HttpStatusCode.GatewayTimeout)
                .WaitAndRetryAsync(
                    retryCount: 5,
                    sleepDurationProvider: attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt)),
                    onRetry: (outcome, timespan, retryAttempt, context) =>
                    {
                        _logger?.LogWarning($"Request failed with {outcome.Result.StatusCode}. Waiting {timespan} before retry {retryAttempt}.");
                    });
        }

        private List<DateTime> GenerateDateRange(DateTime start, DateTime end)
        {
            var range = new List<DateTime>();
            for (var dt = start; dt <= end; dt = dt.AddDays(1))
            {
                range.Add(dt);
            }
            return range;
        }

        private async Task<(DateTime date, string html)> FetchPageAsync(DateTime date, CancellationToken cancellationToken = default)
        {
            string formattedDate = date.ToString("yyyy-MM-dd");
            string url = $"{_baseUrl}?date={formattedDate}";
            var headers = new Dictionary<string, string>
            {
                { "User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
                                "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3" }
            };

            using (var request = new HttpRequestMessage(HttpMethod.Get, url))
            {
                foreach (var header in headers)
                {
                    request.Headers.Add(header.Key, header.Value);
                }

                HttpResponseMessage response = null;
                try
                {
                    response = await _retryPolicy.ExecuteAsync(() => _httpClient.SendAsync(request, cancellationToken));
                }
                catch (Exception ex)
                {
                    _logger?.LogError($"Exception fetching page for date {formattedDate}: {ex.Message}");
                    return (date, null);
                }

                if (response.IsSuccessStatusCode)
                {
                    string html = await response.Content.ReadAsStringAsync();
                    return (date, html);
                }
                else
                {
                    _logger?.LogError($"Failed to fetch page for date {formattedDate}. Status Code: {response.StatusCode}");
                    return (date, null);
                }
            }
        }

        private List<GameData> ParsePage(string html, DateTime date)
        {
            var records = new List<GameData>();

            if (string.IsNullOrEmpty(html))
            {
                return records;
            }

            var doc = new HtmlDocument();
            doc.LoadHtml(html);

            var games = doc.DocumentNode.SelectNodes("//div[contains(@class, 'GameRows_eventMarketGridContainer__GuplK')]");

            if (games == null)
            {
                _logger?.LogWarning($"No games found on date {date.ToString("yyyy-MM-dd")}");
                return records;
            }

            foreach (var game in games)
            {
                try
                {
                    // Extract game time
                    var gameTimeNode = game.SelectSingleNode(".//span[contains(@class, 'fs-9')]");
                    string gameTime = gameTimeNode?.InnerText.Trim() ?? string.Empty;

                    // Extract game URL
                    var gameUrlNode = game.SelectSingleNode(".//a[contains(@class, 'fs-9 py-2 pe-1 text-primary')]");
                    string gameUrl = gameUrlNode?.GetAttributeValue("href", string.Empty) ?? string.Empty;

                    // Extract teams
                    var teams = game.SelectNodes(".//b");
                    string team1 = teams?.Count > 0 ? teams[0].InnerText.Trim() : string.Empty;
                    string team2 = teams?.Count > 1 ? teams[1].InnerText.Trim() : string.Empty;

                    // Extract wager percentages
                    var wagerDataNode = game.SelectSingleNode(".//div[contains(@class, 'GameRows_consensusColumn__AOd1q')]");
                    var wagerPercentages = wagerDataNode?.SelectNodes(".//span[contains(@class, 'opener')]");
                    string team1Wager = wagerPercentages?.Count > 0 ? wagerPercentages[0].InnerText.Trim() : string.Empty;
                    string team2Wager = wagerPercentages?.Count > 1 ? wagerPercentages[1].InnerText.Trim() : string.Empty;

                    // Extract sportsbook data
                    var sportsbookDataNodes = game.SelectNodes(".//div[contains(@class, 'OddsCells_numbersContainer__6V_XO')]");
                    if (sportsbookDataNodes != null)
                    {
                        foreach (var sportsbook in sportsbookDataNodes)
                        {
                            var sportsbookLink = sportsbook.SelectSingleNode(".//a");
                            string sportsbookName = sportsbookLink != null 
                                ? sportsbookLink.GetAttributeValue("data-aatracker", "").Split(" - ").Last().Trim()
                                : string.Empty;

                            var oddsLines = sportsbook.SelectNodes(".//div[contains(@class, 'OddsCells_oddsNumber__u3rsp')]");
                            if (oddsLines != null && oddsLines.Count >= 2)
                            {
                                var team1OddsSpan = oddsLines[0].SelectNodes(".//span[contains(@class, 'fs-9')]");
                                var team2OddsSpan = oddsLines[1].SelectNodes(".//span[contains(@class, 'fs-9')]");
                                string team1Odds = team1OddsSpan != null && team1OddsSpan.Count > 0 
                                    ? team1OddsSpan.Last().InnerText.Trim() 
                                    : string.Empty;
                                string team2Odds = team2OddsSpan != null && team2OddsSpan.Count > 0 
                                    ? team2OddsSpan.Last().InnerText.Trim() 
                                    : string.Empty;

                                // Record for Team 1
                                var recordTeam1 = new GameData
                                {
                                    Date = date.ToString("yyyy-MM-dd"),
                                    Team = team1,
                                    Opponent = team2,
                                    Sportsbook = sportsbookName,
                                    Odds = team1Odds,
                                    WagerPercentage = team1Wager,
                                    GameTime = gameTime,
                                    GameUrl = gameUrl
                                };
                                records.Add(recordTeam1);

                                // Record for Team 2
                                var recordTeam2 = new GameData
                                {
                                    Date = date.ToString("yyyy-MM-dd"),
                                    Team = team2,
                                    Opponent = team1,
                                    Sportsbook = sportsbookName,
                                    Odds = team2Odds,
                                    WagerPercentage = team2Wager,
                                    GameTime = gameTime,
                                    GameUrl = gameUrl
                                };
                                records.Add(recordTeam2);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger?.LogError($"Error parsing game data on {date.ToString("yyyy-MM-dd")}: {ex.Message}");
                    continue;
                }
            }

            return records;
        }

        public async Task<List<GameData>> ScrapeAsync(CancellationToken cancellationToken = default)
        {
            var allRecords = new List<GameData>();

            using (var semaphore = new SemaphoreSlim(10)) // Limit concurrent connections
            {
                var tasks = _dateRange.Select(async date =>
                {
                    await semaphore.WaitAsync(cancellationToken);
                    try
                    {
                        var (fetchedDate, html) = await FetchPageAsync(date, cancellationToken);
                        var records = ParsePage(html, fetchedDate);
                        return records;
                    }
                    finally
                    {
                        semaphore.Release();
                    }
                }).ToList();

                var results = await Task.WhenAll(tasks);
                allRecords = results.SelectMany(r => r).ToList();
            }

            return allRecords;
        }

        // Optional: Method to export data to CSV using CsvHelper
        public void ExportToCsv(List<GameData> data, string filePath)
        {
            using (var writer = new System.IO.StreamWriter(filePath))
            using (var csv = new CsvHelper.CsvWriter(writer, System.Globalization.CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(data);
            }
        }
    }
}
