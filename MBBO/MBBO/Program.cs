// Program.cs

using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using MBBO.LineData;

namespace MBBO
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Set up logging
            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder
                    .AddConsole()
                    .SetMinimumLevel(LogLevel.Information);
            });
            ILogger<SportsbookReviewScraper> logger = loggerFactory.CreateLogger<SportsbookReviewScraper>();

            // Validate input arguments
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: SportsbookReviewScraper <start_date> <end_date>");
                Console.WriteLine("Date format: YYYY-MM-DD");
                return;
            }

            string startDate = args[0];
            string endDate = args[1];

            // Initialize scraper
            var scraper = new SportsbookReviewScraper(startDate, endDate, logger);

            try
            {
                Console.WriteLine("Starting scraping process...");
                var data = await scraper.ScrapeAsync();

                Console.WriteLine($"Scraped {data.Count} records.");

                // Optionally, export to CSV
                string csvPath = $"SportsbookData_{startDate}_to_{endDate}.csv";
                scraper.ExportToCsv(data, csvPath);
                Console.WriteLine($"Data exported to {csvPath}");
            }
            catch (Exception ex)
            {
                logger.LogError($"An error occurred during scraping: {ex.Message}");
            }
        }
    }
}