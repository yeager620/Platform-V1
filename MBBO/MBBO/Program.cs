// Program.cs

using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using MBBO.LineData;
using MBBO.Savant;
using Newtonsoft.Json.Linq;

namespace MBBO
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Initialize the LiveGamelogsFetcher with default parameters (7 days ahead, 10 concurrent requests)
            LiveGamelogs fetcher = new LiveGamelogs();

            try
            {
                Console.WriteLine("Fetching game logs for the next 7 days...");
                List<JObject> gamelogs = await fetcher.GetGamelogsForNextGamesAsync();

                Console.WriteLine($"Fetched {gamelogs.Count} game logs.");

                // Example: Display the first game's log
                if (gamelogs.Count > 0)
                {
                    Console.WriteLine("Sample Game Log:");
                    Console.WriteLine(gamelogs[0].ToString());
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }
        }
    
    }
}