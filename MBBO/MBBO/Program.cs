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
            // Setup Logger
            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.AddSimpleConsole(options =>
                {
                    options.SingleLine = true;
                    options.TimestampFormat = "hh:mm:ss ";
                });
                builder.SetMinimumLevel(LogLevel.Information);
            });
            ILogger<VectorGenerator> logger = loggerFactory.CreateLogger<VectorGenerator>();

            // Initialize VectorGenerator
            var vectorGenerator = new VectorGenerator(logger);

            // Specify the Game Pk
            int gamePk = 487628; // Replace with a valid GamePk

            // Generate Game Vector
            GameVector gameVector = await vectorGenerator.GenerateGameVectorAsync(gamePk);

            if (gameVector != null)
            {
                Console.WriteLine($"Game Date: {gameVector.GameDate}");
                Console.WriteLine($"Home Team: {gameVector.HomeTeamAbbr}");
                Console.WriteLine($"Away Team: {gameVector.AwayTeamAbbr}");
                Console.WriteLine($"Park ID: {gameVector.ParkId}");
                Console.WriteLine($"Home Win: {gameVector.HomeWin}");

                Console.WriteLine("\nHome Team Batting Stats:");
                foreach (var stat in gameVector.HomeBattingStats)
                {
                    Console.WriteLine($"{stat.Key}: {stat.Value}");
                }

                Console.WriteLine("\nHome Team Pitching Stats:");
                foreach (var stat in gameVector.HomePitchingStats)
                {
                    Console.WriteLine($"{stat.Key}: {stat.Value}");
                }

                Console.WriteLine("\nHome Team Fielding Stats:");
                foreach (var stat in gameVector.HomeFieldingStats)
                {
                    Console.WriteLine($"{stat.Key}: {stat.Value}");
                }

                Console.WriteLine("\nHome Team Bullpen Stats:");
                foreach (var stat in gameVector.HomeBullpenStats)
                {
                    Console.WriteLine($"{stat.Key}: {stat.Value}");
                }

                Console.WriteLine("\nAway Team Batting Stats:");
                foreach (var stat in gameVector.AwayBattingStats)
                {
                    Console.WriteLine($"{stat.Key}: {stat.Value}");
                }

                Console.WriteLine("\nAway Team Pitching Stats:");
                foreach (var stat in gameVector.AwayPitchingStats)
                {
                    Console.WriteLine($"{stat.Key}: {stat.Value}");
                }

                Console.WriteLine("\nAway Team Fielding Stats:");
                foreach (var stat in gameVector.AwayFieldingStats)
                {
                    Console.WriteLine($"{stat.Key}: {stat.Value}");
                }

                Console.WriteLine("\nAway Team Bullpen Stats:");
                foreach (var stat in gameVector.AwayBullpenStats)
                {
                    Console.WriteLine($"{stat.Key}: {stat.Value}");
                }
            }
            else
            {
                Console.WriteLine("Failed to generate game vector.");
            }
        }
    
    }
}