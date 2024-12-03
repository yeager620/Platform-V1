using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Globalization;
using Microsoft.Extensions.Logging;

namespace MBBO.Savant
{
    /// <summary>
    /// Represents the game vector with aggregated statistics for both home and away teams.
    /// </summary>
    public class GameVector
    {
        // Common Game Information
        public string GameDate { get; set; }
        public int GamePk { get; set; }
        public string HomeTeamAbbr { get; set; }
        public string AwayTeamAbbr { get; set; }
        public int ParkId { get; set; }

        // Home Team Stats
        public Dictionary<BattingStat, double> HomeBattingStats { get; set; } = new Dictionary<BattingStat, double>();
        public Dictionary<PitchingStat, double> HomePitchingStats { get; set; } = new Dictionary<PitchingStat, double>();
        public Dictionary<FieldingStat, double> HomeFieldingStats { get; set; } = new Dictionary<FieldingStat, double>();
        public Dictionary<PitchingStat, double> HomeBullpenStats { get; set; } = new Dictionary<PitchingStat, double>();

        // Away Team Stats
        public Dictionary<BattingStat, double> AwayBattingStats { get; set; } = new Dictionary<BattingStat, double>();
        public Dictionary<PitchingStat, double> AwayPitchingStats { get; set; } = new Dictionary<PitchingStat, double>();
        public Dictionary<FieldingStat, double> AwayFieldingStats { get; set; } = new Dictionary<FieldingStat, double>();
        public Dictionary<PitchingStat, double> AwayBullpenStats { get; set; } = new Dictionary<PitchingStat, double>();

        // Outcome
        public int HomeWin { get; set; }
    }

    /// <summary>
    /// Enumeration for Batting Statistics.
    /// </summary>
    public enum BattingStat
    {
        GamesPlayed,
        PlateAppearances,
        AtBats,
        Runs,
        Hits,
        TotalBases,
        Doubles,
        Triples,
        HomeRuns,
        RBI,
        BaseOnBalls,
        IntentionalWalks,
        StrikeOuts,
        GroundIntoDoublePlay,
        GroundIntoTriplePlay,
        HitByPitch,
        CaughtStealing,
        StolenBases,
        LeftOnBase,
        SacBunts,
        SacFlies,
        CatchersInterference,
        Pickoffs,
        FlyOuts,
        GroundOuts
    }

    /// <summary>
    /// Enumeration for Pitching Statistics.
    /// </summary>
    public enum PitchingStat
    {
        GamesPlayed,
        GamesStarted,
        GroundOuts,
        AirOuts,
        Runs,
        Doubles,
        Triples,
        HomeRuns,
        StrikeOuts,
        BaseOnBalls,
        IntentionalWalks,
        Hits,
        HitByPitch,
        AtBats,
        CaughtStealing,
        StolenBases,
        NumberOfPitches,
        InningsPitched,
        Wins,
        Losses,
        Saves,
        SaveOpportunities,
        Holds,
        BlownSaves,
        EarnedRuns,
        BattersFaced,
        Outs,
        CompleteGames,
        Shutouts,
        PitchesThrown,
        Balls,
        Strikes,
        HitBatsmen,
        Balks,
        WildPitches,
        Pickoffs,
        RBI,
        GamesFinished,
        InheritedRunners,
        InheritedRunnersScored,
        CatchersInterference,
        SacBunts,
        SacFlies,
        PassedBall
    }

    /// <summary>
    /// Enumeration for Fielding Statistics.
    /// </summary>
    public enum FieldingStat
    {
        GamesStarted,
        Chances,
        PutOuts,
        Assists,
        Errors
        // Note: 'CaughtStealing' and 'StolenBases' are marked as "N/A" in the original mapping and thus excluded
    }

    /// <summary>
    /// VectorGenerator class responsible for generating game vectors based on a given Game Pk.
    /// </summary>
    public class VectorGenerator
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<VectorGenerator> _logger;

        // Base URLs
        private const string BaseGamelogUrl = "https://baseballsavant.mlb.com/gf?game_pk={0}";

        /// <summary>
        /// Initializes a new instance of the VectorGenerator class.
        /// </summary>
        /// <param name="logger">ILogger instance for logging.</param>
        public VectorGenerator(ILogger<VectorGenerator> logger)
        {
            _httpClient = new HttpClient();
            _logger = logger;
        }

        /// <summary>
        /// Generates a game vector for the specified Game Pk.
        /// </summary>
        /// <param name="gamePk">The unique identifier for the game.</param>
        /// <returns>A GameVector object containing aggregated statistics.</returns>
        public async Task<GameVector> GenerateGameVectorAsync(int gamePk)
        {
            var gameVector = new GameVector
            {
                GamePk = gamePk
            };

            // Fetch game data
            var gameJson = await FetchGameJsonAsync(gamePk);
            if (gameJson == null)
            {
                _logger.LogError($"Failed to fetch game data for GamePk: {gamePk}");
                return null;
            }

            // Parse game date
            if (!DateTime.TryParseExact(gameJson.GameDate, "MM/dd/yyyy", CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime parsedDate))
            {
                try
                {
                    parsedDate = DateTime.Parse(gameJson.GameDate); // Fallback parsing
                }
                catch (Exception ex)
                {
                    _logger.LogError($"Failed to parse game date '{gameJson.GameDate}' for GamePk {gamePk}: {ex.Message}");
                    parsedDate = DateTime.MinValue; // Default date
                }
            }
            gameVector.GameDate = parsedDate.ToString("yyyy-MM-dd");

            // Extract team information
            var homeTeam = gameJson.HomeTeamData;
            var awayTeam = gameJson.AwayTeamData;

            gameVector.HomeTeamAbbr = homeTeam.Abbreviation;
            gameVector.AwayTeamAbbr = awayTeam.Abbreviation;

            gameVector.ParkId = homeTeam.Venue?.Id ?? 0; // Handle null Venue

            // Aggregate Home Team Stats
            var homeAggregatedStats = await AggregateTeamStatsAsync(gameJson, homeTeam, "home", gameVector.GameDate);
            gameVector.HomeBattingStats = homeAggregatedStats.Item1;
            gameVector.HomePitchingStats = homeAggregatedStats.Item2;
            gameVector.HomeFieldingStats = homeAggregatedStats.Item3;
            gameVector.HomeBullpenStats = homeAggregatedStats.Item4;

            // Aggregate Away Team Stats
            var awayAggregatedStats = await AggregateTeamStatsAsync(gameJson, awayTeam, "away", gameVector.GameDate);
            gameVector.AwayBattingStats = awayAggregatedStats.Item1;
            gameVector.AwayPitchingStats = awayAggregatedStats.Item2;
            gameVector.AwayFieldingStats = awayAggregatedStats.Item3;
            gameVector.AwayBullpenStats = awayAggregatedStats.Item4;

            // Determine Game Outcome
            try
            {
                int homeRuns = gameJson.Scoreboard?.Linescore?.Teams?.Home?.Runs ?? 0;
                int awayRuns = gameJson.Scoreboard?.Linescore?.Teams?.Away?.Runs ?? 0;
                gameVector.HomeWin = homeRuns >= awayRuns ? 1 : 0;
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error determining game outcome for GamePk {gamePk}: {ex.Message}");
                gameVector.HomeWin = 0;
            }

            return gameVector;
        }

        #region Private Helper Methods

        /// <summary>
        /// Fetches the game JSON data for a given Game Pk.
        /// </summary>
        /// <param name="gamePk">The unique identifier for the game.</param>
        /// <returns>A GameJson object containing the game's data.</returns>
        private async Task<GameJson> FetchGameJsonAsync(int gamePk)
        {
            string url = string.Format(BaseGamelogUrl, gamePk);
            try
            {
                var response = await _httpClient.GetAsync(url);
                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogError($"Failed to fetch gamelog for GamePk {gamePk}: {response.StatusCode}");
                    return null;
                }

                var content = await response.Content.ReadAsStringAsync();
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    // Handle missing members gracefully
                    IgnoreNullValues = true
                };
                var gameJson = JsonSerializer.Deserialize<GameJson>(content, options);
                return gameJson;
            }
            catch (Exception ex)
            {
                _logger.LogError($"Exception while fetching GamePk {gamePk}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Aggregates team statistics based on player data.
        /// </summary>
        /// <param name="gameJson">The game data.</param>
        /// <param name="team">The team data.</param>
        /// <param name="teamType">"home" or "away".</param>
        /// <param name="gameDate">The date of the game in "yyyy-MM-dd" format.</param>
        /// <returns>
        /// A tuple containing:
        /// - Batting statistics
        /// - Pitching statistics
        /// - Fielding statistics
        /// - Bullpen pitching statistics
        /// </returns>
        private async Task<(Dictionary<BattingStat, double>, Dictionary<PitchingStat, double>, Dictionary<FieldingStat, double>, Dictionary<PitchingStat, double>)> AggregateTeamStatsAsync(GameJson gameJson, TeamData team, string teamType, string gameDate)
        {
            var battingStats = new Dictionary<BattingStat, double>();
            var pitchingStats = new Dictionary<PitchingStat, double>();
            var fieldingStats = new Dictionary<FieldingStat, double>();
            var bullpenStats = new Dictionary<PitchingStat, double>();

            // Get lineup and bullpen
            var startingBatters = team.Lineup != null ? new List<int>(team.Lineup) : new List<int>();
            var startingPitcher = team.ProbablePitcherId;
            var bullpen = team.Bullpen ?? new List<int>();

            // Include starting pitcher in lineup if not already present
            if (startingPitcher != 0 && !startingBatters.Contains(startingPitcher))
            {
                startingBatters.Add(startingPitcher);
            }

            // Determine if there is a DH in the lineup
            bool hasDH = false;
            foreach (var batterId in startingBatters)
            {
                var playerInfo = GetPlayerInfo(gameJson, teamType, batterId);
                if (playerInfo == null)
                    continue;

                var positionAbbr = playerInfo.Position?.Abbreviation?.ToUpper() ?? "";
                if (positionAbbr == "DH")
                {
                    hasDH = true;
                    break;
                }
            }

            int numBatters = 0;
            int numPitchers = 0;

            foreach (var playerId in startingBatters)
            {
                var playerInfo = GetPlayerInfo(gameJson, teamType, playerId);
                if (playerInfo == null)
                {
                    _logger.LogWarning($"Player info not found for PlayerId {playerId} in team {teamType}");
                    continue;
                }

                var stats = ReconstructPlayerStats(gameJson, playerInfo, teamType == "home", gameDate);

                // Determine if player is a pitcher
                bool isPitcher = playerInfo.Position?.Abbreviation?.ToUpper() == "P";
                bool isBatter = !isPitcher;

                if (isBatter)
                {
                    numBatters++;
                    foreach (BattingStat stat in Enum.GetValues(typeof(BattingStat)))
                    {
                        if (stats.BattingStats.ContainsKey(stat))
                        {
                            if (battingStats.ContainsKey(stat))
                                battingStats[stat] += stats.BattingStats[stat];
                            else
                                battingStats[stat] = stats.BattingStats[stat];
                        }
                    }

                    // Fielding Stats
                    var position = playerInfo.Position?.Abbreviation?.ToUpper() ?? "";
                    if (Enum.TryParse<FieldingStat>(position, out FieldingStat fieldStatEnum))
                    {
                        foreach (FieldingStat stat in Enum.GetValues(typeof(FieldingStat)))
                        {
                            if (stats.FieldingStats.ContainsKey(stat))
                            {
                                if (fieldingStats.ContainsKey(stat))
                                    fieldingStats[stat] += stats.FieldingStats[stat];
                                else
                                    fieldingStats[stat] = stats.FieldingStats[stat];
                            }
                        }
                    }
                }

                if (isPitcher)
                {
                    numPitchers++;
                    foreach (PitchingStat stat in Enum.GetValues(typeof(PitchingStat)))
                    {
                        if (stats.PitchingStats.ContainsKey(stat))
                        {
                            if (pitchingStats.ContainsKey(stat))
                                pitchingStats[stat] += stats.PitchingStats[stat];
                            else
                                pitchingStats[stat] = stats.PitchingStats[stat];
                        }
                    }
                }
            }

            // Aggregate Bullpen Pitchers
            foreach (var bullpenPitcherId in bullpen)
            {
                if (bullpenPitcherId == 0)
                {
                    _logger.LogWarning($"Encountered invalid bullpen pitcher ID 0 in team {teamType}");
                    continue;
                }

                var pitcherInfo = GetPlayerInfo(gameJson, teamType, bullpenPitcherId);
                if (pitcherInfo == null)
                {
                    _logger.LogWarning($"Bullpen pitcher info not found for PlayerId {bullpenPitcherId} in team {teamType}");
                    continue;
                }

                var pitcherStats = ReconstructPlayerStats(gameJson, pitcherInfo, teamType == "home", gameDate);

                foreach (PitchingStat stat in Enum.GetValues(typeof(PitchingStat)))
                {
                    if (pitcherStats.PitchingStats.ContainsKey(stat))
                    {
                        if (bullpenStats.ContainsKey(stat))
                            bullpenStats[stat] += pitcherStats.PitchingStats[stat];
                        else
                            bullpenStats[stat] = pitcherStats.PitchingStats[stat];
                    }
                }
            }

            // Normalize Batting Stats
            if (numBatters > 0)
            {
                foreach (var stat in battingStats.Keys)
                {
                    battingStats[stat] /= numBatters;
                }
            }

            // Normalize Pitching Stats
            if (numPitchers > 0)
            {
                foreach (var stat in pitchingStats.Keys)
                {
                    pitchingStats[stat] /= numPitchers;
                }
            }

            // Normalize Bullpen Pitching Stats
            int numSubs = bullpen.Count;
            if (numSubs > 0)
            {
                foreach (var stat in bullpenStats.Keys)
                {
                    bullpenStats[stat] /= numSubs;
                }
            }

            // Normalize Fielding Stats
            if (numBatters > 0)
            {
                foreach (var stat in fieldingStats.Keys)
                {
                    fieldingStats[stat] /= numBatters;
                }
            }

            return (battingStats, pitchingStats, fieldingStats, bullpenStats);
        }

        /// <summary>
        /// Reconstructs player statistics adjusted for the current game.
        /// </summary>
        /// <param name="gameJson">The game data.</param>
        /// <param name="playerInfo">The player's information.</param>
        /// <param name="isHome">Indicates if the player is on the home team.</param>
        /// <param name="gameDate">The date of the game in "yyyy-MM-dd" format.</param>
        /// <returns>A PlayerStatsAdjusted object containing adjusted statistics.</returns>
        private PlayerStatsAdjusted ReconstructPlayerStats(GameJson gameJson, PlayerInfo playerInfo, bool isHome, string gameDate)
        {
            var statsDict = new PlayerStatsAdjusted();

            string positionAbbr = playerInfo.Position?.Abbreviation?.ToUpper() ?? "";

            // Adjust stats
            // Helper function to adjust stats
            double AdjustStat(Dictionary<string, double> seasonStats, Dictionary<string, double> gameStats, string statKey)
            {
                double seasonStat = seasonStats.ContainsKey(statKey) ? seasonStats[statKey] : 0;
                double gameStat = gameStats.ContainsKey(statKey) ? gameStats[statKey] : 0;
                return seasonStat - gameStat;
            }

            // Batting Stats
            var seasonBatting = playerInfo.SeasonStats?.Batting ?? new Dictionary<string, double>();
            var gameBatting = playerInfo.Stats?.Batting ?? new Dictionary<string, double>();
            foreach (BattingStat stat in Enum.GetValues(typeof(BattingStat)))
            {
                string statKey = Enum.GetName(typeof(BattingStat), stat);
                if (statKey == null)
                    continue;

                double adjusted = AdjustStat(seasonBatting, gameBatting, statKey);
                statsDict.BattingStats[stat] = adjusted;
            }

            if (positionAbbr == "P")
            {
                // Pitching Stats
                var seasonPitching = playerInfo.SeasonStats?.Pitching ?? new Dictionary<string, double>();
                var gamePitching = playerInfo.Stats?.Pitching ?? new Dictionary<string, double>();
                foreach (PitchingStat stat in Enum.GetValues(typeof(PitchingStat)))
                {
                    string statKey = Enum.GetName(typeof(PitchingStat), stat);
                    if (statKey == null)
                        continue;

                    double adjusted = AdjustStat(seasonPitching, gamePitching, statKey);
                    // Special handling for InningsPitched -> Outs
                    if (stat == PitchingStat.InningsPitched)
                    {
                        adjusted = ConvertInningsToOuts(adjusted);
                    }
                    statsDict.PitchingStats[stat] = adjusted;
                }
            }

            // Fielding Stats
            if (Enum.TryParse<FieldingStat>(positionAbbr, out FieldingStat fieldStatEnum))
            {
                var seasonFielding = playerInfo.SeasonStats?.Fielding ?? new Dictionary<string, double>();
                var gameFielding = playerInfo.Stats?.Fielding ?? new Dictionary<string, double>();

                foreach (FieldingStat stat in Enum.GetValues(typeof(FieldingStat)))
                {
                    string statKey = Enum.GetName(typeof(FieldingStat), stat);
                    if (statKey == null)
                        continue;

                    double adjusted = AdjustStat(seasonFielding, gameFielding, statKey);
                    statsDict.FieldingStats[stat] = adjusted;
                }
            }

            // Normalize stats by games played to get per-game averages
            int gamesPlayed = 1; // Default to 1 to avoid division by zero
            if (positionAbbr == "P")
            {
                // Ensure "GamesPlayed" exists and is greater than 0
                if (playerInfo.SeasonStats?.Pitching != null && playerInfo.SeasonStats.Pitching.ContainsKey("GamesPlayed"))
                {
                    gamesPlayed = (int)playerInfo.SeasonStats.Pitching["GamesPlayed"] - 1;
                }

                if (gamesPlayed <= 0)
                {
                    _logger.LogWarning($"GamesPlayed is {gamesPlayed} for PlayerId {playerInfo.Person.Id}. Defaulting to 1.");
                    gamesPlayed = 1;
                }
            }
            else
            {
                // Ensure "GamesPlayed" exists and is greater than 0
                if (playerInfo.SeasonStats?.Batting != null && playerInfo.SeasonStats.Batting.ContainsKey("GamesPlayed"))
                {
                    gamesPlayed = (int)playerInfo.SeasonStats.Batting["GamesPlayed"] - 1;
                }

                if (gamesPlayed <= 0)
                {
                    _logger.LogWarning($"GamesPlayed is {gamesPlayed} for PlayerId {playerInfo.Person.Id}. Defaulting to 1.");
                    gamesPlayed = 1;
                }
            }

            // Normalize Batting Stats
            foreach (var stat in statsDict.BattingStats.Keys)
            {
                statsDict.BattingStats[stat] /= gamesPlayed;
            }

            if (positionAbbr == "P")
            {
                // Normalize Pitching Stats
                foreach (var stat in statsDict.PitchingStats.Keys)
                {
                    statsDict.PitchingStats[stat] /= gamesPlayed;
                }
            }

            // Normalize Fielding Stats
            foreach (var stat in statsDict.FieldingStats.Keys)
            {
                statsDict.FieldingStats[stat] /= gamesPlayed;
            }

            return statsDict;
        }

        /// <summary>
        /// Converts innings pitched to outs.
        /// </summary>
        /// <param name="innings">Innings pitched as a decimal.</param>
        /// <returns>Outs pitched.</returns>
        private double ConvertInningsToOuts(double innings)
        {
            // Assuming innings is represented as whole + decimal (e.g., 6.1 innings = 6 and 1 out)
            int wholeInnings = (int)Math.Floor(innings);
            double fractional = innings - wholeInnings;
            int outs = wholeInnings * 3 + (int)Math.Round(fractional * 10 / 3);
            return outs / 3.0; // Convert back to innings if needed
        }

        /// <summary>
        /// Retrieves player information from the game data.
        /// </summary>
        /// <param name="gameJson">The game data.</param>
        /// <param name="teamType">"home" or "away".</param>
        /// <param name="playerId">The player's unique identifier.</param>
        /// <returns>A PlayerInfo object or null if not found.</returns>
        private PlayerInfo GetPlayerInfo(GameJson gameJson, string teamType, int playerId)
        {
            var teamPlayers = teamType.ToLower() == "home" ? gameJson.HomeTeamData.Players : gameJson.AwayTeamData.Players;
            if (teamPlayers == null)
            {
                _logger.LogWarning($"Players list is null for team type {teamType}");
                return null;
            }

            foreach (var player in teamPlayers)
            {
                if (player.Person.Id == playerId)
                    return player;
            }
            _logger.LogWarning($"PlayerStats not found for PlayerId {playerId} in team {teamType}");
            return null;
        }

        #endregion
    }

    #region JSON Models

    // Define JSON models based on the expected structure of the API responses.
    // Adjust the properties as per the actual JSON structure.

    public class GameJson
    {
        [JsonPropertyName("gamePk")]
        public int GamePk { get; set; }

        [JsonPropertyName("gameDate")]
        public string GameDate { get; set; }

        [JsonPropertyName("scoreboard")]
        public Scoreboard Scoreboard { get; set; }

        [JsonPropertyName("home_team_data")]
        public TeamData HomeTeamData { get; set; }

        [JsonPropertyName("away_team_data")]
        public TeamData AwayTeamData { get; set; }
    }

    public class Scoreboard
    {
        [JsonPropertyName("linescore")]
        public LineScore Linescore { get; set; }
    }

    public class LineScore
    {
        [JsonPropertyName("teams")]
        public LineScoreTeams Teams { get; set; }
    }

    public class LineScoreTeams
    {
        [JsonPropertyName("home")]
        public LineScoreTeam Home { get; set; }

        [JsonPropertyName("away")]
        public LineScoreTeam Away { get; set; }
    }

    public class LineScoreTeam
    {
        [JsonPropertyName("runs")]
        public int Runs { get; set; }
    }

    public class TeamData
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }

        [JsonPropertyName("abbreviation")]
        public string Abbreviation { get; set; }

        [JsonPropertyName("venue")]
        public Venue Venue { get; set; }

        [JsonPropertyName("players")]
        public List<PlayerInfo> Players { get; set; }

        [JsonPropertyName("lineup")]
        public List<int> Lineup { get; set; }

        [JsonPropertyName("probablePitcherId")]
        public int ProbablePitcherId { get; set; }

        [JsonPropertyName("bullpen")]
        public List<int> Bullpen { get; set; }
    }

    public class Venue
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }
    }

    public class PlayerInfo
    {
        [JsonPropertyName("person")]
        public Person Person { get; set; }

        [JsonPropertyName("position")]
        public Position Position { get; set; }

        [JsonPropertyName("parentTeamId")]
        public int ParentTeamId { get; set; }

        [JsonPropertyName("seasonStats")]
        public SeasonStats SeasonStats { get; set; }

        [JsonPropertyName("stats")]
        public PlayerStats Stats { get; set; }
    }

    public class Person
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }
    }

    public class Position
    {
        [JsonPropertyName("abbreviation")]
        public string Abbreviation { get; set; }
    }

    public class SeasonStats
    {
        [JsonPropertyName("batting")]
        public Dictionary<string, double> Batting { get; set; }

        [JsonPropertyName("pitching")]
        public Dictionary<string, double> Pitching { get; set; }

        [JsonPropertyName("fielding")]
        public Dictionary<string, double> Fielding { get; set; }
    }

    public class PlayerStats
    {
        [JsonPropertyName("batting")]
        public Dictionary<string, double> Batting { get; set; }

        [JsonPropertyName("pitching")]
        public Dictionary<string, double> Pitching { get; set; }

        [JsonPropertyName("fielding")]
        public Dictionary<string, double> Fielding { get; set; }
    }

    #endregion

    #region Supporting Classes

    /// <summary>
    /// Represents adjusted player statistics after accounting for the current game.
    /// </summary>
    public class PlayerStatsAdjusted
    {
        public Dictionary<BattingStat, double> BattingStats { get; set; } = new Dictionary<BattingStat, double>();
        public Dictionary<PitchingStat, double> PitchingStats { get; set; } = new Dictionary<PitchingStat, double>();
        public Dictionary<FieldingStat, double> FieldingStats { get; set; } = new Dictionary<FieldingStat, double>();
    }

    #endregion
}
