using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace MBBO.Savant
{
    public class LiveGamelogs
    {
        private readonly string _baseScheduleUrl = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}&fields=dates,date,games,gamePk,teams,team";
        private readonly string _baseGamelogUrl = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}";

        private readonly int _daysAhead;
        private readonly int _maxConcurrentRequests;
        private readonly SemaphoreSlim _semaphore;
        private readonly HttpClient _httpClient;

        /// <summary>
        /// Initializes a new instance of the LiveGamelogsFetcher class.
        /// </summary>
        /// <param name="daysAhead">Number of days ahead to look for upcoming games.</param>
        /// <param name="maxConcurrentRequests">Maximum number of concurrent HTTP requests.</param>
        public LiveGamelogs(int daysAhead = 7, int maxConcurrentRequests = 10)
        {
            _daysAhead = daysAhead;
            _maxConcurrentRequests = maxConcurrentRequests;
            _semaphore = new SemaphoreSlim(_maxConcurrentRequests);
            _httpClient = new HttpClient();
        }

        /// <summary>
        /// Converts a DateTime object to 'MM/dd/yyyy' string format.
        /// </summary>
        /// <param name="date">DateTime object.</param>
        /// <returns>Formatted date string.</returns>
        private static string GetDateStr(DateTime date)
        {
            return date.ToString("MM/dd/yyyy");
        }

        /// <summary>
        /// Generates a list of upcoming dates as strings.
        /// </summary>
        /// <returns>List of date strings.</returns>
        private List<string> GetUpcomingDates()
        {
            List<string> upcomingDates = new List<string>();
            DateTime today = DateTime.Today;

            for (int i = 0; i < _daysAhead; i++)
            {
                DateTime date = today.AddDays(i);
                upcomingDates.Add(GetDateStr(date));
            }

            return upcomingDates;
        }

        /// <summary>
        /// Asynchronously fetches JSON data from a given URL.
        /// </summary>
        /// <param name="url">URL to fetch data from.</param>
        /// <returns>Deserialized JSON object.</returns>
        private async Task<JObject> FetchJsonAsync(string url)
        {
            await _semaphore.WaitAsync();
            try
            {
                HttpResponseMessage response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();
                string jsonString = await response.Content.ReadAsStringAsync();
                return JObject.Parse(jsonString);
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine($"HTTP error for URL {url}: {e.Message}");
                return null;
            }
            catch (Exception e)
            {
                Console.WriteLine($"Unexpected error for URL {url}: {e.Message}");
                return null;
            }
            finally
            {
                _semaphore.Release();
            }
        }

        /// <summary>
        /// Asynchronously fetches a single game log.
        /// </summary>
        /// <param name="gamePk">GamePk integer.</param>
        /// <returns>Game log JSON object.</returns>
        private async Task<JObject> FetchGameLogAsync(int gamePk)
        {
            string url = _baseGamelogUrl.Replace("{game_pk}", gamePk.ToString());
            return await FetchJsonAsync(url);
        }

        /// <summary>
        /// Asynchronously fetches the schedule between start_date and end_date.
        /// </summary>
        /// <param name="startDate">Start date string in 'MM/dd/yyyy'.</param>
        /// <param name="endDate">End date string in 'MM/dd/yyyy'.</param>
        /// <returns>ScheduleResponse object.</returns>
        private async Task<ScheduleResponse> FetchScheduleAsync(string startDate, string endDate)
        {
            string url = _baseScheduleUrl.Replace("{start_date}", startDate).Replace("{end_date}", endDate);
            JObject scheduleJson = await FetchJsonAsync(url);
            if (scheduleJson == null)
                return null;

            return scheduleJson.ToObject<ScheduleResponse>();
        }

        /// <summary>
        /// Identifies the next game for each team and returns unique gamePk's.
        /// </summary>
        /// <returns>List of unique gamePk integers.</returns>
        public async Task<List<int>> GetNextGamesAsync()
        {
            List<string> upcomingDates = GetUpcomingDates();
            Dictionary<int, int> teamNextGame = new Dictionary<int, int>(); // team_id -> gamePk

            foreach (var date in upcomingDates)
            {
                ScheduleResponse schedule = await FetchScheduleAsync(date, date);
                if (schedule?.Dates == null)
                    continue;

                foreach (var dateInfo in schedule.Dates)
                {
                    foreach (var game in dateInfo.Games)
                    {
                        int gamePk = game.GamePk;
                        int homeTeamId = game.Teams.Home.Team.Id;
                        int awayTeamId = game.Teams.Away.Team.Id;

                        // Assign the gamePk as the next game for both teams if not already assigned
                        foreach (var teamId in new List<int> { homeTeamId, awayTeamId })
                        {
                            if (!teamNextGame.ContainsKey(teamId))
                            {
                                teamNextGame[teamId] = gamePk;
                            }
                        }

                        // Early exit if all teams have been assigned (assuming 30 MLB teams)
                        if (teamNextGame.Count >= 30)
                            break;
                    }

                    if (teamNextGame.Count >= 30)
                        break;
                }

                if (teamNextGame.Count >= 30)
                    break;
            }

            // Extract unique gamePk's from the next games of all teams
            List<int> uniqueGamePks = new HashSet<int>(teamNextGame.Values).ToList();
            Console.WriteLine($"Identified {uniqueGamePks.Count} unique upcoming games.");
            return uniqueGamePks;
        }

        /// <summary>
        /// Asynchronously fetches game logs for a list of game Pk's.
        /// </summary>
        /// <param name="gamePks">List of gamePk integers.</param>
        /// <returns>List of game log JSON objects.</returns>
        public async Task<List<JObject>> FetchGameLogsAsync(List<int> gamePks)
        {
            List<JObject> gamelogs = new List<JObject>();

            List<Task<JObject>> tasks = new List<Task<JObject>>();
            foreach (var gamePk in gamePks)
            {
                tasks.Add(FetchGameLogAsync(gamePk));
            }

            var results = await Task.WhenAll(tasks);
            foreach (var result in results)
            {
                if (result != null)
                    gamelogs.Add(result);
            }

            return gamelogs;
        }

        /// <summary>
        /// Fetches game logs for the next game of each team.
        /// </summary>
        /// <returns>List of game log JSON objects.</returns>
        public async Task<List<JObject>> GetGamelogsForNextGamesAsync()
        {
            List<int> nextGamePks = await GetNextGamesAsync();
            List<JObject> gamelogs = await FetchGameLogsAsync(nextGamePks);
            return gamelogs;
        }

        /// <summary>
        /// Synchronous wrapper to fetch game logs.
        /// </summary>
        /// <returns>List of game log JSON objects.</returns>
        public List<JObject> GetGamelogs()
        {
            return GetGamelogsForNextGamesAsync().GetAwaiter().GetResult();
        }
    }
}