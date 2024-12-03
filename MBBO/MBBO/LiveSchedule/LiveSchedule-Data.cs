using Newtonsoft.Json;
using System.Collections.Generic;

namespace MBBO.Savant
{
    public class ScheduleResponse
    {
        [JsonProperty("dates")]
        public List<ScheduleDate> Dates { get; set; }
    }

    public class ScheduleDate
    {
        [JsonProperty("date")]
        public string Date { get; set; }

        [JsonProperty("games")]
        public List<Game> Games { get; set; }
    }

    public class Game
    {
        [JsonProperty("gamePk")]
        public int GamePk { get; set; }

        [JsonProperty("teams")]
        public GameTeams Teams { get; set; }
    }

    public class GameTeams
    {
        [JsonProperty("home")]
        public TeamInfo Home { get; set; }

        [JsonProperty("away")]
        public TeamInfo Away { get; set; }
    }

    public class TeamInfo
    {
        [JsonProperty("team")]
        public Team Team { get; set; }
    }

    public class Team
    {
        [JsonProperty("id")]
        public int Id { get; set; }

        [JsonProperty("name")]
        public string Name { get; set; }
    }
}