using Newtonsoft.Json;
using System.Collections.Generic;

namespace MBBO.Savant
{
    public class GameLog
    {
        [JsonProperty("gamePk")]
        public int GamePk { get; set; }

        [JsonProperty("gameDate")]
        public string GameDate { get; set; }

        [JsonProperty("boxscore")]
        public Boxscore Boxscore { get; set; }

        [JsonProperty("home_team_data")]
        public TeamData HomeTeamData { get; set; }

        [JsonProperty("away_team_data")]
        public TeamData AwayTeamData { get; set; }

        // Additional properties as per the JSON structure
    }

    public class Boxscore
    {
        [JsonProperty("teams")]
        public BoxscoreTeams Teams { get; set; }
    }

    public class BoxscoreTeams
    {
        [JsonProperty("home")]
        public TeamBoxscore Home { get; set; }

        [JsonProperty("away")]
        public TeamBoxscore Away { get; set; }
    }

    public class TeamBoxscore
    {
        [JsonProperty("batters")]
        public List<int> Batters { get; set; }

        [JsonProperty("pitchers")]
        public List<int> Pitchers { get; set; }

        [JsonProperty("bullpen")]
        public List<int> Bullpen { get; set; }

        [JsonProperty("battingOrder")]
        public List<int> BattingOrder { get; set; }

        [JsonProperty("bench")]
        public List<int> Bench { get; set; }
    }

    public class TeamData
    {
        [JsonProperty("id")]
        public int Id { get; set; }

        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("abbreviation")]
        public string Abbreviation { get; set; }

        // Additional properties as needed
    }
}