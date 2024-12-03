// Models/StatsEnums.cs

namespace MBBO.Savant
{
    public enum BattingStats
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
        RBIs,
        Walks,
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

    public enum PitchingStats
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
        Walks,
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
        RBIs,
        GamesFinished,
        InheritedRunners,
        InheritedRunnersScored,
        CatchersInterference,
        SacBunts,
        SacFlies,
        PassedBall
    }

    public enum FieldingStats
    {
        GamesStarted,
        Chances,
        PutOuts,
        Assists,
        Errors,
        CaughtStealing,
        StolenBases
    }

    public enum FieldingPositions
    {
        P,   // Pitcher
        C,   // Catcher
        _1B, // First Baseman
        _2B, // Second Baseman
        _3B, // Third Baseman
        SS,  // Shortstop
        LF,  // Left Fielder
        CF,  // Center Fielder
        RF   // Right Fielder
    }
}