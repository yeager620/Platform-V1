from enum import Enum
from dataclasses import dataclass, field
from typing import List

from enum import Enum
from typing import List


class BattingStat(Enum):
    GAMES_PLAYED = "gamesPlayed"
    PLATE_APPEARANCES = "plateAppearances"
    AT_BATS = "atBats"
    RUNS = "runs"
    HITS = "hits"
    TOTAL_BASES = "totalBases"
    DOUBLES = "doubles"
    TRIPLES = "triples"
    HOME_RUNS = "homeRuns"
    RBI = "rbi"
    BASE_ON_BASES = "baseOnBalls"
    INTENTIONAL_WALKS = "intentionalWalks"
    STRIKE_OUTS = "strikeOuts"
    GROUND_INTO_DOUBLE_PLAY = "groundIntoDoublePlay"
    GROUND_INTO_TRIPLE_PLAY = "groundIntoTriplePlay"
    HIT_BY_PITCH = "hitByPitch"
    CAUGHT_STEALING = "caughtStealing"
    STOLEN_BASES = "stolenBases"
    LEFT_ON_BASE = "leftOnBase"
    SAC_BUNTS = "sacBunts"
    SAC_FLIES = "sacFlies"
    CATCHERS_INTERFERENCE = "catchersInterference"
    PICKOFFS = "pickoffs"
    FLY_OUTS = "flyOuts"
    GROUND_OUTS = "groundOuts"


class PitchingStat(Enum):
    GAMES_PLAYED = "gamesPlayed"
    GAMES_STARTED = "gamesStarted"
    GROUND_OUTS = "groundOuts"
    AIR_OUTS = "airOuts"
    RUNS = "runs"
    DOUBLES = "doubles"
    TRIPLES = "triples"
    HOME_RUNS = "homeRuns"
    STRIKE_OUTS = "strikeOuts"
    BASE_ON_BASES = "baseOnBalls"
    INTENTIONAL_WALKS = "intentionalWalks"
    HITS = "hits"
    HIT_BY_PITCH = "hitByPitch"
    AT_BATS = "atBats"
    CAUGHT_STEALING = "caughtStealing"
    STOLEN_BASES = "stolenBases"
    NUMBER_OF_PITCHES = "numberOfPitches"
    INNINGS_PITCHED = "inningsPitched"
    WINS = "wins"
    LOSSES = "losses"
    SAVES = "saves"
    SAVE_OPPORTUNITIES = "saveOpportunities"
    HOLDS = "holds"
    BLOWN_SAVES = "blownSaves"
    EARNED_RUNS = "earnedRuns"
    BATTERS_FACED = "battersFaced"
    OUTS = "outs"
    COMPLETE_GAMES = "completeGames"
    SHUTOUTS = "shutouts"
    PITCHES_THROWN = "pitchesThrown"
    BALLS = "balls"
    STRIKES = "strikes"
    HIT_BATSMEN = "hitBatsmen"
    BALKS = "balks"
    WILD_PITCHES = "wildPitches"
    PICKOFFS = "pickoffs"
    RBI = "rbi"
    INHERITED_RUNNERS = "inheritedRunners"
    INHERITED_RUNNERS_SCORED = "inheritedRunnersScored"
    CATCHERS_INTERFERENCE = "catchersInterference"
    SAC_BUNTS = "sacBunts"
    SAC_FLIES = "sacFlies"
    PASSED_BALL = "passedBall"


class FieldingStat(Enum):
    GAMES_STARTED = "gamesStarted"
    CHANCES = "chances"
    PUT_OUTS = "putOuts"
    ASSISTS = "assists"
    ERRORS = "errors"
    CAUGHT_STEALING = "caughtStealing"
    STOLEN_BASES = "stolenBases"


@dataclass
class BattingStats:
    stats: List[BattingStat] = field(default_factory=lambda: list(BattingStat))


@dataclass
class PitchingStats:
    stats: List[PitchingStat] = field(default_factory=lambda: list(PitchingStat))


@dataclass
class FieldingStats:
    stats: List[FieldingStat] = field(default_factory=lambda: list(FieldingStat))

@dataclass
class StartingBatters:
    batters: List[BattingStats] = field(default_factory=lambda: [BattingStats() for _ in range(9)])
    # Initializes a list of 9 BattingStats instances representing the starting lineup

@dataclass
class BenchBatters:
    aggregated_stats: BattingStats = BattingStats()
    # Aggregated batting stats for benched batters

@dataclass
class StartingPitcher:
    pitcher: PitchingStats = PitchingStats()
    # Stats for the starting pitcher

@dataclass
class BullpenPitchers:
    aggregated_stats: PitchingStats = PitchingStats()
    # Aggregated pitching stats for bullpen pitchers

@dataclass
class StartingInfielders:
    infielders: List[FieldingStats] = field(default_factory=lambda: [FieldingStats() for _ in range(4)])
    # Typically 4 infielders: 1B, 2B, 3B, SS

@dataclass
class StartingOutfielders:
    outfielders: List[FieldingStats] = field(default_factory=lambda: [FieldingStats() for _ in range(3)])
    # Typically 3 outfielders: LF, CF, RF

@dataclass
class BenchedInfielders:
    aggregated_stats: FieldingStats = FieldingStats()
    # Aggregated fielding stats for benched infielders

@dataclass
class BenchedOutfielders:
    aggregated_stats: FieldingStats = FieldingStats()
    # Aggregated fielding stats for benched outfielders

@dataclass
class TeamStats:
    starting_batters: StartingBatters = StartingBatters()
    bench_batters: BenchBatters = BenchBatters()
    starting_pitcher: StartingPitcher = StartingPitcher()
    bullpen_pitchers: BullpenPitchers = BullpenPitchers()
    starting_infielders: StartingInfielders = StartingInfielders()
    starting_outfielders: StartingOutfielders = StartingOutfielders()
    benched_infielders: BenchedInfielders = BenchedInfielders()
    benched_outfielders: BenchedOutfielders = BenchedOutfielders()

