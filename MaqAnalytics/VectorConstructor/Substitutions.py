class SubstitutionPredictor:
    """
    Predicts the expected number of innings pitched by each pitcher on a team's roster
    given the probable pitcher.
    """

    def __init__(self, game_json):
        """
        Initializes the SubstitutionPredictor with game JSON data.

        Parameters:
            game_json (dict): JSON data for the game, including boxscore and lineup information.
        """
        self.game_json = game_json

    def get_pitcher_stats(self, pitcher_id):
        """
        Retrieves season-level pitching statistics for a given pitcher.

        Parameters:
            pitcher_id (int): Unique identifier for the pitcher.

        Returns:
            dict: Season pitching stats for the pitcher or an empty dictionary if not found.
        """
        players = self.game_json['boxscore']['teams']['home']['players']
        players.update(self.game_json['boxscore']['teams']['away']['players'])

        pitcher_key = f"ID{pitcher_id}"
        if pitcher_key in players:
            return players[pitcher_key].get('seasonStats', {}).get('pitching', {})
        return {}

    def predict_innings_pitched(self, probable_pitcher_id, team_type):
        """
        Predicts the expected number of innings pitched for each pitcher on a team's roster.

        Parameters:
            probable_pitcher_id (int): Unique identifier for the probable pitcher.
            team_type (str): 'home' or 'away' indicating the team.

        Returns:
            dict: A dictionary mapping pitcher IDs to their expected innings pitched.
        """
        team = self.game_json['boxscore']['teams'][team_type]
        bullpen = team['bullpen']  # List of bullpen pitcher IDs
        pitchers = team['pitchers']  # List of all pitchers who might pitch
        probable_pitcher_stats = self.get_pitcher_stats(probable_pitcher_id)

        total_innings = probable_pitcher_stats.get('inningsPitched', 0)

        if total_innings == 0:
            print(f"Warning: Probable pitcher {probable_pitcher_id} has no innings pitched stats.")
            total_innings = 5  # Assume a default of 5 innings if no data available

        expected_innings = {pitcher_id: 0 for pitcher_id in pitchers}

        # Assign a baseline expected innings pitched to the probable pitcher
        probable_pitcher_expected = total_innings * 0.6  # Assume they pitch 60% of their average innings
        expected_innings[probable_pitcher_id] = probable_pitcher_expected

        # Distribute remaining innings among bullpen pitchers
        remaining_innings = total_innings - probable_pitcher_expected
        bullpen_pitchers = [p for p in bullpen if p != probable_pitcher_id]
        num_bullpen_pitchers = len(bullpen_pitchers)

        if num_bullpen_pitchers > 0:
            bullpen_innings = remaining_innings / num_bullpen_pitchers
            for bullpen_pitcher in bullpen_pitchers:
                expected_innings[bullpen_pitcher] = bullpen_innings
        else:
            print("Warning: No bullpen pitchers available for innings distribution.")

        # Normalize expected innings for non-pitchers (e.g., if they mistakenly show up in the list)
        for pitcher_id in expected_innings:
            if pitcher_id not in pitchers:
                expected_innings[pitcher_id] = 0

        return expected_innings
