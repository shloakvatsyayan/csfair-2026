class Hands:
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


class GameHandler:
    def __init__(self):
        self._winning_rules = self._build_winning_rules()

    def _build_winning_rules(self):
        return {
            Hands.ROCK: Hands.SCISSORS,
            Hands.PAPER: Hands.ROCK,
            Hands.SCISSORS: Hands.PAPER
        }

    def get_winner(self, player1_choice, player2_choice):
        if player1_choice == player2_choice:
            return "It's a tie!"
        
        if self._winning_rules[player1_choice] == player2_choice:
            return "Player 1 wins!"
        
        return "Player 2 wins!"
