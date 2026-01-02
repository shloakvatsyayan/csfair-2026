class Hands:
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


def get_winner(player1_choice, player2_choice):
    if player1_choice == player2_choice:
        return "It's a tie!"
    elif ((player1_choice == Hands.ROCK and player2_choice == Hands.SCISSORS) or
          (player1_choice == Hands.PAPER and player2_choice == Hands.ROCK) or
          (player1_choice == Hands.SCISSORS and player2_choice == Hands.PAPER)):
        return "Player 1 wins!"
    else:
        return "Player 2 wins!"

