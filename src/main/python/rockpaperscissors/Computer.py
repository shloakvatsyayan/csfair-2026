import random
from rockpaperscissors.GameHandler import Hands

class Computer:
    def __init__(self, win_probability=0.6, tie_probability=0.2, lose_probability=0.2):
        self._win_probability = win_probability
        self._tie_probability = tie_probability
        self._lose_probability = lose_probability

    def choose(self, user_choice):
        winning_choice = self._get_winning_choice(user_choice)
        losing_choice = self._get_losing_choice(user_choice)
        random_value = random.random()
        if random_value < self._win_probability:
            return winning_choice
        elif random_value < self._win_probability + self._tie_probability:
            return user_choice
        else:
            return losing_choice

    def _get_winning_choice(self, user_choice):
        if user_choice == Hands.ROCK:
            return Hands.PAPER
        elif user_choice == Hands.PAPER:
            return Hands.SCISSORS
        else:
            return Hands.ROCK

    def _get_losing_choice(self, user_choice):
        if user_choice == Hands.ROCK:
            return Hands.SCISSORS
        elif user_choice == Hands.PAPER:
            return Hands.ROCK
        else:
            return Hands.PAPER

