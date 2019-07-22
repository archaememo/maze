from players.abstract_player import AbstractPlayer


class Player(AbstractPlayer):
    def __init__(self, **kwargs):
        super(Player, self).__init__(**kwargs)
        self.run()
