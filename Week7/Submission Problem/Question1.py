from collections import Counter
import random


# BOARD CLASS
class Board:    
    def __init__(self):
        self.state = [" "] * 9

    def __str__(self):
        b = self.state
        return (
            f"\n 0 | 1 | 2     {b[0]} | {b[1]} | {b[2]}\n"
            f"---+---+---   ---+---+---\n"
            f" 3 | 4 | 5     {b[3]} | {b[4]} | {b[5]}\n"
            f"---+---+---   ---+---+---\n"
            f" 6 | 7 | 8     {b[6]} | {b[7]} | {b[8]}\n"
        )

    def valid_move(self, idx):
        try:
            idx = int(idx)
            return 0 <= idx <= 8 and self.state[idx] == " "
        except:
            return False

    def play(self, idx, mark):
        self.state[idx] = mark

    def as_string(self):
        return "".join(self.state)

    def is_win(self):
        b = self.state
        wins = [
            (0,1,2),(3,4,5),(6,7,8),   # rows
            (0,3,6),(1,4,7),(2,5,8),   # columns
            (0,4,8),(2,4,6)            # diagonals
        ]
        return any(b[a] != " " and b[a] == b[b_] == b[c] for a,b_,c in wins)

    def is_draw(self):
        return " " not in self.state


# MENACE PLAYER
class Menace:
    def __init__(self):
        self.boxes = {}          # board_string → [moves...]
        self.memory = []         # [(board_string, chosen_move)]
        self.W = self.D = self.L = 0

    def start(self):
        self.memory = []

    def choose(self, board: Board):
        key = board.as_string()

        # If first time seeing this state — create matchbox
        if key not in self.boxes:
            legal = [i for i, v in enumerate(key) if v == " "]
            # Start with 2 beads per legal move → closer to MENACE
            beads = []
            for mv in legal:
                beads += [mv, mv]
            self.boxes[key] = beads

        beads = self.boxes[key]

        if not beads:
            return -1  # resign

        move = random.choice(beads)
        self.memory.append((key, move))
        return move

    # learning rules
    def reward_win(self):
        for key, mv in self.memory:
            self.boxes[key].extend([mv, mv, mv])  # +3
        self.W += 1

    def reward_draw(self):
        for key, mv in self.memory:
            self.boxes[key].append(mv)            # +1
        self.D += 1

    def reward_loss(self):
        for key, mv in self.memory:
            if mv in self.boxes[key]:
                self.boxes[key].remove(mv)        # -1
        self.L += 1

    # utilities
    def stats(self):
        print(f"Learned {len(self.boxes)} matchboxes")
        print(f"W/D/L = {self.W}/{self.D}/{self.L}")

    def view_prob(self, board: Board):
        key = board.as_string()
        if key not in self.boxes:
            print("Never seen this board before.")
            return
        print("Bead counts:", Counter(self.boxes[key]).most_common())

# HUMAN PLAYER
class Human:
    def start(self):
        print("Your turn!")

    def choose(self, board: Board):
        while True:
            pos = input("Move (0–8): ").strip()
            if board.valid_move(pos):
                return int(pos)
            print("Invalid move.")

    def reward_win(self):  print("You win!")
    def reward_draw(self): print("Draw.")
    def reward_loss(self): print("You lose.")
    def view_prob(self, b): pass


# Game loop
def play(pX, pO, silent=False):
    board = Board()
    pX.start()
    pO.start()

    if not silent:
        print(board)

    turn = pX
    mark = "X"

    while True:

        # Print MENACE bead-probabilities
        if not silent:
            turn.view_prob(board)

        mv = turn.choose(board)
        if mv == -1:
            # resignation
            if turn is pX:
                pX.reward_loss()
                pO.reward_win()
            else:
                pO.reward_loss()
                pX.reward_win()
            break

        board.play(mv, mark)

        if not silent:
            print(board)

        if board.is_win():
            if turn is pX:
                pX.reward_win()
                pO.reward_loss()
            else:
                pO.reward_win()
                pX.reward_loss()
            break

        if board.is_draw():
            pX.reward_draw()
            pO.reward_draw()
            break

        # switch turn
        if turn is pX:
            turn = pO
            mark = "O"
        else:
            turn = pX
            mark = "X"


if __name__ == "__main__":
    mX = Menace()
    mO = Menace()
    human = Human()

    print("Training MENACE vs MENACE for 2000 games...")
    for _ in range(2000):
        play(mX, mO, silent=True)

    print("MENACE stats as X:")
    mX.stats()
    print("MENACE stats as O:")
    mO.stats()

    print("\nPlay vs MENACE:")
    play(mX, human)
    play(human, mO)
