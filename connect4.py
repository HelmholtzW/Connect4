import numpy as np
import random


class Connect4:
    ACTION_SPACE_SIZE = 7
    REWARD_WIN = 1
    REWARD_LOSS = -1

    def __init__(self):
        self.board = self.init_board()
        self.available_moves = [0, 1, 2, 3, 4, 5, 6]
        self.winner = 0
        self.set_pieces = [0, 0, 0, 0, 0, 0, 0]

    def reset(self):
        self.board = self.init_board()
        self.available_moves = [0, 1, 2, 3, 4, 5, 6]
        self.winner = 0
        self.set_pieces = [0, 0, 0, 0, 0, 0, 0]
        return self.board

    def get_available_moves(self):
        return self.available_moves

    def init_board(self):
        return np.zeros([6, 7])

    def print_board(self):
        for i in range(6):
            print(self.board[5 - 1])

    def random_move(self):
        return random.choice(self.available_moves)

    def place_piece(self, player, column, board, set_pieces):
        board[set_pieces[column]][column] = player
        set_pieces[column] += 1

    def update_available_moves(self):
        new_available_moves = []
        for i in range(7):
            if self.board[5][i] == 0:
                new_available_moves.append(i)
        self.available_moves = new_available_moves

    def determine_win(self, board):
        boardHeight = 7
        boardWidth = 6

        # check horizontal spaces
        for y in range(boardHeight):
            for x in range(boardWidth - 3):
                # print(x, y, board[x][y])
                if board[x][y] == board[x + 1][y] == board[x + 2][y] == board[x + 3][y] != 0:
                    winner = board[x][y]
                    # board[x][y] = board[x + 1][y] = board[x +
                    #                                                      2][y] = board[x + 3][y] = 8
                    return winner

        # check vertical spaces
        for x in range(boardWidth):
            for y in range(boardHeight - 3):
                # print(x, y)
                if board[x][y] == board[x][y + 1] == board[x][y + 2] == board[x][y + 3] != 0:
                    winner = board[x][y]
                    # board[x][y] = board[x][y + 1] = board[x][y +
                    #                                                         2] = board[x][y + 3] = 8
                    return winner

        # check / diagonal spaces
        for x in range(boardWidth - 3):
            for y in range(3, boardHeight):
                if board[x][y] == board[x + 1][y - 1] == board[x + 2][y - 2] == board[x + 3][y - 3] != 0:
                    winner = board[x][y]
                    # board[x][y] = board[x + 1][y - 1] = board[x +
                    #                                                          2][y - 2] = board[x + 3][y - 3] = 8
                    return winner

        # check \ diagonal spaces
        for x in range(boardWidth - 3):
            for y in range(boardHeight - 3):
                if board[x][y] == board[x + 1][y + 1] == board[x + 2][y + 2] == board[x + 3][y + 3] != 0:
                    winner = board[x][y]
                    # board[x][y] = board[x + 1][y + 1] = board[x +
                    #                                                          2][y + 2] = board[x + 3][y + 3] = 8
                    return winner

        return 0

    def win_next_round(self, player):
        actions = random.sample(self.available_moves,
                                len(self.available_moves))
        for action in actions:
            test_board = np.array(self.board)
            test_pieces = list(self.set_pieces)
            self.place_piece(player, action, test_board, test_pieces)
            win = self.determine_win(test_board)
            if win == player:
                return action
        return

    def bench_move(self):
        move = self.win_next_round(-1)
        if move is not None:
            return move
        move = self.win_next_round(1)
        if move is not None:
            return move
        return self.random_move()

    def step(self, player, action):
        # check if chosen action is available
        if action not in self.available_moves:
            print(f"Move {action} not available")
            print(self.board)
            print(self.available_moves)
            return

        # Make move
        self.place_piece(player, action, self.board, self.set_pieces)
        self.update_available_moves()

        self. winner = self.determine_win(self.board)

        if self.winner == 1:
            reward = self.REWARD_WIN
            done = True
        if self.winner == -1:
            reward = self.REWARD_LOSS
            done = True
        if self.winner == 0:
            reward = 0
            done = False

        if len(self.available_moves) == 0:
            done = True

        new_observation = self.board

        return new_observation, reward, done
