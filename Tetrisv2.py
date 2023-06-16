'''
Code taken and modified from https://github.com/nuno-faria/tetris-ai/blob/master/tetris.py
'''

import numpy as np
import random
import cv2
from PIL import Image
from time import sleep

class TetrisEnv:

    # BOARD
    MAP_EMPTY = 0
    EXISTING_BLOCK = 1
    MAP_PLAYER = 2

    TETRIMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    COLOURS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }

    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        self.reset()

    def get_midpoint_integer(self,num):
        midpoint = num // 2
        if num % 2 != 0:
            midpoint += 1
        return midpoint


    def reset(self):
        """
        Assume step method will not be called before reset has been called.
        Reset called whenever a done signal is issued
        For Tetris, randomly choose tetriminos but the starting placement is always the same.
        Method shuld return a tuple of initial obsercation and some auziliary information.
        Can use prior two method for that.
        """

        self.board = [[0] * self.WIDTH for _ in range(self.HEIGHT)]
        self.score = 0
        self.cleared_lines = 0
        self.game_over = False

        # # Randomly picks each piece once and ensures all pieces are gone through before the same piece is used again
        # self.bag = list(range(len(TETRIMINOS)))
        # random.shuffle(self.bag)
        # self.next_piece = self.bag.pop()

        # Choose random tetrinimos at random
        self.next_piece = random.choice(list(TetrisEnv.TETRIMINOS))

        self._new_round()

        # QUESTION: Not really needed in this implementation
        return self._get_board_props(self.board)


    def _new_round(self):
        '''Starts a new round (new piece)'''

        # Prior implementation
        # self.current_pos = [3, 0]

        self.current_piece = self.next_piece
        # Current piece starts at the center of the top line
        midpoint = self.get_midpoint_integer(self.WIDTH)
        self.current_pos = self.board[0][midpoint]
        self.current_rotation = 0

        # QUESTION: Why doesn't it end early if it hits an existing block?
        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return TetrisEnv.TETRIMINOS[self.current_piece][self.current_rotation]


    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = TetrisEnv.MAP_PLAYER
        return board


    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score

    def _check_collision(self, piece, pos):
        '''
        Check if there is a collision between the current piece and the board
        Iterates over each coordinate in piece
        Aligns the piece coordinates with board
        Checks if piece is within the width
        Checks if piece is within height
        Checks if piece is colliding with an existing block
        If true on any, will return true, and will return false for no collision
        '''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= TetrisEnv.WIDTH \
                    or y < 0 or y >= TetrisEnv.HEIGHT \
                    or self.board[y][x] == TetrisEnv.EXISTING_BLOCK:
                return True
        return False


    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r
    

    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''        
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = TetrisEnv.EXISTING_BLOCK
        return board


    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == TetrisEnv.WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(TetrisEnv.WIDTH)])
        return len(lines_to_clear), board


    def _number_of_holes(self, board):
        '''Number of holes in the board (empty squares with at least one block above it)'''
        holes = 0

        for col in zip(*board):
            i = 0
            while i < TetrisEnv.HEIGHT and col[i] != TetrisEnv.EXISTING_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == TetrisEnv.MAP_EMPTY])

        return holes


    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < TetrisEnv.HEIGHT and col[i] != TetrisEnv.EXISTING_BLOCK:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness


    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = TetrisEnv.HEIGHT

        for col in zip(*board):
            i = 0
            while i < TetrisEnv.HEIGHT and col[i] == TetrisEnv.MAP_EMPTY:
                i += 1
            height = TetrisEnv.HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height


    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)

        # QUESTION: Additional Properties that can be left out 
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)

        return [lines, holes, total_bumpiness, sum_height]


    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = TetrisEnv.TETRIMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, TetrisEnv.WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states


    def get_state_size(self):
        '''Size of the state'''
        return 4


    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score        
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        score = 1 + (lines_cleared ** 2) * TetrisEnv.WIDTH
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over
    
    def render(self):
        '''Renders the current board'''
        img = [TetrisEnv.COLOURS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(TetrisEnv.HEIGHT, TetrisEnv.WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((TetrisEnv.WIDTH * 25, TetrisEnv.HEIGHT * 25))
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)