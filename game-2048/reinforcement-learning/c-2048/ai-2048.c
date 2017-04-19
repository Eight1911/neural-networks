

# include "unchecked_functions.h"

# include <stdbool.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <math.h>

typedef onionboard (*move_ptr)(onionboard);


float benchmark(void);
uint16_t ai_play(register uint8_t const quickness);
uint32_t fullscore(onionboard const chain, uint16_t const moves);
double bestonion(register onionboard const chain, uint8_t const depth);
move_ptr optimise(register onionboard const chain, uint8_t const depth);


static inline
uint32_t absolute(register int32_t integer) {
	return (integer < 0) ? -integer : integer;
}


static inline
uint32_t range(
	register uint16_t const a, register uint16_t const b,
	register uint16_t const c, register uint16_t const d ) {
	register uint16_t min, max;
	if (a < b) {
		min = a; max = b;
	} else {
		min = b; max = a;
	}
	if (c < d) {
		if (d > max) { max = d; }
		if (c < min) { min = c; }
		return max - min;
	} else {
		if (c > max) { max = c; }
		if (d < min) { min = d; }
		return max - min;
	}
}

static inline
uint32_t max8(
	register uint32_t const a, register uint32_t const b,
	register uint32_t const c, register uint32_t const d,
	register uint32_t const e, register uint32_t const f,
	register uint32_t const g, register uint32_t const h ) {
	register uint32_t max = a;
	if (max < b) { max = b; }
	if (max < c) { max = c; }
	if (max < d) { max = d; }
	if (max < e) { max = e; }
	if (max < f) { max = f; }
	if (max < g) { max = g; }
	if (max < h) { max = h; }
	return max;
}

static inline
double desirability(register onionboard const chain) {
	#define DIFF(a, b, c, d) absolute((a)-(b)) + absolute((b)-(c)) + absolute((c)-(d))
	register uint16_t const scale__0 = (1 << chain.board.t__0);
	register uint16_t const scale__1 = (1 << chain.board.t__1);
	register uint16_t const scale__2 = (1 << chain.board.t__2);
	register uint16_t const scale__3 = (1 << chain.board.t__3);
	register uint16_t const scale__4 = (1 << chain.board.t__4);
	register uint16_t const scale__5 = (1 << chain.board.t__5);
	register uint16_t const scale__6 = (1 << chain.board.t__6);
	register uint16_t const scale__7 = (1 << chain.board.t__7);
	register uint16_t const scale__8 = (1 << chain.board.t__8);
	register uint16_t const scale__9 = (1 << chain.board.t__9);
	register uint16_t const scale_10 = (1 << chain.board.t_10);
	register uint16_t const scale_11 = (1 << chain.board.t_11);
	register uint16_t const scale_12 = (1 << chain.board.t_12);
	register uint16_t const scale_13 = (1 << chain.board.t_13);
	register uint16_t const scale_14 = (1 << chain.board.t_14);
	register uint16_t const scale_15 = (1 << chain.board.t_15);
	register uint32_t const sum_row_0 = scale__0 + scale__1 + scale__2 + scale__3;
	register uint32_t const sum_row_1 = scale__4 + scale__5 + scale__6 + scale__7;
	register uint32_t const sum_row_2 = scale__8 + scale__9 + scale_10 + scale_11;
	register uint32_t const sum_row_3 = scale_12 + scale_13 + scale_14 + scale_15;
	register uint32_t const maximum_row = max8(
		sum_row_0, sum_row_1, sum_row_2, sum_row_3,
		scale__0 + scale__4 + scale__8 + scale_12,
		scale__1 + scale__5 + scale__9 + scale_13,
		scale__2 + scale__6 + scale_10 + scale_14,
		scale__3 + scale__7 + scale_11 + scale_15);

	register uint32_t const total =
		  DIFF(scale__0, scale__1, scale__2, scale__3)
		+ DIFF(scale__4, scale__5, scale__6, scale__7)
		+ DIFF(scale__8, scale__9, scale_10, scale_11)
		+ DIFF(scale_12, scale_13, scale_14, scale_15)

		+ DIFF(scale__0, scale__4, scale__8, scale_12)
		+ DIFF(scale__1, scale__5, scale__9, scale_13)
		+ DIFF(scale__2, scale__6, scale_10, scale_14)
		+ DIFF(scale__3, scale__7, scale_11, scale_15);

	register uint32_t const tdiff =
		  range(scale__0, scale__1, scale__2, scale__3)
		+ range(scale__4, scale__5, scale__6, scale__7)
		+ range(scale__8, scale__9, scale_10, scale_11)
		+ range(scale_12, scale_13, scale_14, scale_15)

		+ range(scale__0, scale__4, scale__8, scale_12)
		+ range(scale__1, scale__5, scale__9, scale_13)
		+ range(scale__2, scale__6, scale_10, scale_14)
		+ range(scale__3, scale__7, scale_11, scale_15);
	register double const concente = maximum_row/(double)(sum_row_0 + sum_row_1 + sum_row_2 + sum_row_3);
	register double const monotone = tdiff / (double) total;
	return pow(pow(concente, 0.9) * monotone, 1.2);
	#undef DIFF

}

double meanonion(register onionboard chain, uint8_t const depth);
// iterates over possibilities before
// passing their scores to the meanonion function
// returns the highest score
// stops when depth == 0
double bestonion(register onionboard const chain, uint8_t const depth) {
	register double max_score = -100000.0;
	register double score;
	onionboard temp;
	if (depth) {
		temp = left(chain);
		if (temp.integer) {
			max_score = meanonion(temp, depth);
		}
		temp = up(chain);
		if (temp.integer) {
			score = meanonion(temp, depth);
			if (score > max_score) { max_score = score; }
		}
		temp = down(chain);
		if (temp.integer) {
			score = meanonion(temp, depth);
			if (score > max_score) { max_score = score; }
		}
		temp = right(chain);
		if (temp.integer) {
			score = meanonion(temp, depth);
			if (score > max_score) { max_score = score; }
		}
		return max_score;
	} else {
		temp = left(chain);
		if (temp.integer) {
			max_score = desirability(temp);
		}
		temp = up(chain);
		if (temp.integer) {
			score = desirability(temp);
			if (score > max_score) { max_score = score; }
		}
		temp = down(chain);
		if (temp.integer) {
			score = desirability(temp);
			if (score > max_score) { max_score = score; }
		}
		temp = right(chain);
		if (temp.integer) {
			score = desirability(temp);
			if (score > max_score) { max_score = score; }
		}
		return max_score;
	}
}

// for each empty tile. add a new tile and check how desirable
// the board is. reports the average score of all the boards
// loops unrolled for efficiency.
double meanonion(register onionboard chain, uint8_t const depth) {
	register uint8_t const new_depth = depth - 1;
	register double score_2 = 0.0;
	register double score_4 = 0.0;
	register uint8_t len = 0;

	if (! chain.board.t__0) {
		++chain.board.t__0; score_2 += bestonion(chain, new_depth);
		++chain.board.t__0; score_4 += bestonion(chain, new_depth);
		  chain.board.t__0 = 0; ++len;
	}
	if (! chain.board.t__1) {
		++chain.board.t__1; score_2 += bestonion(chain, new_depth);
		++chain.board.t__1; score_4 += bestonion(chain, new_depth);
		  chain.board.t__1 = 0; ++len;
	}
	if (! chain.board.t__2) {
		++chain.board.t__2; score_2 += bestonion(chain, new_depth);
		++chain.board.t__2; score_4 += bestonion(chain, new_depth);
		  chain.board.t__2 = 0; ++len;
	}
	if (! chain.board.t__3) {
		++chain.board.t__3; score_2 += bestonion(chain, new_depth);
		++chain.board.t__3; score_4 += bestonion(chain, new_depth);
		  chain.board.t__3 = 0; ++len;
	}
	if (! chain.board.t__4) {
		++chain.board.t__4; score_2 += bestonion(chain, new_depth);
		++chain.board.t__4; score_4 += bestonion(chain, new_depth);
		  chain.board.t__4 = 0; ++len;
	}
	if (! chain.board.t__5) {
		++chain.board.t__5; score_2 += bestonion(chain, new_depth);
		++chain.board.t__5; score_4 += bestonion(chain, new_depth);
		  chain.board.t__5 = 0; ++len;
	}
	if (! chain.board.t__6) {
		++chain.board.t__6; score_2 += bestonion(chain, new_depth);
		++chain.board.t__6; score_4 += bestonion(chain, new_depth);
		  chain.board.t__6 = 0; ++len;
	}
	if (! chain.board.t__7) {
		++chain.board.t__7; score_2 += bestonion(chain, new_depth);
		++chain.board.t__7; score_4 += bestonion(chain, new_depth);
		  chain.board.t__7 = 0; ++len;
	}
	if (! chain.board.t__8) {
		++chain.board.t__8; score_2 += bestonion(chain, new_depth);
		++chain.board.t__8; score_4 += bestonion(chain, new_depth);
		  chain.board.t__8 = 0; ++len;
	}

	if (! chain.board.t__9) {
		++chain.board.t__9; score_2 += bestonion(chain, new_depth);
		++chain.board.t__9; score_4 += bestonion(chain, new_depth);
		  chain.board.t__9 = 0; ++len;
	}
	if (! chain.board.t_10) {
		++chain.board.t_10; score_2 += bestonion(chain, new_depth);
		++chain.board.t_10; score_4 += bestonion(chain, new_depth);
		  chain.board.t_10 = 0; ++len;
	}
	if (! chain.board.t_11) {
		++chain.board.t_11; score_2 += bestonion(chain, new_depth);
		++chain.board.t_11; score_4 += bestonion(chain, new_depth);
		  chain.board.t_11 = 0; ++len;
	}
	if (! chain.board.t_12) {
		++chain.board.t_12; score_2 += bestonion(chain, new_depth);
		++chain.board.t_12; score_4 += bestonion(chain, new_depth);
		  chain.board.t_12 = 0; ++len;
	}
	if (! chain.board.t_13) {
		++chain.board.t_13; score_2 += bestonion(chain, new_depth);
		++chain.board.t_13; score_4 += bestonion(chain, new_depth);
		  chain.board.t_13 = 0; ++len;
	}
	if (! chain.board.t_14) {
		++chain.board.t_14; score_2 += bestonion(chain, new_depth);
		++chain.board.t_14; score_4 += bestonion(chain, new_depth);
		  chain.board.t_14 = 0; ++len;
	}
	if (! chain.board.t_15) {
		++chain.board.t_15; score_2 += bestonion(chain, new_depth);
		++chain.board.t_15; score_4 += bestonion(chain, new_depth);
		  chain.board.t_15 = 0; ++len;
	}
	// weighted score based on expectations
	return (0.9 * score_2 + 0.1 * score_4) / len;
	// since the game has not ended and a move had just been made
	// by either the function 'best' or 'optimise', the array 'chain' is
	// guaranteed to have at least one zero.

	// so length is never zero and we don't need to worry
	// about zero division error in the return statement
}


// takes a pointer to a board and yields a direction
// that would give the most desirable board after
// a depth number moves
move_ptr optimise(register onionboard const chain, uint8_t const depth) {
	register double score;
	register double max_score = -100000.0;
	register move_ptr best_func = NULL;
	register onionboard temp;

	temp = left(chain);
	if (temp.integer) {
		score = meanonion(temp, depth);
		if (score > max_score) {
			max_score = score;
			best_func = &checkless_left;
		}
	}

	temp = down(chain);
	if (temp.integer) {
		max_score = meanonion(temp, depth);
		best_func = &checkless_down;
	}

	temp = up(chain);
	if (temp.integer) {
		score = meanonion(temp, depth);
		if (score > max_score) {
			max_score = score;
			best_func = &checkless_up;
		}
	}

	temp = right(chain);
	if (temp.integer) {
		score = meanonion(temp, depth);
		if (score > max_score) {
			// max_score = score;
			best_func = &checkless_right;
		}
	}
	return best_func;
}

// calculates the 2048 score
// given a board and the number of times
// a tile was added to it
// this function assumes that the board
// has no empty tile in it
uint32_t fullscore( onionboard const chain, uint16_t const moves ) {
	uint32_t score = 0;
	uint16_t sum = 0;
	score += (chain.board.t__0 - 1) * (1 << chain.board.t__0); sum += (1 << chain.board.t__0);
	score += (chain.board.t__1 - 1) * (1 << chain.board.t__1); sum += (1 << chain.board.t__1);
	score += (chain.board.t__2 - 1) * (1 << chain.board.t__2); sum += (1 << chain.board.t__2);
	score += (chain.board.t__3 - 1) * (1 << chain.board.t__3); sum += (1 << chain.board.t__3);
	score += (chain.board.t__4 - 1) * (1 << chain.board.t__4); sum += (1 << chain.board.t__4);
	score += (chain.board.t__5 - 1) * (1 << chain.board.t__5); sum += (1 << chain.board.t__5);
	score += (chain.board.t__6 - 1) * (1 << chain.board.t__6); sum += (1 << chain.board.t__6);
	score += (chain.board.t__7 - 1) * (1 << chain.board.t__7); sum += (1 << chain.board.t__7);
	score += (chain.board.t__8 - 1) * (1 << chain.board.t__8); sum += (1 << chain.board.t__8);
	score += (chain.board.t__9 - 1) * (1 << chain.board.t__9); sum += (1 << chain.board.t__9);
	score += (chain.board.t_10 - 1) * (1 << chain.board.t_10); sum += (1 << chain.board.t_10);
	score += (chain.board.t_11 - 1) * (1 << chain.board.t_11); sum += (1 << chain.board.t_11);
	score += (chain.board.t_12 - 1) * (1 << chain.board.t_12); sum += (1 << chain.board.t_12);
	score += (chain.board.t_13 - 1) * (1 << chain.board.t_13); sum += (1 << chain.board.t_13);
	score += (chain.board.t_14 - 1) * (1 << chain.board.t_14); sum += (1 << chain.board.t_14);
	score += (chain.board.t_15 - 1) * (1 << chain.board.t_15); sum += (1 << chain.board.t_15);
	score -= ((sum - 2*moves) << 1);
	return score;
}


// let the ai play the game for you
// quickness defines how much time the
// ai takes to evaluate each board
// the greater the quickness,
// the higher the amount of time
uint16_t ai_play(register uint8_t const quickness) {
	register onionboard chain = {.integer = 0};
	register onionboard (*move)(onionboard);
	register uint16_t turn = 2;

	srandom(time(NULL));	 	// initialize random number generator using
								// current time. granularity is in seconds.
	chain = add(chain);			// add the first element.
	print(chain);
	getchar();
	register uint8_t zero;
	register uint8_t const standard = quickness + 4;
	for (;;++turn) {
		zero = countzero(chain);
		move = NULL;
		// spends more time checking if
		// it is more likely to be stuck
		if (zero < quickness)
			{ move = optimise(chain, 4); }
		else if (zero < standard)
			{ move = optimise(chain, 3); }
		else
			{ move = optimise(chain, 2); }
		printf( "\33[2K\r\033[A\r"
				"\33[2K\r\033[A\r"
				"\33[2K\r\033[A\r"
				"\33[2K\r\033[A\r"
				"\33[2K\r\033[A\r"
				"\33[2K\r\033[A\r");
		if (move) {


			chain = checkless_cycle(chain, move);
			printf("current turn: %i\n", turn);
		} else {
			print(chain);
			printf("current turn: %i\n", turn);
			printf("\nscore: %i\n", fullscore(chain, turn));
			printf("\nYOU LOST\n");
			return turn;
		}
	}
}


float benchmark(void) {
	float total_time;
	total_time  = clock() / (float) CLOCKS_PER_SEC;
	ai_play(3);
	total_time -= clock() / (float) CLOCKS_PER_SEC;
	printf("time benchmarked: %f\n", -total_time);
	return -total_time;
}

int main(void) {
	benchmark();
}
