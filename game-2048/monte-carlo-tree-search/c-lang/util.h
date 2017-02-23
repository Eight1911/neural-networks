#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef unsigned char uint8_t;

typedef struct board {
	uint8_t t__0 : 4; uint8_t t__1 : 4;
	uint8_t t__2 : 4; uint8_t t__3 : 4;
	uint8_t t__4 : 4; uint8_t t__5 : 4;
	uint8_t t__6 : 4; uint8_t t__7 : 4;
	uint8_t t__8 : 4; uint8_t t__9 : 4;
	uint8_t t_10 : 4; uint8_t t_11 : 4;
	uint8_t t_12 : 4; uint8_t t_13 : 4;
	uint8_t t_14 : 4; uint8_t t_15 : 4;
} board;

typedef struct quart {
		uint8_t e_0 : 4; uint8_t e_1 : 4;
		uint8_t e_2 : 4; uint8_t e_3 : 4;
} quart ;

typedef union onionquart {
	quart quart;
	uint16_t integer : 16;
} onionquart;

typedef struct rows {
	onionquart row_0, row_1, row_2, row_3;
} rows;

typedef union onionboard {
	rows rows;
	board board;
	uint64_t integer : 64;
} onionboard;

onionboard add(register onionboard);
void print(onionboard const);



static inline
onionquart compress(
	register uint8_t row_0,
	register uint8_t row_1,
	register uint8_t row_2,
	register uint8_t row_3) {
	if (row_0) {
		if (row_1) {						// 1 1 1 1 // 1 1 1 0
			if (!row_2 && row_3) {			// 1 1 0 1
				row_2 = row_3;
				row_3 = 0;
			}	// else do nothing 			// 1 1 0 0
		} else {
			if (row_2) {
				if (row_3) { 				// 1 0 1 1
					row_1 = row_2;
					row_2 = row_3;
					row_3 = 0;
				} else { 					// 1 0 1 0
					row_1 = row_2;
					row_2 = 0;
				}
			} else {
				if (row_3) { 				// 1 0 0 1
					row_1 = row_3;
					row_3 = 0;
				} else { // no change		// 1 0 0 0
					return (onionquart) {.quart = (quart) {row_0, 0, 0, 0}};
				}
			}
		}
	} else {
		if (row_1) {
			if (row_2) {
				if (row_3) { 				// 0 1 1 1
					row_0 = row_1;
					row_1 = row_2;
					row_2 = row_3;
					row_3 = 0;
				} else {					// 0 1 1 0
					row_0 = row_1;
					row_1 = row_2;
					row_2 = 0;
				}
			} else {
				if (row_3) { 				// 0 1 0 1
					row_0 = row_1;
					row_1 = row_3;
					row_3 = 0;
				} else {					// 0 1 0 0
					return (onionquart) {.quart = (quart) {row_1, 0, 0, 0}};
				}
			}
		} else {
			if (row_2) {
				if (row_3) { 				// 0 0 1 1
					row_0 = row_2;
					row_1 = row_3;
					row_2 = 0;
					row_3 = 0;
				} else { 					// 0 0 1 0
					return (onionquart) {.quart = (quart) {row_2, 0, 0, 0}};
				}
			} else {
				if (row_3) { 				// 0 0 0 1
					return (onionquart) {.quart = (quart) {row_3, 0, 0, 0}};
				} else {					// 0 0 0 0
					return (onionquart) {.quart = (quart) {0, 0, 0, 0}};
				}
			}
		}
	}
	if (row_0 == row_1) {
		row_1 = row_2;
		row_2 = row_3;
		row_3 = 0;
		++row_0;
	}
	if (!row_2) { return (onionquart) {.quart = (quart) {row_0, row_1, 0, 0}}; }
	if (row_1 == row_2) {
		row_2 = row_3;
		row_3 = 0;
		++row_1;
	}
	if (!row_3) { return (onionquart) {.quart = (quart) {row_0, row_1, row_2, 0}}; }
	if (row_2 == row_3) {
		row_3 = 0;
		++row_2;
	}
	return (onionquart) {.quart = (quart) {row_0, row_1, row_2, row_3}};
}

static inline
onionquart reversecompress(
	register uint8_t row_3,
	register uint8_t row_2,
	register uint8_t row_1,
	register uint8_t row_0) {
	if (row_0) {
		if (row_1) {						// 1 1 1 1 // 1 1 1 0
			if (!row_2 && row_3) {			// 1 1 0 1
				row_2 = row_3;
				row_3 = 0;
			}	// else do nothing 			// 1 1 0 0
		} else {
			if (row_2) {
				if (row_3) { 				// 1 0 1 1
					row_1 = row_2;
					row_2 = row_3;
					row_3 = 0;
				} else { 					// 1 0 1 0
					row_1 = row_2;
					row_2 = 0;
				}
			} else {
				if (row_3) { 				// 1 0 0 1
					row_1 = row_3;
					row_3 = 0;
				} else { // no change		// 1 0 0 0
					return (onionquart) {.quart = (quart) {0, 0, 0, row_0}};
				}
			}
		}
	} else {
		if (row_1) {
			if (row_2) {
				if (row_3) { 				// 0 1 1 1
					row_0 = row_1;
					row_1 = row_2;
					row_2 = row_3;
					row_3 = 0;
				} else {					// 0 1 1 0
					row_0 = row_1;
					row_1 = row_2;
					row_2 = 0;
				}
			} else {
				if (row_3) { 				// 0 1 0 1
					row_0 = row_1;
					row_1 = row_3;
					row_3 = 0;
				} else {					// 0 1 0 0
					return (onionquart) {.quart = (quart) {0, 0, 0, row_1}};
				}
			}
		} else {
			if (row_2) {
				if (row_3) { 				// 0 0 1 1
					row_0 = row_2;
					row_1 = row_3;
					row_2 = 0;
					row_3 = 0;
				} else { 					// 0 0 1 0
					return (onionquart) {.quart = (quart) {0, 0, 0, row_2}};
				}
			} else {
				if (row_3) { 				// 0 0 0 1
					return (onionquart) {.quart = (quart) {0, 0, 0, row_3}};
				} else {					// 0 0 0 0
					return (onionquart) {.quart = (quart) {0, 0, 0, 0}};
				}
			}
		}
	}
	if (row_0 == row_1) {
		row_1 = row_2;
		row_2 = row_3;
		row_3 = 0;
		++row_0;
	}
	if (!row_2) { return (onionquart) {.quart = (quart) {0, 0, row_1, row_0}}; }
	if (row_1 == row_2) {
		row_2 = row_3;
		row_3 = 0;
		++row_1;
	}
	if (!row_3) { return (onionquart) {.quart = (quart) {0, row_2, row_1, row_0}}; }
	if (row_2 == row_3) {
		row_3 = 0;
		++row_2;
	}
	return (onionquart) {.quart = (quart) {row_3, row_2, row_1, row_0}};
}


static inline
onionboard left(register onionboard chain) {
	onionboard const copy = chain;
	chain.rows = (rows) {
		compress(chain.board.t__0, chain.board.t__1, chain.board.t__2, chain.board.t__3),
		compress(chain.board.t__4, chain.board.t__5, chain.board.t__6, chain.board.t__7),
		compress(chain.board.t__8, chain.board.t__9, chain.board.t_10, chain.board.t_11),
		compress(chain.board.t_12, chain.board.t_13, chain.board.t_14, chain.board.t_15)
	};
	if (copy.integer == chain.integer) { return (onionboard) {.integer = 0}; }
	return chain;
}


static inline
onionboard right(register onionboard chain) {
	onionboard const copy = chain;
	chain.rows = (rows) {
		reversecompress(chain.board.t__0, chain.board.t__1, chain.board.t__2, chain.board.t__3),
		reversecompress(chain.board.t__4, chain.board.t__5, chain.board.t__6, chain.board.t__7),
		reversecompress(chain.board.t__8, chain.board.t__9, chain.board.t_10, chain.board.t_11),
		reversecompress(chain.board.t_12, chain.board.t_13, chain.board.t_14, chain.board.t_15)
	};
	if (copy.integer == chain.integer) { return (onionboard) {.integer = 0}; }
	return chain;
}


static inline
onionboard up(register onionboard chain) {
	onionboard const copy = chain;
	register onionquart row;
	row = compress(chain.board.t__0, chain.board.t__4, chain.board.t__8, chain.board.t_12);
	chain.board.t__0 = row.quart.e_0;
	chain.board.t__4 = row.quart.e_1;
	chain.board.t__8 = row.quart.e_2;
	chain.board.t_12 = row.quart.e_3;
	row = compress(chain.board.t__1, chain.board.t__5, chain.board.t__9, chain.board.t_13);
	chain.board.t__1 = row.quart.e_0;
	chain.board.t__5 = row.quart.e_1;
	chain.board.t__9 = row.quart.e_2;
	chain.board.t_13 = row.quart.e_3;
	row = compress(chain.board.t__2, chain.board.t__6, chain.board.t_10, chain.board.t_14);
	chain.board.t__2 = row.quart.e_0;
	chain.board.t__6 = row.quart.e_1;
	chain.board.t_10 = row.quart.e_2;
	chain.board.t_14 = row.quart.e_3;
	row = compress(chain.board.t__3, chain.board.t__7, chain.board.t_11, chain.board.t_15);
	chain.board.t__3 = row.quart.e_0;
	chain.board.t__7 = row.quart.e_1;
	chain.board.t_11 = row.quart.e_2;
	chain.board.t_15 = row.quart.e_3;
	if (copy.integer == chain.integer) { return (onionboard) {.integer = 0}; }
	return chain;
}
static inline
onionboard down(register onionboard chain) {
	onionboard const copy = chain;
	register onionquart row;
	row = compress(chain.board.t_12, chain.board.t__8, chain.board.t__4, chain.board.t__0);
	chain.board.t__0 = row.quart.e_3;
	chain.board.t__4 = row.quart.e_2;
	chain.board.t__8 = row.quart.e_1;
	chain.board.t_12 = row.quart.e_0;
	row = compress(chain.board.t_13, chain.board.t__9, chain.board.t__5, chain.board.t__1);
	chain.board.t__1 = row.quart.e_3;
	chain.board.t__5 = row.quart.e_2;
	chain.board.t__9 = row.quart.e_1;
	chain.board.t_13 = row.quart.e_0;
	row = compress(chain.board.t_14, chain.board.t_10, chain.board.t__6, chain.board.t__2);
	chain.board.t__2 = row.quart.e_3;
	chain.board.t__6 = row.quart.e_2;
	chain.board.t_10 = row.quart.e_1;
	chain.board.t_14 = row.quart.e_0;
	row = compress(chain.board.t_15, chain.board.t_11, chain.board.t__7, chain.board.t__3);
	chain.board.t__3 = row.quart.e_3;
	chain.board.t__7 = row.quart.e_2;
	chain.board.t_11 = row.quart.e_1;
	chain.board.t_15 = row.quart.e_0;
	if (copy.integer == chain.integer) { return (onionboard) {.integer = 0}; }
	return chain;
}

static inline
uint8_t countzero(register onionboard const chain) {
	register uint8_t zero = 0;
	if (!chain.board.t__0) ++zero;
	if (!chain.board.t__1) ++zero;
	if (!chain.board.t__2) ++zero;
	if (!chain.board.t__3) ++zero;
	if (!chain.board.t__4) ++zero;
	if (!chain.board.t__5) ++zero;
	if (!chain.board.t__6) ++zero;
	if (!chain.board.t__7) ++zero;
	if (!chain.board.t__8) ++zero;
	if (!chain.board.t__9) ++zero;
	if (!chain.board.t_10) ++zero;
	if (!chain.board.t_11) ++zero;
	if (!chain.board.t_12) ++zero;
	if (!chain.board.t_13) ++zero;
	if (!chain.board.t_14) ++zero;
	if (!chain.board.t_15) ++zero;
	return zero;
}


onionboard add(register onionboard chain) {
	uint8_t zero = countzero(chain);
	long randnum = random() % (zero*10);
	uint8_t addvalue = randnum / zero;
	register uint8_t position = randnum / 10;
	register uint8_t pos = 0;
	if (!chain.board.t__0) { if (pos == position) { chain.board.t__0 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__1) { if (pos == position) { chain.board.t__1 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__2) { if (pos == position) { chain.board.t__2 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__3) { if (pos == position) { chain.board.t__3 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__4) { if (pos == position) { chain.board.t__4 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__5) { if (pos == position) { chain.board.t__5 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__6) { if (pos == position) { chain.board.t__6 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__7) { if (pos == position) { chain.board.t__7 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__8) { if (pos == position) { chain.board.t__8 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t__9) { if (pos == position) { chain.board.t__9 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t_10) { if (pos == position) { chain.board.t_10 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t_11) { if (pos == position) { chain.board.t_11 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t_12) { if (pos == position) { chain.board.t_12 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t_13) { if (pos == position) { chain.board.t_13 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t_14) { if (pos == position) { chain.board.t_14 = addvalue?1:2 ; return chain; } ++pos; }
	if (!chain.board.t_15) { if (pos == position) { chain.board.t_15 = addvalue?1:2 ; return chain; } ++pos; }
	return chain;
	// there is a 1 in 8 chance that
	// a four tile rather than a two tile
	// is spawned
}


void print(onionboard const chain) {
	// printf("\33[2K\r\033[A\r");
	// delete last line and move cursor up one line
	//*
	printf( "\33[2K\r\033[A\r"
			"\33[2K\r\033[A\r"
			"\33[2K\r\033[A\r"
			"\33[2K\r\033[A\r"
			"\33[2K\r\033[A\r"
			"\33[2K\r\033[A\r"
			); // 5 times
	//*/
	printf("\n");
	printf("|");
	chain.board.t__0 ? printf("%6i|", 1 << chain.board.t__0) : printf("      |");
	chain.board.t__1 ? printf("%6i|", 1 << chain.board.t__1) : printf("      |");
	chain.board.t__2 ? printf("%6i|", 1 << chain.board.t__2) : printf("      |");
	chain.board.t__3 ? printf("%6i|", 1 << chain.board.t__3) : printf("      |");
	printf("\n");
	printf("|");
	chain.board.t__4 ? printf("%6i|", 1 << chain.board.t__4) : printf("      |");
	chain.board.t__5 ? printf("%6i|", 1 << chain.board.t__5) : printf("      |");
	chain.board.t__6 ? printf("%6i|", 1 << chain.board.t__6) : printf("      |");
	chain.board.t__7 ? printf("%6i|", 1 << chain.board.t__7) : printf("      |");
	printf("\n");
	printf("|");
	chain.board.t__8 ? printf("%6i|", 1 << chain.board.t__8) : printf("      |");
	chain.board.t__9 ? printf("%6i|", 1 << chain.board.t__9) : printf("      |");
	chain.board.t_10 ? printf("%6i|", 1 << chain.board.t_10) : printf("      |");
	chain.board.t_11 ? printf("%6i|", 1 << chain.board.t_11) : printf("      |");
	printf("\n");
	printf("|");
	chain.board.t_12 ? printf("%6i|", 1 << chain.board.t_12) : printf("      |");
	chain.board.t_13 ? printf("%6i|", 1 << chain.board.t_13) : printf("      |");
	chain.board.t_14 ? printf("%6i|", 1 << chain.board.t_14) : printf("      |");
	chain.board.t_15 ? printf("%6i|", 1 << chain.board.t_15) : printf("      |");
	printf("\n");
	printf("\n");
}


/*/
static inline
onionboard cycle(onionboard chain, onionboard (*direction)(onionboard)) {
	//printf("\33[2K\r\033[A\r");
	onionboard new = direction(chain);
	if (!(new.integer)) return chain;
	chain = add(new);
	print(chain);
	return chain;
}


void play() {
	system("stty cbreak");		// change mode so that putunsigned char immediately
								// returns value after one unsigned character is pressed
	srandom(0);//time(NULL)); 	// initialize random number generator using
								//	current time. granularity is in seconds.

	onionboard chain = {.integer = 0};
	chain = add(chain);		// add the first element.
	print(chain);

	for (;;) {
		switch (getchar()) {
			case 'a': chain = cycle(chain, left);  break;
			case 's': chain = cycle(chain, down);  break;
			case 'd': chain = cycle(chain, right); break;
			case 'w': chain = cycle(chain, up);	   break;
		}
		printf("\33[2K\r");
	}
}
//*/
