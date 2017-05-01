# include "2048.h"
# include <stdbool.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <assert.h>


onionboard checkless_cycle(onionboard chain, onionboard (*direction) (onionboard));

static inline
onionboard checkless_left(register onionboard chain) {
	return (onionboard) {.rows = (rows) { 
		compress(chain.board.t__0, chain.board.t__1, chain.board.t__2, chain.board.t__3),
		compress(chain.board.t__4, chain.board.t__5, chain.board.t__6, chain.board.t__7),
		compress(chain.board.t__8, chain.board.t__9, chain.board.t_10, chain.board.t_11),
		compress(chain.board.t_12, chain.board.t_13, chain.board.t_14, chain.board.t_15)
	}};
}


static inline
onionboard checkless_right(register onionboard chain) {
	return (onionboard) {.rows = (rows) { 
		reversecompress(chain.board.t__0, chain.board.t__1, chain.board.t__2, chain.board.t__3),
		reversecompress(chain.board.t__4, chain.board.t__5, chain.board.t__6, chain.board.t__7),
		reversecompress(chain.board.t__8, chain.board.t__9, chain.board.t_10, chain.board.t_11),
		reversecompress(chain.board.t_12, chain.board.t_13, chain.board.t_14, chain.board.t_15)
	}};
}



static inline
onionboard checkless_up(register onionboard chain) {
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
	return chain;
}

static inline
onionboard checkless_down(register onionboard chain) {
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
	return chain;
}
onionboard checkless_cycle(onionboard chain, onionboard (*direction) (onionboard)) {
	chain = add(direction(chain));
	print(chain);
	return chain;
	//
}




