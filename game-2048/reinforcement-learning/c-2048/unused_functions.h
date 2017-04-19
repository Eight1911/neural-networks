
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


static inline bool compressible(char const item_0, const char item_1, char const item_2, char const item_3) {
	char new[5] = {item_0, item_1, item_2, item_3, 0};
	char const * const end = new + 4;
	char * first = new;
	// check whether anything was moved
	while (*first) ++first; ++first;
	while (first < end) { if (*first) return true; ++first; }

	first = new;
	char * second = new + 1;
	
	if (!*second) return false;
	if (*first == *second) return true;

	++first; ++second;
	if (!*second) return false;
	if (*first == *second) return true;

	++first; ++second;
	if (!*second) return false;
	if (*first == *second) return true;

	return false;
}



static inline bool leftability(char const chain[16]) {
	return (compressible(chain[ 0], chain[ 1], chain[ 2], chain[ 3])
		 || compressible(chain[ 4], chain[ 5], chain[ 6], chain[ 7])
		 || compressible(chain[ 8], chain[ 9], chain[10], chain[11])
		 || compressible(chain[12], chain[13], chain[14], chain[15]));
}

static inline bool rightability(char const chain[16]) {
	return (compressible(chain[ 3], chain[ 2], chain[ 1], chain[ 0])
		 || compressible(chain[ 7], chain[ 6], chain[ 5], chain[ 4])
		 || compressible(chain[11], chain[10], chain[ 9], chain[ 8])
		 || compressible(chain[15], chain[14], chain[13], chain[12]));
}

static inline bool uppability(char const chain[16]) {
	return (compressible(chain[ 0], chain[ 4], chain[ 8], chain[12])
		 || compressible(chain[ 1], chain[ 5], chain[ 9], chain[13])
		 || compressible(chain[ 2], chain[ 6], chain[10], chain[14])
		 || compressible(chain[ 3], chain[ 7], chain[11], chain[15]));
}

static inline bool downability(char const chain[16]) {
	return (compressible(chain[15], chain[11], chain[ 7], chain[ 3])
		 || compressible(chain[14], chain[10], chain[ 6], chain[ 2])
		 || compressible(chain[13], chain[ 9], chain[ 5], chain[ 1])
		 || compressible(chain[12], chain[ 8], chain[ 4], chain[ 0]));
}



