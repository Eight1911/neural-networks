

# include "header.h"


typedef struct vec vec_t;

struct vec {
    real_t * ent;
    const int32_t len;
};

void vec_print(vec_t vec) {
    printf("vec [");
    for (int32_t i = 0; i < vec.len; ++i) {
        printf("%.5Lf, ", vec.ent[i]);
    }
    printf("]\n");
}

real_t maximum(real_t a, real_t b) {
    return a > b ? a : b;
}

void copy(vec_t out, vec_t inp) {
    assert(out.len == inp.len);
    
    for (int32_t i = 0; i < out.len; ++i) {
        out.ent[i] = inp.ent[i];
    }
}
vec_t zero_vec(int32_t len) {
    return (vec_t) {
        .ent = (real_t *) calloc(len, sizeof(real_t)),
        .len = len
    };
}





void vec_scale_inp(vec_t out, real_t scale) {
    
    for (int32_t i = 0; i < out.len; ++i)
        out.ent[i] *= scale;
}


void vec_add(vec_t out, vec_t a, vec_t b) {
    assert (out.len == b.len);
    assert (out.len == a.len);
    
    for (int32_t i = 0; i < out.len; ++i)
        out.ent[i] = b.ent[i] + a.ent[i];
}


void vec_diff(vec_t out, vec_t a, vec_t b) {
    assert (out.len == b.len);
    assert (out.len == a.len);
    
    for (int32_t i = 0; i < out.len; ++i)
        out.ent[i] = b.ent[i] - a.ent[i];
}


void vec_mul(vec_t out, vec_t a, vec_t b) {
    assert(out.len == a.len);
    assert(out.len == b.len);

    for (int32_t i = 0; i < out.len; ++i) {
        out.ent[i] = a.ent[i] * b.ent[i];
    }
}


void vec_add_inp(vec_t out, vec_t inp) {
    assert (out.len == inp.len);
    
    for (int32_t i = 0; i < out.len; ++i)
        out.ent[i] += inp.ent[i];
}

void vec_mul_inp(vec_t out, vec_t inp) {
    assert (out.len == inp.len);
    
    for (int32_t i = 0; i < out.len; ++i)
        out.ent[i] *= inp.ent[i];
}

void vec_mul_prime(vec_t out, vec_t vec, vec_t inp) {
    assert(out.len == vec.len);
    assert(out.len == inp.len);

    for (int32_t i = 0; i < out.len; ++i) {
        out.ent[i] = vec.ent[i] > 0.0 ? inp.ent[i] : inp.ent[i]*leak;
    }
}

void vec_relu_inp(vec_t out) {
    
    for (int32_t i = 0; i < out.len; ++i)
        out.ent[i] = maximum(leak * out.ent[i], out.ent[i]);
}



void vec_prime_inp(vec_t out, vec_t inp) {
    for (int32_t i = 0; i < out.len; ++i)
        out.ent[i] = inp.ent[i] > 0 ? 1 : leak;

}


real_t norm2(vec_t inp) {
    real_t sum = 0.0;
    for (int32_t i = 0; i < inp.len; ++i) {
        sum += inp.ent[i] * inp.ent[i];
    }
    return sum;
} 

