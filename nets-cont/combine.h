
# include "vector.h"
# include "matrix.h"


void mul(vec_t out, mat_t mat, vec_t vec) {
    assert(out.len == mat.h);
    assert(vec.len == mat.w);

    real_t * curr = mat.ent;
    
    for (int32_t i = 0; i < mat.h; ++i) {
    
        real_t sum = 0.0;
        for (int32_t i = 0; i < vec.len; ++i)
            sum += vec.ent[i] * curr[i];
        
        out.ent[i] = sum;
        curr += mat.w;
    }
}

void t_mul(vec_t out, mat_t mat, vec_t vec) {
    assert(out.len == mat.w);
    assert(vec.len == mat.h);
    real_t * curr = mat.ent;

    for (int32_t j = 0; j < out.len; ++j)
        out.ent[j] = 0.0;
    
    for (int32_t i = 0; i < vec.len; ++i) {
        for (int32_t j = 0; j < mat.w; ++j)
            out.ent[j] += vec.ent[i] * curr[j];
        curr += mat.w;
    }

}

void mat_add_scale_outer_inp(mat_t out, real_t keep, vec_t a, vec_t b) {
    assert (out.h == a.len);
    assert (out.w == b.len);
    
    real_t * curr = out.ent;
    for (int32_t i = 0; i < out.h; ++i) {
        real_t scalar = keep * a.ent[i];
        for (int32_t j = 0; j < out.w; ++j)
            curr[j] += scalar * b.ent[j];
        curr += out.w;
    }
}


void vec_add_scale_inp(vec_t out, real_t keep, vec_t inp) {
    assert (out.len == inp.len);
    for (int32_t i = 0; i < out.len; ++i) {
        out.ent[i] += keep * inp.ent[i];
    }
}


real_t approx_norm_rand() {
    return rand() / rand_max + rand() / rand_max + rand() / rand_max
         + rand() / rand_max + rand() / rand_max + rand() / rand_max
         + rand() / rand_max + rand() / rand_max + rand() / rand_max
         + rand() / rand_max + rand() / rand_max + rand() / rand_max
         - 6.0;
}

void xav_init(real_t * ent, real_t var, int64_t len) {
    for (int64_t i = 0; i < len; ++i) {
        ent[i] = approx_norm_rand() * var; 
    }
}

