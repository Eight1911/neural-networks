
# include "header.h"


typedef struct mat mat_t;

struct mat {
    real_t * ent;
    const int32_t w;
    const int32_t h;
};


void mat_print(mat_t mat) {
    real_t * curr = mat.ent;
    
    for (int32_t i = 0; i < mat.h; ++i) {
        printf("row [");
        for (int32_t i = 0; i < mat.w; ++i) {
            printf("%.5Lf, ", curr[i]);
        }
        printf("]\n");
        curr += mat.w;
    }
}

mat_t zero_mat(int32_t w, int32_t h) {
    return (mat_t) {
        .ent = (real_t *) calloc(h * w, sizeof(real_t)),
        .h = h,
        .w = w
    };
}


void mat_scale_inp(mat_t out, real_t scale) {
    int64_t end = out.w * out.h;
    for (int32_t i = 0; i < end; ++i)
        out.ent[i] *= scale;
}

void mat_add(mat_t out, mat_t a, mat_t b) {
    assert (out.h == a.h);
    assert (out.w == a.w);
    
    assert (out.h == b.h);
    assert (out.w == b.w);
    
    int64_t end = out.w * out.h;
    for (int32_t i = 0; i < end; ++i) {
        out.ent[i] = a.ent[i] + b.ent[i];
    }
}


void mat_mul(mat_t out, mat_t a, mat_t b) {
    assert (out.h == a.h);
    assert (out.w == a.w);
    
    assert (out.h == b.h);
    assert (out.w == b.w);
    
    int64_t end = out.w * out.h;
    for (int32_t i = 0; i < end; ++i) {
        out.ent[i] = a.ent[i] * b.ent[i];
    }
}

void mat_add_inp(mat_t out, mat_t inp) {
    assert (out.h == inp.h);
    assert (out.w == inp.w);
    
    int64_t end = out.w * out.h;
    for (int32_t i = 0; i < end; ++i) {
        out.ent[i] += inp.ent[i];
    }
}


void mat_mul_inp(mat_t out, mat_t inp) {
    assert (out.h == inp.h);
    assert (out.w == inp.w);
    
    int64_t end = out.w * out.h;
    for (int32_t i = 0; i < end; ++i) {
        out.ent[i] *= inp.ent[i];
    }
}


