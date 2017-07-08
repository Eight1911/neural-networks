# include <unistd.h>
# include "combine.h"


typedef struct net net_t;
typedef struct data data_t;
typedef struct train train_t;

typedef data_t (*gen_t) (net_t * );

struct net {
    mat_t * weight;
    vec_t * bias;
    int32_t depth;
};

struct train {
    net_t * net;
    real_t l_rate;
    real_t moment;
    
    mat_t * w_moment; // length n
    vec_t * b_moment; // length n
    vec_t * deltas; // length n
    vec_t * output; // length n + 1
    vec_t * errors; // length n + 1
};

struct data {
    vec_t * inp;
    vec_t * out;
    int32_t len;
};




mat_t new_mat(int32_t w, int32_t h) {
    real_t * ent = malloc(sizeof(real_t) * w * h);
    xav_init(ent, 2.0 / (w + h), w * h);
    return (mat_t) { 
        .ent = ent,
        .w = w,
        .h = h 
    };

}


vec_t new_vec(int32_t len) {
    return zero_vec(len);
}

net_t new_net(int32_t * dim, int32_t depth) {
    mat_t * ws = malloc(sizeof(mat_t) * depth);
    vec_t * bs = malloc(sizeof(vec_t) * depth);

    for (int i = 0; i < depth; ++i) {
        ws[i] = new_mat(dim[i], dim[i+1]);
        bs[i] = new_vec(dim[i+1]);
    }
    
    return (net_t) {
        .weight = ws,
        .bias = bs,
        .depth = depth
    };
}

train_t new_train(net_t * net, real_t l_rate, real_t momentum) {
    int32_t depth = net -> depth;
    mat_t * ws  = net -> weight;
    vec_t * bs  = net -> bias;
    mat_t * w_moment = malloc(sizeof(mat_t) * depth);
    vec_t * b_moment = malloc(sizeof(vec_t) * depth),
          * deltas   = malloc(sizeof(vec_t) * depth),
          * errors   = malloc(sizeof(vec_t) * (depth + 1)),
          * output   = malloc(sizeof(vec_t) * (depth + 1));
    
    
    output[0] = zero_vec(ws[0].w);
    errors[0] = zero_vec(ws[0].w);
    
    for (int32_t i = 0; i < depth; ++i) {
        mat_t w = ws[i];
        vec_t b = bs[i];
        w_moment[i] = zero_mat(w.w, w.h);
        b_moment[i] = zero_vec(b.len);
        deltas[i]   = zero_vec(b.len);
        output[i+1] = zero_vec(b.len);
        errors[i+1] = zero_vec(b.len);
        
        
    }

    return (train_t) {
        .net      = net,
        .l_rate   = l_rate,
        .moment   = momentum,
        .w_moment = w_moment,
        .b_moment = b_moment,
        .deltas   = deltas,
        .errors   = errors,
        .output   = output
    };
    
}



// assume output[0] == input
void push(train_t * train) {
    vec_t * output = train -> output,
          * bias   = train -> net -> bias;
    mat_t * weight = train -> net -> weight;
    int32_t depth = train -> net -> depth;
    
    for (int32_t i = 0; i < depth; ++i) {
        mul(output[i+1], weight[i], output[i]);
        vec_add_inp(output[i+1], bias[i]);
        vec_relu_inp(output[i+1]);
    }
}

// assume errors[n] == error
void pull(train_t * train) {
    vec_t * output = train -> output,
          * errors = train -> errors,
          * deltas = train -> deltas;
    mat_t * weight = train -> net -> weight;
    
    for (int32_t i = train -> net -> depth; i > 0; --i) {
        vec_mul_prime(deltas[i-1], output[i], errors[i]); 
        t_mul(errors[i-1], weight[i-1], deltas[i-1]);
    }
}

// update with nesterov momentum
void date(train_t * train) {
    
    net_t * net = train -> net;
    int32_t depth = net -> depth;
    real_t moment = train -> moment,
           l_rate = train -> l_rate;
    
    mat_t * w_moment = train -> w_moment,
          * weight   = net -> weight;
        
    vec_t * b_moment = train -> b_moment,
          * deltas   = train -> deltas,
          * output   = train -> output,
          * bias     = net -> bias;
    
    
    real_t keep = l_rate * (1 - moment);

    for (int32_t i = depth - 1; i--;) {
        mat_t w_m = w_moment[i];
        vec_t b_m = b_moment[i],
            delta = deltas[i];
        mat_scale_inp(w_m, moment); 
        // w_moment[i] += keep * outer(deltas[i], output[i])
        mat_add_scale_outer_inp(w_m, keep, delta, output[i]); 
        mat_add_inp(weight[i], w_m);
        
        vec_scale_inp(b_m, moment); 
        // b_moment[i] += keep * delta
        vec_add_scale_inp(b_m, keep, delta);
        vec_add_inp(bias[i], b_m);
        
    }
} 



void run(train_t * train, gen_t generate, int32_t count) {
    int32_t depth = train -> net -> depth;
    for (int index = 0; index < count; ++index) {
        data_t next = generate(train -> net);
        
    
            
        for (int i = 0; i < next.len; ++i) {
            copy(train -> output[0], next.inp[i]);
            push(train);
            vec_diff(train -> errors[depth], train -> output[depth], next.out[i]);
            pull(train);
            date(train);
            if (index % 10 == 0)
                printf("%.10Lf\n", norm2(train -> errors[depth]));
        }
    }
}

data_t initgen() {
    vec_t * inp = malloc(sizeof(vec_t) * 4);
    vec_t * out = malloc(sizeof(vec_t) * 4);

    for (int i = 0; i < 4; ++i) {
        real_t * inp_ent = malloc(sizeof(real_t) * 2) ;
        real_t * out_ent = malloc(sizeof(real_t) * 1); 
        inp_ent[0] = (real_t) (i % 2);
        inp_ent[1] =(real_t) (i / 2);
        out_ent[0] = (real_t) (i % 2 != i / 2);
        inp[i] = (vec_t) {.ent = inp_ent, .len = 2};
        out[i] = (vec_t) {.ent = out_ent, .len = 1};
    }
    
    return (data_t) {
        .len = 4,
        .inp = inp,
        .out = out
    };
}


data_t gen(net_t * net) {
    return initgen();
}

int main(void) {
    /*
    int32_t depth = 3;
    int32_t dim[5] = {2, 300, 300, 1};
    
    net_t net = new_net(dim, depth);
    train_t train = new_train(&net, 10, 0.95);
    
    
    run(&train, gen, 30000);
    
    */   
    
}