#ifndef NURAL_NET
#define NEURAL_NET

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nnMath.h"
#include <time.h>


typedef struct
{
    matrix_t weights;
    vector_t biases;
    vector_t weighted_outputs;
    vector_t activated_outputs;
    vector_t error;
    int length;
} layer_t;

typedef struct neuralnet
{
    layer_t *layers;
    int num_layers;
} neural_net_t;

void print_matrix(matrix_t *mat);
void print_vector(vector_t *vec);

neural_net_t allocate_neural_net(int, int*);

layer_t init_layer(int length, int previous_layer_length);

void free_layer(layer_t *layer);

void free_network(neural_net_t *);

void feed_forward(layer_t *current_layer, layer_t *previous_layer);

void forward_pass(neural_net_t *network);

void loss(vector_t *, vector_t*);

void backward_pass(neural_net_t *network, vector_t *expected_outputs);

void train(neural_net_t *network, matrix_t *inputs, matrix_t *expected_outputs,
           int epochs, int batch_size, double learning_rate,
           matrix_t *test_inputs, matrix_t *test_expected_outputs);

void* train_t(void* args_t);

void update_weights(neural_net_t *network, matrix_t *temp_weights, int batch_size, double learning_rate);

void test(neural_net_t *network, matrix_t *inputs, matrix_t *expected_outputs);

void* test_t(void* args_t);

void update_temp_weights(matrix_t *temp_weights, neural_net_t *network, double learning_rate);

void update_temp_biases(vector_t *temp_biases, neural_net_t *network, double learning_rate);

void update_biases(neural_net_t *network, vector_t *temp_biases, int batch_size, double learning_rate);

#endif