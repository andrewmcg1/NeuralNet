#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef NN_MATH_H
#define NN_MATH_H

typedef struct
{
    float *arr;
    int row;
    int col;
} matrix_t;

typedef struct
{
    float *arr;
    int len;
} vector_t;

// Allocates space of a matrix and a vector
vector_t init_vector(int len);
matrix_t init_matrix(int row, int col);
float *allocate_vec_arr(int len);
float *allocate_mat_arr(int rows, int cols);
matrix_t *allocate_mat();
vector_t *allocate_vec();

// Frees memory for matrices and vectors
void free_vector(vector_t *vec);
void free_matrix(matrix_t *mat);

// Multiply a matrix with a vector
void multiply_mat_vec(vector_t *out, matrix_t *mat, vector_t *vec);
// Add two vectors
void add_vec(vector_t *out, vector_t *v1, vector_t *v2);
// Subtract two vectors
void subtract_vec(vector_t *out, vector_t *v1, vector_t *v2);

// Sigmoid of a single value
float sigmoid(float val);

// Sigmoid of a matrix
void sigmoid_mat(matrix_t *out, matrix_t *mat);
// Sigmoid derivative of a matrix
void dsigmoid(matrix_t *out, matrix_t *mat);

// Sigmoid of a vector
void sigmoid_vec(vector_t *out, vector_t *vec);
// Sigmoid derivative of a vector
void dsigmoid_vec(vector_t *out, vector_t *vec);

void hadamard_product(vector_t *out, vector_t *vec1, vector_t *vec2);

// Gets the error at the output layer
// last_layer_activations - output of last layer after sigmoid
// last_layer_weighted - output of last layer before sigmoid
void output_error(vector_t *out,
                  vector_t *expected_output,
                  vector_t *last_layer_activations,
                  vector_t *last_layer_weighted);

void transpose(matrix_t *out, matrix_t *mat);

// Gets the error of all layers except output
// next_layer_weights - weights of next layer
// next_layer_error - error of the next layer
// output of the current layer before sigmoid
void layer_error(vector_t *out,
                 matrix_t *next_layer_weights,
                 vector_t *next_layer_error,
                 vector_t *current_layer_weighted);

void multiply_vec_vec(matrix_t *out, vector_t *v1, vector_t *v2);

void scalar_multiply_mat(matrix_t *out, matrix_t *mat, float scalar);

void subtract_mat(matrix_t *out, matrix_t *mat1, matrix_t *mat2);

void scalar_multiply_vec(vector_t *out, vector_t *vec, float scalar);

void add_mat(matrix_t *out, matrix_t *mat1, matrix_t *mat2);

#endif