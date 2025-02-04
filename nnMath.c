#include "nnMath.h"

// * is the hadamard product
// L is the last layer
// y is a vector of outputs
// z is the weighted output of a layer
// a is the activated output of a layer
// w are the weights of a layer
// b are the biases of a layer
// dsigmoid is the derivative of the sigmoud function
// delta is the error of a layer
// ^T is the transpose of a matrix
// l is a layer index
// n is the learning rate
// m is the number of samples
// sum is a summation over all of the samples x
//
// For each training example x:
//  Feedforward:
//      For each layer l:
//          z(x, l) = w(l)a(x, l-1) + b(l)
//          a(x, l) = sigmoid(z(x, l))
//  Output Error:
//      delta(x, L) = (a(L) - y) * dsigmoid(z(x, L))

//  Backpropogate Error:
//      For each layer l starting at L:
//          delta(x, l) = (w(l+1)^T delta(x, l+1)) * dsigmoid(z(x, l))
//
// Gradient descent
//  For each layer l starting at L:
//      w(l) = w(l) - (n/m)sum(delta(x, l)a(x, l-1)^T)
//      b(l) = b(l) -   (n/m)sum(delta(x, l))


vector_t init_vector(int len)
{
    vector_t vec;
    vec.len = len;
    vec.arr = allocate_vec_arr(vec.len);

    return vec;
}
matrix_t init_matrix(int row, int col)
{
    matrix_t mat;
    mat.row = row;
    mat.col = col;
    mat.arr = allocate_mat_arr(mat.row, mat.col);

    return mat;
}

inline void free_vector(vector_t *vec)
{
    free(vec->arr);
}
inline void free_matrix(matrix_t *mat)
{
    free(mat->arr);
}

inline float *allocate_mat_arr(int row, int col)
{
    return (float*)calloc(row*col, sizeof(float));
}
inline float *allocate_vec_arr(int len)
{
    return (float*)calloc(len, sizeof(float));
}

inline matrix_t *allocate_mat()
{
    return (matrix_t*)malloc(sizeof(matrix_t));
}

inline vector_t *allocate_vec()
{
    return (vector_t*)malloc(sizeof(vector_t));
}

inline void multiply_mat_vec(vector_t *out, matrix_t *mat, vector_t *vec)
{
    int j, k = 0;
    for (int i = 0; i < mat->row; i++)
    {
        for (j = 0; j < mat->col; j++)
        {
            out->arr[i] += mat->arr[k++] * vec->arr[j];
        }
    }
}

inline void add_vec(vector_t *out, vector_t *v1, vector_t *v2)
{
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = v1->arr[i] + v2->arr[i];
    }
}

inline void subtract_vec(vector_t *out, vector_t *v1, vector_t *v2)
{
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = v1->arr[i] - v2->arr[i];
    }
}

inline float sigmoid(float val)
{
    return 1 / (1 + exp(-1 * val));
}

inline void sigmoid_mat(matrix_t *out, matrix_t* mat)
{
    for (int i = 0; i < mat->row*mat->col; i++)
    {
        out->arr[i] = sigmoid(mat->arr[i]);
    }
}

inline void dsigmoid_mat(matrix_t *out, matrix_t* mat)
{
    float sig;
    for (int i = 0; i < mat->row*mat->col; i++)
    {   
        sig = sigmoid(mat->arr[i]);
        out->arr[i] = sig * (1 - sig);
    }
}

inline void sigmoid_vec(vector_t *out, vector_t* vec)
{
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = sigmoid(vec->arr[i]);
    }
}

inline void dsigmoid_vec(vector_t *out, vector_t* vec)
{
    float sig;
    for (int i = 0; i < out->len; i++)
    {
        sig = sigmoid(vec->arr[i]);
        out->arr[i] = sig * (1 - sig);
    }
}

inline void hadamard_product(vector_t *out, vector_t *vec1, vector_t *vec2)
{
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = vec1->arr[i] * vec2->arr[i];
    }
}

void output_error(vector_t *out,
                  vector_t *expected_output, 
                  vector_t *last_layer_activations, 
                  vector_t *last_layer_weighted)
{
    vector_t llw_dsig = init_vector(last_layer_weighted->len);
    dsigmoid_vec(&llw_dsig, last_layer_weighted);

    vector_t error = init_vector(last_layer_activations->len);
    subtract_vec(&error, last_layer_activations, expected_output);

    hadamard_product(out, &error, &llw_dsig);

    free_vector(&llw_dsig);
    free_vector(&error);
}

void layer_error(vector_t *out, 
                 matrix_t *next_layer_weights,
                 vector_t *next_layer_error,
                 vector_t *current_layer_weighted)
{
    vector_t clw_dsig = init_vector(current_layer_weighted->len);
    dsigmoid_vec(&clw_dsig, current_layer_weighted);

    matrix_t nlwT = init_matrix(next_layer_weights->col, next_layer_weights->row);
    transpose(&nlwT, next_layer_weights);

    vector_t product = init_vector(nlwT.row);
    multiply_mat_vec(&product, &nlwT, next_layer_error);

    hadamard_product(out, &product, &clw_dsig);

    free_vector(&clw_dsig);
    free_matrix(&nlwT);
    free_vector(&product);
}

inline void transpose(matrix_t *out, matrix_t* mat)
{
    for (int i = 0; i < out->row; i++)
    {
        for (int j = 0; j < out->col; j++)
        {
            out->arr[j + i * out->col] = mat->arr[i + j * mat->col];
        }
    }
}

inline void multiply_vec_vec(matrix_t *out, vector_t *v1, vector_t *v2)
{
    int k = 0;
    float* arr1= v1->arr;
    float* arr2 = v2->arr;
    for (int i = 0; i < out->row; i++)
    {
        for (int j = 0; j < out->col; j++)
        {
            out->arr[k++] = arr1[i] * arr2[j];
        }
    }
}

inline void scalar_multiply_mat(matrix_t *out, matrix_t *mat, float scalar)
{
    int max = mat->row*mat->col;
    for (int i = 0; i < max; i+=4)
    {
        out->arr[i] = mat->arr[i] * scalar;
        out->arr[i+1] = mat->arr[i+1] * scalar;
        out->arr[i+2] = mat->arr[i+2] * scalar;
        out->arr[i+3] = mat->arr[i+3] * scalar;
    }
}

inline void subtract_mat(matrix_t *out, matrix_t *mat1, matrix_t *mat2)
{
    int max = out->row*out->col;
    int i;
    for (i = 0; i < max; i+=4)
    {
        out->arr[i] = mat1->arr[i] - mat2->arr[i];
        out->arr[i+1] = mat1->arr[i+1] - mat2->arr[i+1];
        out->arr[i+2] = mat1->arr[i+2] - mat2->arr[i+2];
        out->arr[i+3] = mat1->arr[i+3] - mat2->arr[i+3];
    }
}

inline void scalar_multiply_vec(vector_t *out, vector_t *vec, float scalar)
{
    for (int i = 0; i < out->len; i++)
    {
        out->arr[i] = vec->arr[i] * scalar;
    }
}

inline void add_mat(matrix_t *out, matrix_t *mat1, matrix_t *mat2)
{
    int max = out->row*out->col;
    int i;
    for (i = 0; i < max; i+=4)
    {
        out->arr[i] = mat1->arr[i] + mat2->arr[i];
        out->arr[i+1] = mat1->arr[i+1] + mat2->arr[i+1];
        out->arr[i+2] = mat1->arr[i+2] + mat2->arr[i+2];
        out->arr[i+3] = mat1->arr[i+3] + mat2->arr[i+3];
    }
}
