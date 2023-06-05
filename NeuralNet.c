#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void init_layer(struct layer* layer, int input, int output, struct matrix *weights, struct vector *biases, struct vector *nodes)
{

    //init the inputs and outputs of layer
    layer->input = input;
    layer->output = output;    

    //defining the rows and columns of random-weight matrix
    weights->col = input;
    weights->row = output; 


    //allocating space for the weight matrix rows & cols
    weights->arr = allocate_mat_arr(input, output);

    for (int i = 0; i < weights->row; i++)
    {
        for (int j = 0; j < input; j++)
        {
            weights->arr[j + i * weights->col] = (double)(rand() / (RAND_MAX+ 1.0));
        }
    }

    //init the size of vector
    biases->len = input;
    //init the bias vector
    biases->arr = allocate_vec_arr(biases->len);
    
    //allocating random doubles to bias
    for (int i = 0; i < biases->len; i++)
    {   
        //random biases from [0, 1)
        biases->arr[i]=  (double)(rand() / (RAND_MAX+ 1.0));
    }
    
    layer->nodes = *nodes;
    layer->random_weights = weights;
    layer->random_bias = biases;
}


void free_layer(struct matrix* matrix, struct vector* vector)
{
    free(matrix->arr);
    free(vector->arr);
}

struct vector forward(struct layer* input)
{
    struct vector weight_inputs = multiply(input->random_weights, &(input->nodes));

    struct vector result = add(&weight_inputs, input->random_bias);

    free_vector(&weight_inputs);

    return result;

 
}

struct vector activation(struct layer* input, int length)
{
    int i = 0;
    while (i < length)
    {
        //make sure its not the first layer bc it already has iputs
        if(i != 0)
        {
            
            (input[i].nodes) = forward(&input[i-1]);
            (input[i].activation) = sigmoid_vector(&input[i].nodes);
        }
        i++;
    }
    return (input[length-1].activation);
}

