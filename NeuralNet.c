#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


neural_net_t allocate_neural_net(int layers, int* layer_sizes)
{
    neural_net_t new_net;
    new_net.num_layers = layers;

    new_net.layers = (layer_t *)malloc(sizeof(layer_t) * layers);

    for (int i = 0; i < new_net.num_layers; i++)
    {
        if(i == 0)
            new_net.layers[i] = init_layer(layer_sizes[i], 1);
        else
            new_net.layers[i] = init_layer(layer_sizes[i], layer_sizes[i - 1]);
    }

    return new_net;
}

void free_network(neural_net_t *network)
{
    for(int i = 0; i < network->num_layers; i++)
    {
        free_layer(&network->layers[i]);
    }
    free(network->layers);
}

void free_layer(layer_t* layer)
{
    free_matrix(&layer->weights);
    free_vector(&layer->biases);
    free_vector(&layer->weighted_outputs);
    free_vector(&layer->activated_outputs);
    free_vector(&layer->error);
}

layer_t init_layer(int length, int previous_layer_length)
{
    layer_t out;
    out.length = length;

    //defining the rows and columns of random-weight matrix
    out.weights = init_matrix(length, previous_layer_length);

    for (int i = 0; i < out.weights.row; i++)
    {
        for (int j = 0; j < out.weights.col; j++)
        {
            //random vals between [0,1)
            out.weights.arr[j + i * out.weights.col] = (float)((float)rand() / (RAND_MAX / 2)) - 1;
        }
    }

    //init the size of vector
    out.biases = init_vector(length);
    
    //allocating random floats to bias
    for (int i = 0; i < out.biases.len; i++)
    {   
        //random biases from [0, 1)
        out.biases.arr[i] = (float)(rand() / (RAND_MAX / 2)) - 1;
    }

    out.activated_outputs = init_vector(length);
    out.weighted_outputs = init_vector(length);

    out.error = init_vector(length);

    return out;
}

void feed_forward(layer_t *current_layer, layer_t *previous_layer)
{
    vector_t weight_inputs = init_vector(current_layer->weights.row);

    //example for second layer, [16x10][10x1]+[16x1]
    multiply_mat_vec(&weight_inputs, &current_layer->weights, &previous_layer->activated_outputs);

    add_vec(&current_layer->weighted_outputs, &weight_inputs, &current_layer->biases);

    free_vector(&weight_inputs);
}

void forward_pass(neural_net_t *network)
{
    for (int i = 1; i < network->num_layers; i++)
    {
        feed_forward(&network->layers[i], &network->layers[i - 1]);
        sigmoid_vec(&network->layers[i].activated_outputs, &network->layers[i].weighted_outputs);
    }
}

float loss_function(vector_t *predict, vector_t *actual)
{
    float sum = 0;
    for (int i = 0; i < predict->len; i++)
    {
        sum = fabs(predict->arr[i] - actual->arr[i]);
    }
    return sum / predict->len;
}

void backward_pass(neural_net_t *network, vector_t *expected_outputs)
{
    for (int i = network->num_layers - 1; i > 0; i--)
    {
        if (i == network->num_layers - 1)
        {
            vector_t cost_derivative = init_vector(network->layers[i].length);
            subtract_vec(&cost_derivative, &network->layers[i].activated_outputs, expected_outputs);
            vector_t sigmoid_derivative = init_vector(network->layers[i].length);
            dsigmoid_vec(&sigmoid_derivative, &network->layers[i].weighted_outputs);

            hadamard_product(&network->layers[i].error, &cost_derivative, &sigmoid_derivative);

            free_vector(&cost_derivative);
            free_vector(&sigmoid_derivative);
        }
        else
        {
            matrix_t weights_transpose = init_matrix(network->layers[i + 1].weights.col, network->layers[i + 1].weights.row);
            transpose(&weights_transpose, &network->layers[i + 1].weights);
            vector_t error_product = init_vector(weights_transpose.row);
            multiply_mat_vec(&error_product, &weights_transpose, &network->layers[i + 1].error);

            vector_t sigmoid_derivative = init_vector(network->layers[i].length);
            dsigmoid_vec(&sigmoid_derivative, &network->layers[i].weighted_outputs);

            hadamard_product(&network->layers[i].error, &error_product, &sigmoid_derivative);

            free_matrix(&weights_transpose);
            free_vector(&error_product);
            free_vector(&sigmoid_derivative);
        }
    }
}

void train(neural_net_t *network, matrix_t *inputs, matrix_t *expected_outputs, 
           int epochs, int batch_size, float learning_rate,
           matrix_t *test_inputs, matrix_t *test_expected_outputs, char* filename)
{
    printf("\n");
    matrix_t temp_weights[network->num_layers];
    vector_t temp_biases[network->num_layers];

    for (int i = 1; i < network->num_layers; i++)
    {
        temp_weights[i] = init_matrix(network->layers[i].weights.row, network->layers[i].weights.col);
        temp_biases[i] = init_vector(network->layers[i].biases.len);
    }

    vector_t expected_outputs2;
    expected_outputs2.arr = (float*)malloc(1);
    for (int i = 0; i < epochs; i++)
    {
        printf("Starting epoch %d\n", i + 1);
        for (int j = 0; j < inputs->row; j++)
        {
            expected_outputs2.len = network->layers[network->num_layers - 1].length;
            expected_outputs2.arr = (float*)realloc(expected_outputs2.arr, expected_outputs2.len*sizeof(float));
            for (int k = 0; k < inputs->col; k++)
            {
                network->layers[0].activated_outputs.arr[k] = inputs->arr[j * inputs->col + k];
            }
            for (int r = 0; r < network->layers[network->num_layers - 1].length; r++)
            {
                expected_outputs2.arr[r] = expected_outputs->arr[j * expected_outputs->col + r];
            }
            forward_pass(network);
            backward_pass(network, &expected_outputs2);
            update_temp_weights(temp_weights, network, learning_rate);
            update_temp_biases(temp_biases, network, learning_rate);

            if (j % batch_size == 0 || j == inputs->row - 1)
            {
                update_weights(network, temp_weights, batch_size, learning_rate);
                update_biases(network, temp_biases, batch_size, learning_rate);
                for (int i = 1; i < network->num_layers; i++)
                {
                    free_matrix(&temp_weights[i]);
                    free_vector(&temp_biases[i]);
                    temp_weights[i] = init_matrix(network->layers[i].weights.row, network->layers[i].weights.col);
                    temp_biases[i] = init_vector(network->layers[i].biases.len);
                }
                
            }
        }
        if((i + 1) % 1 == 0)
        {
            test(network, test_inputs, test_expected_outputs);
            save_network(network, filename);
        }
    }
    printf("Training complete\n");
    printf("Testing network...\n");
    test(network, test_inputs, test_expected_outputs);
    save_network(network, filename);

    for (int i = 1; i < network->num_layers; i++)
    {
        free_matrix(&temp_weights[i]);
        free_vector(&temp_biases[i]);
    }
}

void update_biases(neural_net_t *network, vector_t *temp_biases, int batch_size, float learning_rate)
{
    for (int i = 1; i < network->num_layers; i++)
    {
        scalar_multiply_vec(&temp_biases[i], &temp_biases[i], learning_rate / (float)batch_size);
        subtract_vec(&network->layers[i].biases, &network->layers[i].biases, &temp_biases[i]);
    }
}

void update_temp_biases(vector_t *biases, neural_net_t *net, float learning_rate)
{
    for (int i = net->num_layers - 1; i > 0; i--)
    {
        add_vec(&biases[i], &biases[i], &net->layers[i].error);
    }
}

void update_temp_weights(matrix_t *weights, neural_net_t *net, float learning_rate)
{
    for (int i = net->num_layers - 1; i > 0; i--)
    {
        matrix_t grad = init_matrix(net->layers[i].weights.row, net->layers[i].weights.col);
        multiply_vec_vec(&grad, &net->layers[i].error, &net->layers[i - 1].activated_outputs);
        add_mat(&weights[i], &weights[i], &grad);
        free_matrix(&grad);
    }
}

void update_weights(neural_net_t *net, matrix_t *temp_weights, int batch_size, float learning_rate)
{
    for (int i = 1; i < net->num_layers; i++)
    {
        scalar_multiply_mat(&temp_weights[i], &temp_weights[i], learning_rate / (float)batch_size);
        subtract_mat(&net->layers[i].weights, &net->layers[i].weights, &temp_weights[i]);
    }
}

void test(neural_net_t *network, matrix_t *inputs, matrix_t *expected_outputs)
{
    matrix_t network_output = init_matrix(inputs->row, inputs->col);
    int sum = 0;
    for (int i = 0; i < inputs->row; i++)
    {
        for (int j = 0; j < inputs->col; j++)
        {
            network->layers[0].activated_outputs.arr[j] = inputs->arr[i * inputs->col + j];
        }
        forward_pass(network);
        int max_index = 0;
        for (int j = 0; j < network->layers[network->num_layers - 1].activated_outputs.len; j++)
        {
            if (network->layers[network->num_layers - 1].activated_outputs.arr[j] > network->layers[network->num_layers - 1].activated_outputs.arr[max_index])
            {
                max_index = j;
            }
        }
        if(expected_outputs->arr[i * expected_outputs->col + max_index] == 1)
        {
            sum++;
        }
    }
    printf("Accuracy: %d / 10000\n", sum);

    free_matrix(&network_output);
}

void print_matrix(matrix_t *mat)
{
    for (int i = 0; i < mat->row; i++)
    {
        for (int j = 0; j < mat->col; j++)
        {
            printf("%lf, ", mat->arr[j + i * mat->col]);
        }
        printf("\n");
    }
}
void print_vector(vector_t *vec)
{
    for (int i = 0; i < vec->len; i++)
    {
        printf("%lf\n", vec->arr[i]);
    }
}

void save_matrix(matrix_t *mat, FILE *file)
{
    __uint8_t* temp = (__uint8_t*)mat->arr;

    for (int i = 0; i < mat->row*mat->col*sizeof(float); i++)
    {
        fprintf(file, "%c", temp[i]);
    }
}
void save_vector(vector_t *vec, FILE *file)
{
    __uint8_t* temp = (__uint8_t*)vec->arr;
    for (int i = 0; i < vec->len*sizeof(float); i++)
    {
        fprintf(file, "%c", temp[i]);
    }
}

void save_network(neural_net_t* network, char* filedescriptor)
{
    char filename[100];
    char* temp = filename;
    int j = 0;
    for(int i = 0; i < network->num_layers; i++)
    {
        snprintf(temp+j, 100, "%d-", network->layers[i].length);

        if(network->layers[i].length > 99999)
            return;
        else if(network->layers[i].length > 9999)
            j += 6;
        else if(network->layers[i].length > 999)
            j += 5;
        else if(network->layers[i].length > 99)
            j += 4;
        else if(network->layers[i].length > 9)
            j += 3;
        else if(network->layers[i].length < 9)
            j += 2;
    }
    snprintf(temp+j, 100, "%s.pickl", filedescriptor);
    printf("%s\n", filename);


    FILE* file = fopen(filename, "w");
    for(int i = 0; i < network->num_layers; i++)
    {
        save_matrix(&network->layers[i].weights, file);
        save_vector(&network->layers[i].biases, file);
    }
    fclose(file);
}

void load_network(neural_net_t* network, char* filename)
{
    char file[100];
    char* token;
    int num_layers = 0;
    int layers[12];

    strcpy(file, filename);
    token = strtok(file, "-");
   
    while (token != NULL && *token < 57 && num_layers < 12)
    {
        layers[num_layers] = atoi(token);
        token = strtok(NULL, "-");
        num_layers++;
    }

    *network = allocate_neural_net(num_layers, layers);

    FILE* fd = fopen(filename, "r");

    read_file(network, fd);

    fclose(fd);

}

void read_file(neural_net_t* net, FILE* file)
{
    __uint8_t* temp_weights;
    __uint8_t* temp_biases;
    int temp_weights_length;
    int temp_biases_length;

    for(int i = 0; i < net->num_layers; i++)
    {
        temp_weights_length = sizeof(float) * (net->layers[i].weights.row*net->layers[i].weights.col);
        temp_biases_length = sizeof(float) * (net->layers[i].biases.len);

        free(net->layers[i].weights.arr);
        free(net->layers[i].biases.arr);
        
        temp_weights = (__uint8_t*)calloc(temp_weights_length, sizeof(float));
        temp_biases = (__uint8_t*)calloc(temp_biases_length, sizeof(float));


        for(int j = 0; j < temp_weights_length || feof(file); j++)
        {
            temp_weights[j] = fgetc(file);
        }
        for(int j = 0; j < temp_biases_length || feof(file); j++)
        {
            temp_biases[j] = fgetc(file); 
        }
        net->layers[i].weights.arr = (float*)temp_weights;
        net->layers[i].biases.arr = (float*)temp_biases;
    }

}