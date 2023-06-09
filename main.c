#include "NeuralNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dataset.h"
#include "mnist.h"

void load_mnist_matrix_vector(matrix_t *x_train, matrix_t *y_train, matrix_t *x_test, matrix_t *y_test)
{
    for (int i = 0; i < NUM_TRAIN; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            x_train->arr[i * SIZE + j] = train_image[i][j];
        }
    }

    for (int i = 0; i < NUM_TEST; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            x_test->arr[i * SIZE + j] = test_image[i][j];
        }
    }

    for (int i = 0; i < NUM_TRAIN; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if(train_label[i] == j)
            {
                y_train->arr[i * 10 + j] = 1;
            }
            else
            {
                y_train->arr[i * 10 + j] = 0;
            }
        }
    }

    for (int i = 0; i < NUM_TEST; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if(test_label[i] == j)
            {
                y_test->arr[i * 10 + j] = 1;
            }
            else
            {
                y_test->arr[i * 10 + j] = 0;
            }
        }
    }
}


int main()
{
    srand(time(NULL));

    int sizes[] = {784, 64, 64, 10};

    load_mnist();

    neural_net_t net = allocate_neural_net(4, sizes);
    printf("Allocated Network\n");

    // TODO: Shuffle data
    matrix_t x_train = init_matrix(NUM_TRAIN, SIZE);
    matrix_t y_train = init_matrix(NUM_TRAIN, 10);

    matrix_t x_test = init_matrix(NUM_TEST, SIZE);
    matrix_t y_test = init_matrix(NUM_TRAIN, 10);

    matrix_t test_input = init_matrix(1, SIZE);
    matrix_t test_output = init_matrix(1, 10);

    load_mnist_matrix_vector(&x_train, &y_train, &x_test, &y_test);
    printf("Loaded MNIST\n");

    printf("\nTraining...\n");
    train(&net, &x_train, &y_train, 3, 10, 3.0, &x_test, &y_test);
    printf("Trained\n");

    free_network(&net);
    free_matrix(&x_train);
    free_matrix(&y_train);
    free_matrix(&x_test);
    free_matrix(&y_test);

    printf("freed\n");

    return 0;
}
//*/
/*
int main()
{
    neural_net_t net = allocate_neural_net(3, (int[]){10, 64, 10});

    vector_t vec = init_vector(10);
    matrix_t mat = init_matrix(10, 10);

    forward_pass(&net);
    backward_pass(&net, &vec);

    train(&net, &mat, &mat, 1, 1, 1.0, &mat, &mat);

    free_network(&net);
    free_vector(&vec);
    free_matrix(&mat);

    return 0;
}
//*/