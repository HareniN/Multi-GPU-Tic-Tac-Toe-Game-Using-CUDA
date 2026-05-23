## TITLE : Multi-GPU Tic-Tac-Toe Game Using CUDA

## DESCRIPTION

This project implements a simple Tic-Tac-Toe game using CUDA and GPU programming concepts. The main objective of the project is to simulate two GPU competitors that automatically play against each other. GPU 0 represents Player X, while GPU 1 represents Player O. Each GPU uses a CUDA kernel to analyze the game board and decide the next valid move.

The program demonstrates how parallel processing can be used in game logic and decision-making. Each thread in the CUDA kernel checks one position of the board, and the GPUs take turns making moves until one player wins or the game ends in a draw. The project also demonstrates memory transfer between CPU and GPU, kernel execution, and simple synchronization between turns.

Since Google Colab provides only one GPU, the demonstration simulates both players on the same GPU while still maintaining the multi-GPU design concept.

## PROGRAM
```
%%writefile multi_gpu_tictactoe.cu

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define SIZE 9

void printBoard(char board[SIZE])
{
    cout << "\n";

    for(int i = 0; i < SIZE; i++)
    {
        cout << " " << board[i] << " ";

        if(i % 3 != 2)
            cout << "|";

        if(i % 3 == 2 && i != 8)
            cout << "\n-----------\n";
    }

    cout << "\n\n";
}

bool checkWinner(char board[SIZE], char p)
{
    int win[8][3] = {
        {0,1,2},{3,4,5},{6,7,8},
        {0,3,6},{1,4,7},{2,5,8},
        {0,4,8},{2,4,6}
    };

    for(int i=0;i<8;i++)
    {
        if(board[win[i][0]] == p &&
           board[win[i][1]] == p &&
           board[win[i][2]] == p)
            return true;
    }

    return false;
}

__global__ void findMove(char *board, int *move)
{
    int idx = threadIdx.x;

    if(idx < SIZE)
    {
        if(board[idx] == ' ')
        {
            atomicMin(move, idx);
        }
    }
}

int gpuMove(int gpuID, char board[SIZE])
{
    cudaSetDevice(gpuID);

    char *d_board;
    int *d_move;

    int move = 99;

    cudaMalloc((void**)&d_board, SIZE * sizeof(char));
    cudaMalloc((void**)&d_move, sizeof(int));

    cudaMemcpy(d_board, board,
               SIZE * sizeof(char),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_move, &move,
               sizeof(int),
               cudaMemcpyHostToDevice);

    findMove<<<1, SIZE>>>(d_board, d_move);

    cudaMemcpy(&move, d_move,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_board);
    cudaFree(d_move);

    return move;
}

int main()
{
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    cout << "Detected GPUs: "
         << deviceCount << endl;

    if(deviceCount < 1)
    {
        cout << "GPU not found.\n";
        return 0;
    }

    char board[SIZE] = {
        ' ',' ',' ',
        ' ',' ',' ',
        ' ',' ',' '
    };

    int turn = 0;

    cout << "GPU Tic-Tac-Toe Game\n";

    printBoard(board);

    while(turn < SIZE)
    {
        int move;

        move = gpuMove(0, board);

        if(turn % 2 == 0)
        {
            board[move] = 'X';
            cout << "GPU plays X at "
                 << move << endl;
        }
        else
        {
            board[move] = 'O';
            cout << "GPU plays O at "
                 << move << endl;
        }

        printBoard(board);

        if(checkWinner(board, 'X'))
        {
            cout << "X Wins!\n";
            break;
        }

        if(checkWinner(board, 'O'))
        {
            cout << "O Wins!\n";
            break;
        }

        turn++;
    }

    if(turn == SIZE)
        cout << "Game Draw!\n";

    return 0;
}
```

## OUTPUT

<img width="384" height="816" alt="Screenshot 2026-05-23 203707" src="https://github.com/user-attachments/assets/46d34bc7-d7ee-4e31-9471-da9b757d5223" />
