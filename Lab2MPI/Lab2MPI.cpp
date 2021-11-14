#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
int ProcNum = 0; 
int ProcRank = 0;
int GridSize; 
int GridCoords[2]; 
MPI_Comm GridComm; 
MPI_Comm ColComm; 
MPI_Comm RowComm; 

void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
	int i, j; 
	srand(unsigned(clock()));
	for (i = 0; i < Size; i++)
		for (j = 0; j < Size; j++) {
			pAMatrix[i * Size + j] = rand() / double(1000);
			pBMatrix[i * Size + j] = rand() / double(1000);
		}
}

void ProcessInitialization(double*& pAMatrix, double*& pBMatrix,
	double*& pCMatrix, double*& pAblock, double*& pBblock, double*& pCblock,
	double*& pTemporaryAblock, int& Size, int& BlockSize) {
	if (ProcRank == 0) {
		do {
			printf("\nEnter size of the initial objects: ");
			fflush(stdout);
			scanf("%d", &Size);
			if (Size % GridSize != 0) {
				printf("Size of matricies must be divisible by the grid size!\n");
			}
		} while (Size % GridSize != 0);
	}
	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	BlockSize = Size / GridSize;
	pAblock = new double[BlockSize * BlockSize];
	pBblock = new double[BlockSize * BlockSize];
	pCblock = new double[BlockSize * BlockSize];
	pTemporaryAblock = new double[BlockSize * BlockSize];
	for (int i = 0; i < BlockSize * BlockSize; i++) {
		pCblock[i] = 0;
	}
	if (ProcRank == 0) {
		pAMatrix = new double[Size * Size];
		pBMatrix = new double[Size * Size];
		pCMatrix = new double[Size * Size];
		RandomDataInitialization(pAMatrix, pBMatrix, Size);
	}
}

void CheckerboardMatrixScatter(double* pMatrix, double* pMatrixBlock, int Size, int BlockSize) {
	double* MatrixRow = new double[BlockSize * Size];
	if (GridCoords[1] == 0) {
		MPI_Scatter(pMatrix, BlockSize * Size, MPI_DOUBLE, MatrixRow, BlockSize * Size, MPI_DOUBLE, 0, ColComm);
	}
	for (int i = 0; i < BlockSize; i++) {
		MPI_Scatter(&MatrixRow[i * Size], BlockSize, MPI_DOUBLE, &(pMatrixBlock[i * BlockSize]), BlockSize, MPI_DOUBLE, 0, RowComm);
	}
	delete[] MatrixRow;
}

void DataDistribution(double* pAMatrix, double* pBMatrix, double*
	pAblock, double* pBblock, int Size, int BlockSize) {
	
	CheckerboardMatrixScatter(pAMatrix, pAblock, Size, BlockSize);
	CheckerboardMatrixScatter(pBMatrix, pBblock, Size, BlockSize);
}

void ResultCollection(double* pCMatrix, double* pCblock, int Size,
	int BlockSize) {
	double* pResultRow = new double[Size * BlockSize];
	for (int i = 0; i < BlockSize; i++) {
		MPI_Gather(&pCblock[i * BlockSize], BlockSize, MPI_DOUBLE,
			&pResultRow[i * Size], BlockSize, MPI_DOUBLE, 0, RowComm);
	}
	if (GridCoords[1] == 0) {
		MPI_Gather(pResultRow, BlockSize * Size, MPI_DOUBLE, pCMatrix,
			BlockSize * Size, MPI_DOUBLE, 0, ColComm);
	}
	delete[] pResultRow;
}

void ParallelResultCalculation(double* pAblock, double* pMatrixAblock,
	double* pBblock, double* pCblock, int BlockSize) {
	for (int iter = 0; iter < GridSize; iter++) {
		
		int Pivot = (GridCoords[0] + iter) % GridSize;
		if (GridCoords[1] == Pivot) {
			for (int i = 0; i < BlockSize * BlockSize; i++)
				pAblock[i] = pMatrixAblock[i];
		}
		MPI_Bcast(pAblock, BlockSize * BlockSize, MPI_DOUBLE, Pivot, RowComm);
		
		int i, j, k;
		for (i = 0; i < BlockSize; i++) {
			for (j = 0; j < BlockSize; j++)
				for (k = 0; k < BlockSize; k++)
					pCblock[i * BlockSize + j] += pAblock[i * BlockSize + k] * pBblock[k * BlockSize + j];
		}
		
		MPI_Status Status;
		int NextProc = GridCoords[0] + 1;
		if (GridCoords[0] == GridSize - 1) NextProc = 0;
		int PrevProc = GridCoords[0] - 1;
		if (GridCoords[0] == 0) PrevProc = GridSize - 1;
		MPI_Sendrecv_replace(pBblock, BlockSize * BlockSize, MPI_DOUBLE, NextProc, 0, PrevProc, 0, ColComm, &Status);
	}
}

void ProcessTermination(double* pAMatrix, double* pBMatrix,
	double* pCMatrix, double* pAblock, double* pBblock, double* pCblock,
	double* pMatrixAblock) {
	if (ProcRank == 0) {
		delete[] pAMatrix;
		delete[] pBMatrix;
		delete[] pCMatrix;
	}
	delete[] pAblock;
	delete[] pBblock;
	delete[] pCblock;
	delete[] pMatrixAblock;
}


void main(int argc, char* argv[]) {
	double* pAMatrix; 
	double* pBMatrix; 
	double* pCMatrix; 
	int Size; 
	int BlockSize; 
	double* pAblock; 
	double* pBblock; 
	double* pCblock; 
	double* pMatrixAblock;
	double Start, Finish, Duration;
	setvbuf(stdout, 0, _IONBF, 0);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	GridSize = sqrt((double)ProcNum);
	if (ProcNum != GridSize * GridSize) {
		if (ProcRank == 0) {
			printf("Number of processes must be a perfect square \n");
		}
	}
	else {
		if (ProcRank == 0)
			printf("Parallel matrix multiplication program\n");
		
		int DimSize[2];
		int Periodic[2];
		int Subdims[2];


		DimSize[0] = GridSize;
		DimSize[1] = GridSize;
		Periodic[0] = 0;
		Periodic[1] = 0;

		MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);
		MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);

		Subdims[0] = 0;
		Subdims[1] = 1;
		MPI_Cart_sub(GridComm, Subdims, &RowComm);

		Subdims[0] = 1;
		Subdims[1] = 0;
		MPI_Cart_sub(GridComm, Subdims, &ColComm);

		
		ProcessInitialization(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock,
			pCblock, pMatrixAblock, Size, BlockSize);

		Start = MPI_Wtime();
		DataDistribution(pAMatrix, pBMatrix, pMatrixAblock, pBblock, Size,
			BlockSize);
		ParallelResultCalculation(pAblock, pMatrixAblock, pBblock,
			pCblock, BlockSize);
		ResultCollection(pCMatrix, pCblock, Size, BlockSize);
		Finish = MPI_Wtime();

		Duration = Finish - Start;
		if (ProcRank == 0) {
			printf("Time of execution = %f\n", Duration);
		}

		ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock,
			pCblock, pMatrixAblock);
	}
	MPI_Finalize();
}