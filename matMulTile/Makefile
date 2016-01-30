all: matMulTile.cu
	nvcc -g -pg matMulTile.cu -o matMulTile.out

getres: matMulTile.cu
	echo "#Running Script for matrixMulTile.cu" > res.data
	echo "#Tile Width	Matrix Dimensions	Execution Time" >> res.data
	nvcc -g -DTILE_WIDTH=4 -DMAT_DIM=512 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=4 -DMAT_DIM=1024 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=4 -DMAT_DIM=2048 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=8 -DMAT_DIM=512 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=8 -DMAT_DIM=1024 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=8 -DMAT_DIM=2048 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=16 -DMAT_DIM=512 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=16 -DMAT_DIM=1024 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=16 -DMAT_DIM=2048 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=32 -DMAT_DIM=512 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=32 -DMAT_DIM=1024 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	nvcc -g -DTILE_WIDTH=32 -DMAT_DIM=2048 matMulTile.cu -o matMulTile1.out
	./matMulTile1.out
	
clean:
	rm -rf *.out
