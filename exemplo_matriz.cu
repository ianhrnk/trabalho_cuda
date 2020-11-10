// Exemplo 17: Soma de matrizes
//             Usa grid e blocos bidimensionais
// Para compilar: nvcc ex17.cu -o ex17
// Para executar: ./ex17

#include <stdio.h>
#include <stdlib.h>

void soma_matriz_CPU(int nLinhas, int nColunas, int *a, int *b, int *c)
{
	int i, j;

	for (i = 0; i < nLinhas; i++)
		for (j = 0; j < nColunas; j++)
			c[i * nColunas + j] = a[i * nColunas + j] + b[i * nColunas + j];	// Célula c[i][j]
}

// Kernel executado na GPU por todas as threads de todos os blocos
__global__ void soma_matriz_GPU(int nLinhas, int nColunas, int *a, int *b, int *c)
{
	int i, j; // id GLOBAL da thread

	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < nLinhas) && (j < nColunas))
		c[i * nColunas + j] = a[i * nColunas + j] + b[i * nColunas + j];
}

// Programa principal: execução inicia no host
int main(void)
{
	int nLinhas = 1024,
		 nColunas = 1024,
		 nBytes = nLinhas * nColunas * sizeof(int),
		 i, j,
		 *h_a, *h_b, *h_c,	// Variáveis do host
		 *d_a, *d_b, *d_c,	// Variáveis da GPU (device)
		 *c_referencia;

	// Aloca matrizes no host
	h_a = (int *) malloc(nBytes);
	h_b = (int *) malloc(nBytes);
	h_c = (int *) malloc(nBytes);
	c_referencia = (int *) malloc(nBytes);

	// Inicializa variáveis do host
	for (i = 0; i < nLinhas; i++)
		for (j = 0; j < nColunas; j++)
		{
			h_a[i * nColunas + j] = i + j;	// Célula h_a[i][j]
			h_b[i * nColunas + j] = i + j;	// Célula h_b[i][j]
		}

	// Soma matrizes na CPU
	soma_matriz_CPU(nLinhas, nColunas, h_a, h_b, c_referencia);

	// Aloca matrizes na memória global da GPU
	cudaMalloc((void **)&d_a, nBytes);
	cudaMalloc((void **)&d_b, nBytes);
	cudaMalloc((void **)&d_c, nBytes);

	// Copia dados de entrada do host para memória global da GPU
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

	// Determina nBlocos em função de nLinhas e NColunas
	dim3 nThreadsBloco(32,32);
	dim3 nBlocos((nLinhas + (nThreadsBloco.x - 1)) / nThreadsBloco.x, (nColunas + (nThreadsBloco.y - 1)) / nThreadsBloco.y);

	// Soma matrizes na GPU
	// Grid bidimensaional com blocos bidimensionais
	soma_matriz_GPU<<<nBlocos, nThreadsBloco>>>(nLinhas, nColunas, d_a, d_b, d_c);

	// Copia resultados da memória global da GPU para host
	cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);

	// Checa resultado
	i = 0;
	bool erro = false;
	while (i < nLinhas && !erro)
	{
		j = 0;
		while (j < nColunas && !erro)
		{
			if (h_c[i * nColunas + j] != c_referencia[i * nColunas + j])
				erro = true;
			j++;
		}
		i++;
	}
	printf("%s\n", (erro ? "Resultado errado" : "Resultado correto"));

	// Libera matrizes na memória global da GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Libera matrizes no host
	free(h_a);
	free(h_b);
	free(h_c);
	free(c_referencia);

	return 0;
}
