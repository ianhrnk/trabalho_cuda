/********************************************************************************
 * Trabalho 2: Programação Paralela para Processador Many-core (GPU) Usando CUDA
 * Professora: Nahri Moreano
 * Aluno: Ian Haranaka | RGA: 2018.1904.009-7
 * Comando de compilação: nvcc solucao_inicial.cu -o solucao_inicial
 ********************************************************************************/

#include <iostream>
#include <fstream>

char *AlocaSequencia(int n);
__global__ void DistanciaEdicao(int n, int m, char *S, char *R, int *D);

int main(int argc, char *argv[])
{
  std::ifstream entrada (argv[1]);
  int n, m, resultado;
  char *h_S, *h_R;

  // Leitura do arquivo de entrada
  if (entrada.is_open())
  {
    entrada >> n >> m;

    h_S = AlocaSequencia(n);
    h_R = AlocaSequencia(m);
    entrada >> &(h_S[1]) >> &(h_R[1]);

    entrada.close();
  }
  else
  {
    std::cout << "Arquivo não encontrado!\n";
    return -1;
  }

  int tamanho_grid = (n+1) * (m+1);

  // Aloca matriz e strings na memória global do kernel
  int *d_D;
  char *d_S, *d_R;
  cudaMalloc(&d_D, tamanho_grid * sizeof(int));
  cudaMalloc(&d_S, (n+1) * sizeof(char));
  cudaMalloc(&d_R, (m+1) * sizeof(char));

  // Copia as strings para a memória global do kernel
  cudaMemcpy(d_S, h_S, (n+1) * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, h_R, (m+1) * sizeof(char), cudaMemcpyHostToDevice);

  // Invoca o kernel
  int threadsPerBlock = n; // No máximo 1024
  DistanciaEdicao<<<1, threadsPerBlock>>>(n, m, d_S, d_R, d_D);

  cudaDeviceSynchronize();

  cudaMemcpy(&resultado, &d_D[tamanho_grid - 1], sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << resultado << std::endl;

  // Libera matriz e strings
  free(h_S);
  free(h_R);
  cudaFree(d_D);
  cudaFree(d_S);
  cudaFree(d_R);

  return 0;
}

char *AlocaSequencia(int n)
{
  char *seq;

  seq = (char *) malloc((n+1) * sizeof(char));
  if (seq == NULL)
  {
    std::cout << "Erro na alocação de estruturas!\n";
    std::exit(1);
  }
  seq[0] = ' ';
  return seq;
}

__global__ void DistanciaEdicao(int n, int m, char *S, char *R, int *D)
{
  int a, b, c, t, min;
  int nADiag = n + m - 1;
  int i = threadIdx.x + 1, j;

  // Inicialização da matriz
  if (threadIdx.x == 0)
  {
    for (int i = 0, j = 0; i < (n+1) * (m+1); i += m+1, ++j)
      D[i] = j;
    for (int j = 0; j <= m; ++j)
      D[j] = j;
  }

  __syncthreads();

  // Para cada anti-diagonal
	for (int aD = 2; aD <= nADiag + 1; aD++)
	{
    j = aD - i;
    // Se é uma célula válida
    if (j > 0 && j <= m)
    {
      t = (S[i] != R[j] ? 1 : 0);
      a = D[i*(m+1)+j-1] + 1;
      b = D[(i-1)*(m+1)+j] + 1;
      c = D[(i-1)*(m+1)+j-1] + t;
      // Calcula d[i][j] = min(a, b, c)
      min = (a < b ? a : b);
      min = (c < min ? c : min);
      D[i*(m+1)+j] = min;
    }
    __syncthreads();
  }
}
