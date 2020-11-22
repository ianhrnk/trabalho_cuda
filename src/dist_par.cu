/********************************************************************************
 * Trabalho 2: Programação Paralela para Processador Many-core (GPU) Usando CUDA
 * Professora: Nahri Moreano
 * Aluno: Ian Haranaka | RGA: 2018.1904.009-7
 * Comando de compilação: nvcc dist_par.cu -o dist_par
 ********************************************************************************/

#include <iostream>
#include <fstream>

const int MAX_THREADS = 1024;

char *AlocaSequencia(int n);
__global__ void InicializaMatriz(int n, int m, int *D);
__global__ void CalculaDistanciaEdicao(int num_linhas, int num_colunas,
                                      int desloc_linha, int desloc_coluna,
                                      int n, int m,
                                      char *S, char *R, int *D);

int main(int argc, char *argv[])
{
  std::ifstream entrada (argv[1]);
  int n, m, resultado = 0;
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
    std::cout << "Arquivo não encontrado!" << std::endl;
    return -1;
  }

  int tam_vetor = (n+1) * (m+1);
  // Número de blocos por linha e por coluna
  int num_blocos_n = (n + MAX_THREADS - 1) / MAX_THREADS;
  int num_blocos_m = (m + MAX_THREADS - 1) / MAX_THREADS;

  // Aloca a matriz e as strings na memória global da GPU
  int *d_D;
  char *d_S, *d_R;
  cudaMalloc((void **)&d_D, tam_vetor * sizeof(int));
  cudaMalloc(&d_S, (n+2) * sizeof(char));
  cudaMalloc(&d_R, (m+2) * sizeof(char));

  // Copia as strings para a memória global da GPU
  cudaMemcpy(d_S, h_S, (n+2) * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, h_R, (m+2) * sizeof(char), cudaMemcpyHostToDevice);

  InicializaMatriz<<<1, 1>>>(n, m, d_D);
  cudaDeviceSynchronize();

  int threads_por_bloco = n;
  if (n > MAX_THREADS)
    threads_por_bloco = MAX_THREADS;

  int num_antidiag = num_blocos_n + num_blocos_m - 1;
  int num_linhas = threads_por_bloco, num_colunas = m;

  if (m > 1024)
    num_colunas = 1024;

  // Para cada anti-diagonal
  for (int antidiag = 0; antidiag < num_antidiag; ++antidiag)
  {
    int desloc_linha = 0;
    int desloc_coluna = antidiag * num_colunas;

    for (int bloco = 0; bloco < num_blocos_n; ++bloco)
    {
      // Se o bloco é válido - Isto implica em dizer se o bloco contém um pedaço da matriz
      if (desloc_coluna >= 0 && desloc_coluna < num_blocos_m * num_colunas)
        if (desloc_linha < num_blocos_n * num_linhas)
        {
          CalculaDistanciaEdicao<<<1, threads_por_bloco>>>(num_linhas, num_colunas,
          desloc_linha, desloc_coluna, n, m, d_S, d_R, d_D);
        }

      desloc_linha += num_linhas;
      desloc_coluna -= num_colunas;
    }
    cudaDeviceSynchronize();
  }

  cudaMemcpy(&resultado, &d_D[tam_vetor - 1], sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Resultado: "<< resultado << std::endl;

  // Libera o vetor e as strings
  delete[] h_S;
  delete[] h_R;
  cudaFree(d_D);
  cudaFree(d_S);
  cudaFree(d_R);

  return 0;
}

char *AlocaSequencia(int n)
{
  char *seq = new (std::nothrow) char[n+2];

  if (seq == nullptr)
    std::cout << "Erro na alocação de memória!" << std::endl;

  seq[0] = ' ';
  return seq;
}

__global__ void InicializaMatriz(int n, int m, int *D)
{
  for (int i = 0, j = 0; i < (n+1) * (m+1); i += m+1, ++j)
    D[i] = j;
  for (int j = 0; j <= m; ++j)
    D[j] = j;
}

__global__ void CalculaDistanciaEdicao(int num_linhas, int num_colunas,
                                      int desloc_linha, int desloc_coluna,
                                      int n, int m,
                                      char *S, char *R, int *D)
{
  int a, b, c, t, min;
  int num_antidiag = num_linhas + num_colunas - 1;
  int i = desloc_linha + threadIdx.x + 1, j;

  // Para cada antidiagonal
	for (int antidiag = 2; antidiag <= num_antidiag + 1; antidiag++)
	{
    j = desloc_coluna + antidiag - (threadIdx.x + 1);
    // Se é uma célula válida
    if (i <= n && j > desloc_coluna && j <= desloc_coluna + num_colunas)
    {
      t = (S[i] != R[j] ? 1 : 0);
      a = D[i*(m+1)+j-1] + 1;
      b = D[(i-1)*(m+1)+j] + 1;
      c = D[(i-1)*(m+1)+j-1] + t;

      min = (a < b ? a : b);
      min = (c < min ? c : min);
      D[i*(m+1)+j] = min;
    }
    __syncthreads();
  }
}
