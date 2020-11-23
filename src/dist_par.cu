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
                                      int antidiag_bloco, int n, int m,
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

  // Aloca a matriz, as strings e copia as strings para a memória global da GPU
  int *d_D;
  char *d_S, *d_R;
  cudaMalloc((void **)&d_D, tam_vetor * sizeof(int));
  cudaMalloc(&d_S, (n+2) * sizeof(char));
  cudaMalloc(&d_R, (m+2) * sizeof(char));
  cudaMemcpy(d_S, h_S, (n+2) * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, h_R, (m+2) * sizeof(char), cudaMemcpyHostToDevice);

  InicializaMatriz<<<1, 1>>>(n, m, d_D);
  cudaDeviceSynchronize();

  int threads_por_bloco = (n < MAX_THREADS ? n : MAX_THREADS);
  int num_antidiag = num_blocos_n + num_blocos_m - 1;
  int num_linhas = threads_por_bloco;
  int num_colunas = (m < 1024 ? m : 1024);

  // Para cada antidiagonal por bloco
  for (int antidiag = 0; antidiag < num_antidiag; ++antidiag)
  {
    CalculaDistanciaEdicao<<<num_blocos_n, threads_por_bloco>>>
    (num_linhas, num_colunas, antidiag, n, m, d_S, d_R, d_D);

    cudaDeviceSynchronize();
  }

  cudaMemcpy(&resultado, &d_D[tam_vetor - 1], sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << resultado << std::endl;

  // Libera a matriz e as strings
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
                                      int antidiag_bloco, int n, int m,
                                      char *S, char *R, int *D)
{
  // A quantidade de colunas que se deve pular para chegar até as células do bloco
  int deslocamento_col = (antidiag_bloco * num_colunas) - (blockIdx.x * num_colunas);
  int num_antidiag = num_linhas + num_colunas - 1;
  int i = (blockIdx.x * num_linhas) + threadIdx.x + 1, j;
  int a, b, c, t, min;

  // Se o bloco é válido - Isto implica em dizer se o bloco contém parte da matriz
  if (deslocamento_col >= 0 && deslocamento_col < m)
  {
    for (int antidiag = 2; antidiag <= num_antidiag + 1; antidiag++)
    {
      j = deslocamento_col + antidiag - (threadIdx.x + 1);
      // Se é uma célula válida
      if (i <= n && j > deslocamento_col && j <= deslocamento_col + num_colunas)
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
}
