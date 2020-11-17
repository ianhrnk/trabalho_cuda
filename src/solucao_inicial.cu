/********************************************************************************
 * Trabalho 2: Programação Paralela para Processador Many-core (GPU) Usando CUDA
 * Professora: Nahri Moreano
 * Aluno: Ian Haranaka | RGA: 2018.1904.009-7
 * Comando de compilação: nvcc solucao_inicial.cu -o solucao_inicial
 ********************************************************************************/

#include <iostream>
#include <fstream>

const int MAX_THREADS = 1024;

char *AlocaSequencia(int n);
__global__ void InicializaVetor(int n, int m, int *D);
__global__ void CalculaDistanciaEdicao(int n, int m, char *S, char *R, int *D);

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

  // Aloca o vetor e as strings na memória global da GPU
  int *d_D;
  char *d_S, *d_R;
  cudaMalloc(&d_D, tam_vetor * sizeof(int));
  cudaMalloc(&d_S, (n+2) * sizeof(char));
  cudaMalloc(&d_R, (m+2) * sizeof(char));

  // Copia as strings para a memória global da GPU
  cudaMemcpy(d_S, h_S, (n+2) * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, h_R, (m+2) * sizeof(char), cudaMemcpyHostToDevice);

  // Invoca o kernel
  int threads_por_bloco = n;
  if (n > MAX_THREADS)
    threads_por_bloco = MAX_THREADS;

  InicializaVetor<<<1, 1>>>(n, m, d_D);
  cudaDeviceSynchronize();

  CalculaDistanciaEdicao<<<1, threads_por_bloco>>>(n, m, d_S, d_R, d_D);
  cudaDeviceSynchronize();

  cudaMemcpy(&resultado, &d_D[tam_vetor - 1], sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << resultado << std::endl;

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

__global__ void InicializaVetor(int n, int m, int *D)
{
  for (int i = 0, j = 0; i < (n+1) * (m+1); i += m+1, ++j)
    D[i] = j;
  for (int j = 0; j <= m; ++j)
    D[j] = j;
}

__global__ void CalculaDistanciaEdicao(int n, int m, char *S, char *R, int *D)
{
  int num_iteracoes = (n-1) / MAX_THREADS + 1;
  int num_anti_diag, deslocamento;
  int a, b, c, t, min;
  int i, j;

  for (int it = 0; it < num_iteracoes; ++it)
  {
    deslocamento = it * MAX_THREADS;
    i = threadIdx.x + deslocamento + 1;

    if (n - deslocamento > MAX_THREADS)
      num_anti_diag = MAX_THREADS + m - 1;
    else
      num_anti_diag = n - deslocamento + m - 1;

    for (int anti_diag = 2; anti_diag <= num_anti_diag + 1; anti_diag++)
    {
      j = anti_diag - (threadIdx.x + 1);
      // Se é uma célula válida
      if (i <= n && j > 0 && j <= m)
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
