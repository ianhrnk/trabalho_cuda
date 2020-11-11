/********************************************************************
 * Aluno: Ian Haranaka | RGA: 2018.1904.009-7
 * Comando de compilação: nvcc solucao_inicial.cu -o solucao_inicial
 ********************************************************************/

#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char *argv[])
{
  std::ifstream entrada (argv[1]);
  std::string S, R;
  int n, m, resultado;

  // Leitura do arquivo de entrada
  if (entrada.is_open())
  {
    entrada >> n >> m >> S >> R;
    S.insert(S.begin(), ' ');
    R.insert(R.begin(), ' ');
    entrada.close();
  }
  else
  {
    std::cout << "Arquivo não encontrado!\n";
    return -1;
  }

  int nElementos = (n+1) * (m+1);
  int nBytes = nElementos * sizeof(int);

  // Aloca matriz no host
  int *h_D = (int *) malloc(nBytes);

  // Inicializa matriz
  for (int i = 0, j = 0; i <= nElementos; i += m+1, ++j)
    h_D[i] = j;
  for (int j = 0; j <= m; ++j)
    h_D[j] = j;

  // Aloca matriz na memória global do kernel
  int *d_D;
  cudaMalloc(&d_D, nBytes);

  // Copia a matriz do host para a GPU
  cudaMemcpy(d_D, h_D, nBytes, cudaMemcpyHostToDevice);

  // Invoca o kernel
  int threadsPerBlock = n; // No máximo 1024
  //NomeDaFuncao<<<1, threadsPerBlock>>>(<params>);

  //cudaMemcpy(resultado, d_D, sizeof(int), cudaMemcpyDeviceToHost);

  // Libera matrizes
  free(h_D);
  cudaFree(d_D);

  return 0;
}
