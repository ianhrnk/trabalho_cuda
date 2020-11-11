import os

for i in range(1,10):
  comando = './solucao_inicial ../Entradas/entrada{}.txt'.format(i)
  print('Executando sequencial {}'.format(i))
  if (os.system(comando) != 0):
    print('Entrada {} retornou valor diferente de zero!'.format(i))
    break
