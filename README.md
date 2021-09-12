# Projeto Físico

## Instalação
Para realizar o treino é necessário instalar a versão 0.2 do mlagents. Isto é feito através do seguinte comando:

```bash
# Instala as bibliotecas necessárias
sudo apt update
sudo apt install cmake
pip install gym ray[rllib] mlagents==0.20 torch
```
## Baixando os binários

Para executar o projeto é necessário baixar os binários dos agentes, para isto usaremos o [gdown](https://pypi.org/project/gdown/)

```bash
pip install gdown
```
Em seguida, utiliza-se o gdown para baixar do google drive os arquivos necessários:
```bash
gdown --id 15ocBljC4BKiquJfQ3nCg49ZEoqyVZUm1 --output CarAgentRL.tar.gz
tar xf CarAgentRL.tar.gz
```

## Treinamento
Para treinar os agentes basta executar o arquivo ```train_rllib.py``` da seguinte forma:

```bash
# Roda o treinamento
python src/train_rllib.py -f CarAgentRL/CarAgentRL
```
### Flags
Algumas flags podem ser passadas para o script de treino, elas são:

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Função&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Tipo |     Default     |
|-----------------|--------------------------------------------------------|---------------------------------|---------------------|
|--file_name, -f  |Indica local e nome do arquivo de agentes a ser treinado|String                           |Nada                 |
|--log_level, -l  |Define a forma como o treino vai gerar um registro      |String (DEBUG, INFO, WARN, ERROR)|WARN                 |
|--agents         |Número total de agentes a serem executados              |Int                              |256                  |
|--workers        |Número total de CPU's a serem utilizadas no treino      |Int                              |[Número de CPU's] - 1|
|--gpus           |Número de GPU's a serem utilizadas no treino            |Int                              |Todas as disponíveis |
|--max_train_iters|Número de iterações de treino a serem executadas        |Int                              |512                  |
|--time_scale     |Quão rápido o jogo será executado                       |Float                            |1000                 |
|--framework      |Escolhe qual framework utilizar entre Torch e TensorFlow|String (torch, tf)               |torch                |