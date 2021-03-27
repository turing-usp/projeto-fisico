# Projeto Físico

Para rodar o arquivo "train_rllib.py", é necessário instalar a versão 0.20 do `mlagents`.
No PC do Turing:

```bash
# Instala as bibliotecas necessárias
apt get update
apt get install cmake
pip install gym ray[rllib] mlagents==0.20 torch

# Baixa o arquivo do jogo
pip install gdown
gdown --id 1ZlUJ3vo5tRtHNyGVdh_p8CqBNelCnCEQ --output CarAgentRL.tar.gz
tar xf CarAgentRL.tar.gz

# Roda o treinamento
python src/train_rllib.py -f CarAgentRL/CarAgentRL
```
