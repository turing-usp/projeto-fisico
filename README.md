# Projeto Físico

Para rodar o arquivo "train_rllib.py", é necessário instalar a versão 0.20 do `mlagents`.
No PC do Turing:

```bash
# Instala as bibliotecas necessárias
sudo apt update
sudo apt install cmake
pip install gym ray[rllib] mlagents==0.20 torch

# Baixa o arquivo do jogo
pip install gdown
gdown --id 15ocBljC4BKiquJfQ3nCg49ZEoqyVZUm1 --output CarAgentRL.tar.gz
tar xf CarAgentRL.tar.gz

# Roda o treinamento
python src/train_rllib.py -f CarAgentRL/CarAgentRL
```
