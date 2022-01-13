# Documentação

As seções seguintes descrevem como, em linhas gerais,
o {class}`~car_env.core.CarEnv` deve ser utilizado. Para uma descrição
mais detalhada, incluindo classes e funções utilizadas internamente
na implementação do ambiente, veja a [documentação da API](api.rst).

## Usando o ambiente com o rllib

O ambiente pode ser utilizado com a [API de treinamento do rllib](https://docs.ray.io/en/latest/rllib-training.html#basic-python-api).
Para um exemplo disso, veja o arquivo src/train.py.

### Configuração do ambiente

O ambiente pode ser configurado e customizado de duas formas:

- por meio de um conjunto pré-definido de configurações do ambiente
  (veja {attr}`car_env.config_side_channel.FIELDS` para a lista de configurações
  suportadas); ou
- por meio de _wrappers_, que permitem que outras operações sejam
  implementadas pelo usuário ({mod}`car_env.wrapper`).

Para exemplos de wrappers, veja o módulo {mod}`wrappers`.

Essas configurações são aplicadas por meio do dicionário de configurações do rllib:

```python
config = {
    "env_config": EnvConfig(
        file_name=...,
        wrappers=[
            wrappers.CheckpointReward,
            wrappers.VelocityReward,
            ...
        ],

        # Curriculo de treinamento
        curriculum=[
            # Fase 0 (configurações iniciais)
            {
                # Configurações do unity
                "unity_config": {
                    "AgentCount": 10,
                    ...
                },

                # Configurações dos wrappers
                "wrappers": {
                    "CheckpointReward": {
                        "max_reward": 100,
                        ...
                    },
                    ...
                },
            },

            # Fase 1
            {
                # Para que essa fase se inicie, as condições abaixo
                # deve ser satisfeitas por pelo menos `for_iterations` iterações.
                "when": {
                    "custom_metrics/agent_checkpoints_mean": 5,
                    "agent_steps_this_phase": 1_000_000,
                },
                "for_iterations": for_iterations,

                # Configurações do unity
                "unity_config": {
                    ...
                },

                # Configurações dos wrappers
                "wrappers": {
                    ...
                },
            },

            ...
        ],
    ),
}
```

### Schedulers

TODO

## Outros módulos utilizados internamente

Ver a [documentação da API](api.rst).
