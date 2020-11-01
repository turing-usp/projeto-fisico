from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def main():
    unity_env = UnityEnvironment(base_port=5004)
    env = UnityToGymWrapper(unity_env, uint8_visual=True)
    env = Monitor(env)

    model = PPO('CnnPolicy', env, n_steps=512, gamma=.95,
                tensorboard_log="./tensorboard/", verbose=1)
    model.learn(total_timesteps=100_000, tb_log_name='PPO')
    model.save("ppo")
    # model.load("ppo")

    print('Done training')

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    env.close()


if __name__ == '__main__':
    main()
