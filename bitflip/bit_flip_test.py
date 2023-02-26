from bit_flip import Bit_Flip_Env

if __name__ == "__main__":
    env = Bit_Flip_Env()

    for _ in range(1):
        env.reset()
        done = False
        print(f"Goal State: {env._desired_goal}")
        while not done:
            action = env.action_space_sampler()
            obs, r, done, info = env.step(action)
            print(env._steps)
            env.render()
            