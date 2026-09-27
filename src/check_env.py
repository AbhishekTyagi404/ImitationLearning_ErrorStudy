"""BRICK 1: does my install work?  Run:  python src/check_env.py

Prints what the robot sees and does, then lets the expert driver make one trip.
If the last line says ALL GOOD, Brick 1 is done.
"""
import numpy as np

from common import ENV_ID, MAX_STEPS, ExpertDriver, make_env


def main():
    env = make_env()
    print(f"Environment : {ENV_ID}")
    print(f"Action space: {env.action_space}   (a push in x and y)")
    print(f"Obs space   : {env.observation_space['observation']}   ([x, y, vx, vy])")
    print("Maze map (1 = wall, 0 = free):")
    print(np.array(env.unwrapped.maze.maze_map))

    obs, info = env.reset(seed=0)
    print("\nFirst observation :", obs["observation"].round(3))
    print("Goal position     :", obs["desired_goal"].round(3))

    driver = ExpertDriver(env)
    driver.plan(obs["observation"][:2], obs["desired_goal"])
    for step in range(1, MAX_STEPS + 1):
        obs, reward, terminated, truncated, info = env.step(driver.act(obs["observation"]))
        if terminated or truncated:
            break
    print(f"\nExpert finished after {step} steps, success = {info['success']}")
    assert info["success"], "Expert did not reach the goal: something is wrong with the install"
    print("ALL GOOD")


if __name__ == "__main__":
    main()
