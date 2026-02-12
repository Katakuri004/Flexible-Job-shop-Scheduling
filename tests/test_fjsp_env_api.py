from pathlib import Path

from src.fjsp_env import FJSPEnv, FJSPEnvConfig
from src.seed_utils import SeedConfig


def _run_fixed_policy_episode(env: FJSPEnv) -> tuple[dict, float, bool, dict]:
    """
    Helper: run a full episode by always picking the first feasible action.
    """
    obs = env.reset()
    done = False
    reward = 0.0
    info: dict = {}

    # Basic structure checks on initial observation.
    assert "jobs" in obs
    assert "machines" in obs
    assert "feasible_actions" in obs
    assert "action_mask" in obs
    assert len(obs["feasible_actions"]) > 0

    while not done:
        obs, r, done, info = env.step(action=0)
        reward += r

    return obs, reward, done, info


def test_fjsp_env_reset_and_step_api() -> None:
    """
    Basic API test for step-wise environment:
    - reset() returns structured observation with jobs/machines/feasible_actions.
    - step() consumes an action index and eventually terminates with a schedule.
    """

    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=999),
    )
    env = FJSPEnv(cfg)

    obs_final, reward, done, info = _run_fixed_policy_episode(env)

    assert done is True
    assert isinstance(reward, float)
    assert "schedule" in info
    assert env.last_makespan is not None
    assert env.last_schedule is not None
    assert env.last_makespan == -reward
    assert env.last_schedule == info["schedule"]


def test_fjsp_env_deterministic_replay_with_seed() -> None:
    """
    Determinism test:
    - With the same seed configuration and a fixed policy (always action 0),
      two runs should produce identical final makespan and identical schedules.
    """

    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=1234),
    )

    # First run
    env1 = FJSPEnv(cfg)
    obs1, reward1, done1, info1 = _run_fixed_policy_episode(env1)

    # Second run
    env2 = FJSPEnv(cfg)
    obs2, reward2, done2, info2 = _run_fixed_policy_episode(env2)

    assert done1 and done2
    assert reward1 == reward2
    assert info1["schedule"] == info2["schedule"]

