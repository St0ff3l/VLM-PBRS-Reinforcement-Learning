import argparse
import os
import sys

from main import run_experiment, DEFAULT_VLM_MSE_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the VLM-PPO experiment (no baseline run)."
    )
    parser.add_argument("--env-id", type=str, default="MountainCar-v0")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--run-name-prefix", type=str, default="vlm_only_fast")
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=5.0,
        help="MSE novelty threshold for VLM query trigger (lowered for 8x8 proto-cache).",
    )
    parser.add_argument("--vlm-call-every-n", type=int, default=32, help="Cooldown threshold for queries.")
    parser.add_argument("--log-every-vlm-calls", type=int, default=64)
    parser.add_argument(
        "--vlm-query-on-reset",
        action="store_true",
        help="If set, query VLM once on every env reset (slower but denser shaping).",
    )
    parser.add_argument(
        "--vlm-alpha",
        type=float,
        default=250.0,
        help="Scaling factor for VLM shaping (alpha * (gamma Phi' - Phi)).",
    )
    parser.add_argument(
        "--env-reward-mode",
        type=str,
        choices=["preserve", "suppress_step"],
        default="preserve",
        help="How to treat original environment reward: preserve or suppress_step.",
    )
    parser.add_argument(
        "--expected-conda-env",
        type=str,
        default="lab",
        help="Expected conda env name. Use empty string to disable the check.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run final evaluation and save video after training (slower).",
    )
    return parser.parse_args()


def ensure_python_environment(expected_conda_env: str) -> None:
    if not expected_conda_env:
        return

    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    expected_path_token = f"\\envs\\{expected_conda_env}\\"
    interpreter_ok = expected_path_token.lower() in sys.executable.lower()

    if active_env == expected_conda_env or interpreter_ok:
        return

    print("[ENV CHECK] Wrong Python environment detected.")
    print(f"[ENV CHECK] Current interpreter: {sys.executable}")
    print(f"[ENV CHECK] CONDA_DEFAULT_ENV: {active_env}")
    print(f"[ENV CHECK] Expected conda env: {expected_conda_env}")
    print("[ENV CHECK] Please run with your conda env interpreter, for example:")
    print(r"C:\Users\Stoffel\.conda\envs\lab\python.exe main_vlm_only.py")
    raise SystemExit(1)


def main() -> None:
    args = parse_args()
    ensure_python_environment(args.expected_conda_env)
    run_experiment(
        env_id=args.env_id,
        use_vlm=True,
        total_timesteps=args.timesteps,
        run_name_prefix=args.run_name_prefix,
        vlm_mse_threshold=args.mse_threshold,
        vlm_call_every_n=args.vlm_call_every_n,
        vlm_log_every_n_calls=args.log_every_vlm_calls,
        vlm_query_on_reset=args.vlm_query_on_reset,
        vlm_alpha=args.vlm_alpha,
        env_reward_mode=args.env_reward_mode,
        run_evaluation=args.run_eval,
    )


if __name__ == "__main__":
    main()
