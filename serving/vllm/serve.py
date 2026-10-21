"""serve_config.yaml을 읽어 vllm serve 명령을 구성하고 실행한다."""
import argparse
import os
import subprocess
from pathlib import Path

import yaml


def build_command(cfg: dict) -> list[str]:
    """config의 model과 args를 vllm serve CLI 인자 목록으로 변환한다."""
    cmd = ["vllm", "serve", cfg["model"]]
    for key, value in cfg["args"].items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            cmd.append(flag)
        elif value is not False and value is not None:
            cmd += [flag, str(value)]
    return cmd


def main() -> None:
    """config 경로를 인자로 받아 지정된 GPU에서 vLLM 서버를 띄운다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("serve_config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["cuda_visible_devices"])}
    cmd = build_command(cfg)
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
