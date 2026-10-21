# vLLM serving

`agent/` 노트북에서 쓰는 로컬 OpenAI 호환 LLM 서버 설정입니다.

## 설치 (처음 한 번)

```bash
cd ~/ai-lab/serving/vllm
uv sync          # 이 폴더에 .venv 생성 (git에는 올라가지 않음)
```

## 실행

```bash
uv run python serve.py                              # serve_config.yaml 사용
uv run python serve.py --config other_config.yaml   # 다른 설정 파일 사용
```

준비 확인: `curl localhost:8000/v1/models`

## 설정 (`serve_config.yaml`)

| 키 | 설명 |
|---|---|
| `cuda_visible_devices` | 서버가 사용할 GPU 번호 |
| `model` | Hugging Face 모델 ID |
| `args` | `vllm serve` 옵션 (`snake_case` → `--kebab-case`로 변환, `true`는 플래그만 추가) |

- `gpu_memory_utilization: 0.8`: 다른 작업이 GPU를 일부 쓰고 있어도 뜰 수 있도록 기본값 0.9보다 낮춤
- `enable_auto_tool_choice` + `tool_call_parser: openai`: `research_agent.ipynb`의 tool calling에 필요

첫 실행 시 모델 다운로드와 FlashInfer 커널 컴파일 때문에 10분 이상 걸릴 수 있습니다 (이후 `~/.cache`에 캐시됨).
