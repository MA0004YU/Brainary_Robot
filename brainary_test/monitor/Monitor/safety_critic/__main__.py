"""Safety Critic 独立入口：python -m Monitor.safety_critic

吃现成的 output/*.json（memory_snapshot + planned_actions），逐动作跑安全裁判，
写 output/safety_critic_review.json。

用法（在仓库根目录）：
    conda activate biea_ssp
    export OPENAI_API_KEY=...           # 或 API_zhongzhuan
    python -m Monitor.safety_critic

    # 无 key 时的离线结构自检（用假 LLM，不发请求，仅验证管线/输入构造）：
    python -m Monitor.safety_critic --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- 路径 bootstrap（保证 Monitor.* 可 import；critic 不依赖 ssp_pkg）---
_SC_DIR = Path(__file__).resolve().parent           # Monitor/safety_critic
_MONITOR_DIR = _SC_DIR.parent                        # Monitor
_REPO_ROOT = _MONITOR_DIR.parent                     # 仓库根
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Monitor.safety_critic.critic_runner import run_safety_critic  # noqa: E402


class _DryRunLLM:
    """离线假 LLM：不发请求，一律回 'not malicious'，用于结构自检。"""

    def __init__(self) -> None:
        self.total_tokens = 0

    def call(self, prompt: str) -> str:  # noqa: ARG002
        return "not malicious (dry-run: no LLM called)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safety Critic 独立运行")
    ap.add_argument("--output-dir", default=str(_REPO_ROOT / "output"),
                    help="output 目录（读 snapshot/planned_actions，写 safety_critic_review.json）")
    ap.add_argument("--dry-run", action="store_true",
                    help="用假 LLM 离线自检（不需要 API key，不发请求）")
    ap.add_argument("--model", default=None, help="覆盖模型（默认 gpt-4o / SAFETY_CRITIC_MODEL）")
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    snapshot = out / "memory_snapshot.json"
    planned = out / "planned_actions.json"
    review_path = out / "safety_critic_review.json"

    if not snapshot.exists() or not planned.exists():
        print(f"[Critic] 缺少 {snapshot} 或 {planned}，无法运行。", file=sys.stderr)
        return 2

    if args.dry_run:
        print("[Critic] DRY-RUN：用假 LLM，仅验证管线与输入构造（不发请求）")
        llm = _DryRunLLM()
    else:
        from Monitor.safety_critic.pipeline_llm import PipelineLLMClient
        llm = PipelineLLMClient(model=args.model)

    result = run_safety_critic(
        snapshot_path=snapshot,
        planned_actions_path=planned,
        output_path=review_path,
        llm=llm,
    )

    s = result.summary()
    print(f"[Critic] 评价 {s['num_steps']} 个动作 -> 总体: {s['overall']}, "
          f"malicious: {s['num_malicious']}, tokens: {s['total_tokens']}")
    if s["first_halt_index"] is not None:
        print(f"[Critic] ⚠️ 按原语义会在第 {s['first_halt_index']} 步中断")
    print(f"[Critic] 已写: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
