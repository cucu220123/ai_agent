from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, get_settings
from app.experience.closed_loop import ClosedLoopAnalyzer
from app.workflow import AlgorithmFactoryWorkflow


def main() -> None:
    base = get_settings()
    settings = Settings(**{**base.__dict__, "llm_provider": "mock"})
    workflow = AlgorithmFactoryWorkflow(settings)
    spec_text = "根据客户年龄、地区、登录频率、消费金额和投诉次数预测客户是否流失，要求 ROC-AUC 不低于 0.75"
    analyzer = ClosedLoopAnalyzer(workflow.store)
    before = analyzer.before_after(workflow.parser.run(spec_text, str(settings.data_dir / "churn_demo.csv")))
    result = workflow.run(spec_text, settings.data_dir / "churn_demo.csv")
    after = analyzer.after(result.spec, result.run_id)
    print(json.dumps({"run_id": result.run_id, **before, **after, "closed_loop_evidence": analyzer.evidence(before, after, result.run_id), "planning_comparison": analyzer.compare_planning(result.spec, before, after)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
