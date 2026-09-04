from __future__ import annotations

import json
import sys
from pathlib import Path

# Support both `python -m scripts.run_demo` and `python scripts/run_demo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.workflow import AlgorithmFactoryWorkflow
from scripts.generate_demo_data import generate


def main() -> None:
    settings = get_settings()
    data_path = settings.data_dir / "churn_demo.csv"
    if not data_path.exists():
        generate(data_path)
    description = "根据客户年龄、地区、登录频率、历史消费金额和投诉次数预测未来30天是否流失，要求ROC-AUC不低于0.75，并输出流失概率。"
    result = AlgorithmFactoryWorkflow(settings).run(description, data_path)
    print(json.dumps({
        "run_id": result.run_id,
        "status": result.validation.status if result.validation else None,
        "algorithm": result.validation.algorithm if result.validation else None,
        "metrics": result.validation.metrics if result.validation else {},
        "report": result.report_markdown,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
