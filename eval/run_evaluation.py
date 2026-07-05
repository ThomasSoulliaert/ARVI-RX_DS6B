from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import predict_with_model, toy_predict
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics
from src.database import insert_run, init_db


def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(mode: str, db_path: Path, cases_file: Path, use_real_model: bool = False) -> tuple[list[dict], dict]:
    cases = read_cases(cases_file)
    rows = []
    init_db(db_path)
    # "rsna" evaluation attempts real MedGemma inference via predict_with_model,
    # which itself falls back to the deterministic toy predictor if the real VLM
    # is unavailable (no GPU/token) instead of crashing the eval run. The toy/
    # baseline/improved modes keep calling toy_predict directly (unchanged,
    # fast, deterministic) for backward compatibility with the synthetic set.
    predictor = predict_with_model if use_real_model else toy_predict
    total = len(cases)
    for i, case in enumerate(cases, start=1):
        image_path = ROOT / case['image_path']
        if use_real_model:
            case_start = time.perf_counter()
            print(f"[{mode}] {i}/{total} {case['case_id']} ...", end=" ", flush=True)
        pred = apply_safety_guardrails(predictor(image_path, mode=mode))
        if use_real_model and pred.get("is_toy"):
            # Garde anti-fallback : si le vrai VLM est indisponible, predict_with_model
            # retombe silencieusement sur le modèle jouet. Des métriques "RSNA"
            # calculées sur le jouet seraient mensongères -> on arrête tout.
            raise SystemExit(
                f"[{mode}] {case['case_id']}: le modèle JOUET a répondu à la place de "
                "MedGemma (GPU ou token Hugging Face indisponible). Évaluation RSNA "
                "interrompue pour ne pas produire de métriques invalides. "
                "Configurer l'accès au modèle puis relancer."
            )
        if use_real_model:
            elapsed = time.perf_counter() - case_start
            print(f"{pred['predicted_class']} ({elapsed:.1f}s)", flush=True)
        valid, errors = validate_prediction(pred)
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'json_valid': valid,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
        }
        rows.append(row)
        insert_run(db_path, case['case_id'], str(image_path), pred)
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['toy', 'baseline', 'improved', 'rsna'], default='toy')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'medical_ai_evidence.sqlite')
    parser.add_argument('--cases-file', type=Path, default=None)
    parser.add_argument(
        '--prompts', nargs='+', default=None,
        help="Prompts à évaluer (ex. --prompts improved_v2). Par défaut : baseline improved.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    use_real_model = args.mode == 'rsna'
    if args.prompts is not None:
        modes = args.prompts
    elif args.mode in ('toy', 'rsna'):
        modes = ['baseline', 'improved']
    else:
        modes = [args.mode]
    if args.cases_file is not None:
        cases_file = args.cases_file
    elif args.mode == 'rsna':
        cases_file = ROOT / 'data' / 'rsna' / 'rsna_catalogue.csv'
    else:
        cases_file = ROOT / 'data' / 'synthetic_cases.csv'
    summary = []
    for mode in modes:
        rows, metrics = run(mode, args.db_path, cases_file, use_real_model=use_real_model)
        write_csv(out_dir / f'{mode}_predictions.csv', rows)
        (out_dir / f'{mode}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        summary.append({'mode': mode, **metrics})
    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
