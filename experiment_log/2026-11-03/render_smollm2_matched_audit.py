"""Render the completed matched-format checkpoint audit from saved scalar evidence."""
from __future__ import annotations
import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    """Write labeled comparison charts and a complete settings table without model execution."""
    repository = Path(__file__).resolve().parents[2]
    evidence = repository / 'experiment_log/2026-11-03/smollm2-matched-validation-artifacts/audit-completed.json'
    original = repository / 'experiment_log/2026-11-02/smollm2-gsm8k-research-artifacts/report/summary.json'
    output = original.parent / 'matched-audit'
    output.mkdir(exist_ok=False)
    audit = json.loads(evidence.read_text())
    prior = json.loads(original.read_text())
    configurations = {trial['trial_id']: trial['configuration'] for trial in prior['trials']}
    assert audit['status'] == 'completed' and len(audit['results']) == 19
    results = sorted(audit['results'], key=lambda record: record['trial_id'])
    rows: list[dict[str, Any]] = []
    for result in results:
        configuration = configurations[result['trial_id']]
        rows.append({'trial': result['trial_id'], 'epoch': result['epoch'], 'learning_rate': configuration['learning_rate'],
            'batch_size': configuration['batch_size'], 'zero_shot_accuracy_percent': 100 * result['zero_shot']['strict_accuracy'],
            'five_shot_accuracy_percent': 100 * result['five_shot']['strict_accuracy'],
            'zero_shot_nll': result['zero_shot']['nll'], 'five_shot_nll': result['five_shot']['nll']})
    with (output / 'settings.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    positions = list(range(len(rows)))
    axes[0].barh([position + .18 for position in positions], [row['zero_shot_accuracy_percent'] for row in rows],
                 height=.34, color='#2563eb', label='Zero-shot')
    axes[0].barh([position - .18 for position in positions], [row['five_shot_accuracy_percent'] for row in rows],
                 height=.34, color='#d97706', label='Five-shot')
    axes[0].set(yticks=positions, yticklabels=[row['trial'] for row in rows],
        xlabel='Strict exact-match accuracy (%) · higher is better', title='Same 747 validation questions, two prompt formats')
    axes[0].legend()
    for row in rows:
        axes[1].plot([row['zero_shot_nll'], row['five_shot_nll']],
            [row['zero_shot_accuracy_percent'], row['five_shot_accuracy_percent']], color='#94a3b8', alpha=.6)
    axes[1].scatter([row['zero_shot_nll'] for row in rows], [row['zero_shot_accuracy_percent'] for row in rows], color='#2563eb', label='Zero-shot')
    axes[1].scatter([row['five_shot_nll'] for row in rows], [row['five_shot_accuracy_percent'] for row in rows], color='#d97706', label='Five-shot')
    axes[1].set(xlabel='Answer-token NLL (nats) · lower is better', ylabel='Strict accuracy (%) · higher is better',
        title='Each line joins one retained checkpoint')
    axes[1].legend()
    for axis in axes: axis.grid(axis='x', alpha=.15)
    figure.savefig(output / 'audit.svg')
    figure.savefig(output / 'audit.png', dpi=170)
    plt.close(figure)
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(value) if not isinstance(value, float) else
        f'{value:.6g}') + '</td>' for value in row.values()) + '</tr>' for row in rows)
    headers = ['Checkpoint', 'Epoch', 'Learning rate', 'Batch', 'Zero-shot accuracy (%)', 'Five-shot accuracy (%)',
               'Zero-shot NLL (nats)', 'Five-shot NLL (nats)']
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>SmolLM2 GSM8K matched validation audit</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1350px;padding:0 20px;color:#172033}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:right;border-bottom:1px solid #dce2eb}th:first-child,td:first-child{text-align:left}thead{background:#eef3fa}a{color:#2563eb}.note{padding:16px;background:#eff6ff;border-radius:8px}</style></head><body>
<h1>SmolLM2-135M · matched validation audit</h1><p class="note">Completed audit of all 19 retained training-only checkpoints. The follow-up fine-tuning study is separate. These are validation results on 747 held-out training-set questions, not test results.</p>
<p>Greedy decoding, 256 generated tokens; exactly five fixed training-only demonstrations in the five-shot condition. Both NLLs score only the query answer and EOS. Every zero-shot result reproduced its original saved-checkpoint metric. Each row describes its retained accuracy-selected checkpoint, which can differ from its minimum-loss epoch.</p>
<img src="audit.svg" alt="Paired zero-shot and five-shot accuracy bars, with accuracy versus answer-token NLL for all nineteen checkpoints">
<p><a href="settings.csv">Download settings and metrics CSV</a> · <a href="audit.png">Download chart PNG</a> · <a href="../comparison-v1/">Original full settings report</a></p><table><thead><tr>'''
    page += ''.join('<th>' + heading + '</th>' for heading in headers) + '</tr></thead><tbody>' + table + '</tbody></table>'
    page += '<p>Validation protocol SHA-256: <code>' + audit['protocol_sha256'] + '</code></p></body></html>'
    (output / 'index.html').write_text(page)
    (output / 'evidence.json').write_text(json.dumps({'source_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
        'validation_examples': 747, 'checkpoint_count': 19, 'rows': rows}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
