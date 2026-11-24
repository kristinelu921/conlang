"""Render scalar search and final benchmark evidence as a reproducible local report."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def validation_key(record: dict[str, Any], reference: dict[str, Any]) -> tuple[Any, ...] | None:
    """Reconstruct the declared ranking solely to explain recorded validation outcomes."""
    relative = max(record['zero_shot_nll'] / reference['zero_shot_nll'], record['five_shot_nll'] / reference['five_shot_nll'])
    if (record['zero_shot_correct'] < reference['zero_shot_correct'] or
            record['five_shot_correct'] < reference['five_shot_correct'] or relative > 1.10):
        return None
    return (-min(record['zero_shot_correct'], record['five_shot_correct']),
        -(record['zero_shot_correct'] + record['five_shot_correct']), relative, record['epoch'], record['trial_id'])


def main() -> None:
    """Create an immutable labeled report version from a saved scalar-only snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--search-id')
    arguments = parser.parse_args()
    if not arguments.version or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for character in arguments.version):
        raise ValueError('Use a simple version identity.')
    repository = Path(__file__).resolve().parents[2]
    report_root = repository / 'experiment_log/2026-11-02/smollm2-gsm8k-research-artifacts/report'
    output = report_root / ('matched-study-' + arguments.version)
    output.mkdir(exist_ok=False)
    snapshot = json.loads(arguments.snapshot.read_text())
    old = json.loads((report_root / 'summary.json').read_text())
    searches = [search for search in snapshot['searches'] if 'frozen-reference' in search
                and (arguments.search_id is None or search['id'] == arguments.search_id)]
    if len(searches) != 1:
        raise ValueError('Select the authoritative search explicitly if a recovery lineage has multiple references.')
    search = searches[0]
    reference = search['frozen-reference']
    # Export folders also contain config.json, but their records describe models rather than trials.
    training_trials = [trial for trial in search['trials']
                       if trial['id'] not in ('balanced-final-flax', 'balanced-final-model-fp32')]
    lineage_trials = list(training_trials)
    if 'recovered-predecessor' in search:
        predecessor_name = Path(search['recovered-predecessor']['path']).name
        predecessor = next(item for item in snapshot['searches'] if item['id'] == predecessor_name)
        if predecessor['frozen-reference'] != reference:
            raise ValueError('Recovery changed the frozen scientific reference.')
        lineage_trials = predecessor['trials'] + lineage_trials
    trials = [trial for trial in lineage_trials if not trial['configuration']['final_fit']]
    rows = []
    figure, axes = plt.subplots(3, 2, figsize=(15, 15), layout='constrained')
    for index, trial in enumerate(trials):
        configuration = trial['configuration']
        epochs = trial['epochs']
        if not epochs:
            if trial.get('summary', {}).get('status') == 'failed':
                rows.append({'trial': trial['id'], 'learning_rate': configuration['learning_rate'], 'batch_size': configuration['batch_size'], 'global_microbatch_size': configuration['batch_size'] // configuration['gradient_accumulation'], 'gradient_accumulation': configuration['gradient_accumulation'],
                    'five_shot_training_percent': 100 * configuration['five_shot_training_fraction'], 'seed': configuration['seed'],
                    'completed_epochs': 0, 'shown_epoch': None, 'zero_shot_validation_accuracy_percent': None,
                    'five_shot_validation_accuracy_percent': None, 'zero_shot_validation_nll': None, 'five_shot_validation_nll': None,
                    'balanced_eligible': False, 'epoch_choice': 'failed before first optimizer update', 'status': 'failed',
                    'wandb_url': trial.get('wandb', {}).get('url', '')})
            continue
        color = plt.get_cmap('tab10')(index % 10)
        label = trial['id'] + ' · ' + str(int(100 * configuration['five_shot_training_fraction'])) + '% five-shot'
        for axis, field, scale in ((axes[0, 0], 'zero_shot_nll', 1), (axes[0, 1], 'five_shot_nll', 1),
                                   (axes[1, 0], 'zero_shot_correct', 100 / 747), (axes[1, 1], 'five_shot_correct', 100 / 747)):
            axis.plot([epoch['epoch'] for epoch in epochs], [epoch[field] * scale for epoch in epochs],
                      marker='o', markersize=3, color=color, label=label)
        eligible = [epoch for epoch in epochs if validation_key(epoch, reference) is not None]
        chosen = min(eligible, key=lambda epoch: validation_key(epoch, reference)) if eligible else min(epochs, key=lambda epoch: epoch['zero_shot_nll'])
        rows.append({'trial': trial['id'], 'learning_rate': configuration['learning_rate'], 'batch_size': configuration['batch_size'], 'global_microbatch_size': configuration['batch_size'] // configuration['gradient_accumulation'], 'gradient_accumulation': configuration['gradient_accumulation'],
            'five_shot_training_percent': 100 * configuration['five_shot_training_fraction'], 'seed': configuration['seed'],
            'completed_epochs': len(epochs), 'shown_epoch': chosen['epoch'],
            'zero_shot_validation_accuracy_percent': 100 * chosen['zero_shot_correct'] / 747,
            'five_shot_validation_accuracy_percent': 100 * chosen['five_shot_correct'] / 747,
            'zero_shot_validation_nll': chosen['zero_shot_nll'], 'five_shot_validation_nll': chosen['five_shot_nll'],
            'balanced_eligible': bool(eligible), 'epoch_choice': 'best eligible balance' if eligible else 'minimum zero-shot NLL; no eligible epoch',
            'status': trial.get('summary', {}).get('status', 'running'), 'wandb_url': trial.get('wandb', {}).get('url', '')})
    for axis, title, ylabel in ((axes[0, 0], 'Zero-shot validation loss', 'Answer-token NLL (nats) · lower is better'),
        (axes[0, 1], 'Five-shot validation loss', 'Answer-token NLL (nats) · lower is better'),
        (axes[1, 0], 'Zero-shot validation accuracy', 'Strict exact match (%) · higher is better'),
        (axes[1, 1], 'Five-shot validation accuracy', 'Strict exact match (%) · higher is better')):
        axis.set(title=title, xlabel='Completed optimization epochs', ylabel=ylabel)
        axis.legend(fontsize=7)
        axis.grid(alpha=.15)
    axes[0, 0].axhline(1.1 * reference['zero_shot_nll'], color='#991b1b', linestyle=':', label='Eligibility ceiling')
    axes[0, 1].axhline(1.1 * reference['five_shot_nll'], color='#991b1b', linestyle=':')
    refit_searches = [item for item in snapshot['searches'] if 'locked-selection' in item]
    for refit_search in refit_searches:
        for trial in refit_search['trials']:
            if trial['id'] in ('balanced-final-flax', 'balanced-final-model-fp32') or not trial['configuration']['final_fit']:
                continue
            curve = trial['loss_curves']
            label = 'Loss refit' if trial['configuration'].get('selection_objective') == 'minimum_zero_shot_validation_nll' else 'Balanced refit'
            axes[2, 0].plot([point['epoch'] for point in curve], [point['train/loss'] for point in curve], label=label + ' · train')
            axes[2, 0].plot([point['epoch'] for point in curve], [point['test/loss'] for point in curve], label=label + ' · test', linestyle='--')
    axes[2, 0].set(title='Final refits: train and diagnostic test loss', xlabel='Completed epochs',
                   ylabel='Answer-token NLL (nats) · lower is better')
    if axes[2, 0].lines:
        axes[2, 0].legend()
    else:
        axes[2, 0].text(.5, .5, 'Final refit not yet recorded', ha='center', transform=axes[2, 0].transAxes)
    comparisons = []
    for item in old['loss_comparison']['results']:
        name = item['name']
        if name not in ('research_accuracy', 'research_loss'):
            continue
        comparisons.append({'model': 'Previous accuracy choice' if name == 'research_accuracy' else 'Previous loss choice',
            'train_nll': item['train']['loss'], 'test_nll': item['test']['loss'],
            'zero_shot_accuracy_percent': 100 * item['zero_shot_test']['strict_accuracy'],
            'five_shot_accuracy_percent': 100 * old['benchmarks'][name]['measurements'][0]['accuracy']})
    for refit_search in refit_searches:
        completion = refit_search.get('study-completed')
        if completion is None:
            continue
        if any(refit_search[name]['status'] != 'passed' for name in ('test-0shot-verification', 'test-5shot-verification', 'export-verification')):
            raise ValueError('Completed benchmark lacks successful export/coverage/prompt verification.')
        label = 'Matched loss refit' if refit_search['locked-selection'].get('selection_objective') == 'minimum_zero_shot_validation_nll' else 'Matched balanced refit'
        comparisons.append({'model': label, 'train_nll': completion['final_train_loss'],
            'test_nll': completion['final_test_loss'],
            'zero_shot_accuracy_percent': 100 * completion['benchmarks']['0']['exact_match,strict-match'],
            'five_shot_accuracy_percent': 100 * completion['benchmarks']['5']['exact_match,strict-match']})
    positions = list(range(len(comparisons)))
    axes[2, 1].bar([position - .18 for position in positions], [row['zero_shot_accuracy_percent'] for row in comparisons], .34, label='Zero-shot')
    axes[2, 1].bar([position + .18 for position in positions], [row['five_shot_accuracy_percent'] for row in comparisons], .34, label='Five-shot')
    axes[2, 1].set(title='Full test benchmark · 1,319 questions', ylabel='Strict exact-match accuracy (%)',
        xticks=positions, xticklabels=[row['model'].replace(' ', '\n', 1) for row in comparisons])
    axes[2, 1].legend()
    for extension in ('svg', 'png'):
        figure.savefig(output / ('study.' + extension), dpi=170)
    plt.close(figure)
    for name, values in (('settings', rows), ('final-comparison', comparisons)):
        with (output / (name + '.csv')).open('x') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    def table(values: list[dict[str, Any]]) -> str:
        """Present all scalar fields with explicit table headings and safe link rendering."""
        heading = '<tr>' + ''.join('<th>' + html.escape(key.replace('_', ' ')) + '</th>' for key in values[0]) + '</tr>'
        body = []
        for row in values:
            cells = []
            for key, value in row.items():
                text = '—' if value is None else f'{value:.6g}' if isinstance(value, float) else str(value)
                if key == 'wandb_url' and value:
                    cells.append('<td><a href="' + html.escape(str(value), quote=True) + '">W&B</a></td>')
                else:
                    cells.append('<td>' + html.escape(text) + '</td>')
            body.append('<tr>' + ''.join(cells) + '</tr>')
        return '<div class="scroll"><table><thead>' + heading + '</thead><tbody>' + ''.join(body) + '</tbody></table></div>'
    status = snapshot.get('study-terminal', {}).get('status', 'running')
    if any('study-completed' not in item for item in refit_searches):
        status = 'running: final refit or evaluation pending'
    page = '<!doctype html><html><head><meta charset="utf-8"><title>SmolLM2 GSM8K matched fine-tuning</title><style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#172033}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:9px;border-bottom:1px solid #dce2eb;text-align:right}th{background:#eef3fa}a{color:#2563eb}.note{padding:15px;background:#eff6ff}</style></head><body>'
    page += '<h1>SmolLM2-135M · GSM8K matched-format fine-tuning</h1><p class="note">Status: ' + html.escape(status) + '. Snapshot version: ' + arguments.version + '.</p>'
    page += '<p>Optimization uses 6,726 original training questions; 747 held-out training questions select recipes. The final refit uses all 7,473 training questions. Test results are diagnostic: the selected recipe and epoch count are locked before test access. The historical test set has already been observed, so these are exploratory comparisons.</p>'
    page += '<p>Eligibility requires both validation accuracy counts to meet the fixed epoch-three reference and both validation NLLs to stay within 10% of their corresponding reference. Ranking maximizes the weaker-format accuracy, then the average accuracy, then minimizes relative loss. Dotted red loss levels show eligibility ceilings. Lower training loss alone cannot promote a checkpoint.</p>'
    page += '<p>The separate loss refit, when present, uses the minimum zero-shot validation-NLL recipe locked before test access. Its objective differs from the balanced eligibility rule above. Test results do not choose between these models.</p>'
    page += '<img src="study.svg" alt="Validation loss and accuracy by epoch for both prompt formats, final train and test loss, and full test benchmark comparison">'
    page += '<h2>Settings and observed validation outcomes</h2><p>One row per run. An ineligible row shows its minimum zero-shot validation-NLL epoch; it is not a selected model. All other defaults: AdamW, weight decay 0.01, warmup 3%, cosine horizon 12 epochs, final learning-rate ratio 0.1, float32, full-model fine-tuning, query-answer-only loss. Full effective configurations and all epoch records are preserved in the scalar evidence.</p>' + table(rows)
    page += '<h2>Final full-test comparisons</h2>' + table(comparisons)
    page += '<p><a href="settings.csv">Settings CSV</a> · <a href="final-comparison.csv">Final comparison CSV</a> · <a href="study.png">PNG figure</a> · <a href="evidence.json">Scalar evidence</a> · <a href="../matched-audit/">Original checkpoint audit</a></p></body></html>'
    (output / 'index.html').write_text(page)
    (output / 'evidence.json').write_text(json.dumps({'snapshot_source': str(arguments.snapshot.resolve()),
        'snapshot_sha256': hashlib.sha256(arguments.snapshot.read_bytes()).hexdigest(), 'status': status,
        'reference': reference, 'settings': rows, 'final_comparisons': comparisons,
        'selection': search.get('locked-selection'), 'snapshot': snapshot}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
