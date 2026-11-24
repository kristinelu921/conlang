"""Render the completed GSM8K study's saved settings and metrics without model computation."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


NOTES = {
    "trial-01": "Lowest learning rate: weakest peak accuracy in this search.",
    "trial-02": "Lower loss than 1e-5, with modest accuracy.",
    "trial-03": "Near the lowest loss, but lower accuracy than 1e-4.",
    "trial-04": "Lowest validation NLL, at epoch 3; accuracy peaks at epoch 4.",
    "trial-05": "More accuracy than 1e-4; NLL rises sharply by its accuracy peak.",
    "trial-06": "Higher accuracy at batch 32, at the cost of worse loss.",
    "trial-07": "Batch 8 trails batches 16 and 32 on peak accuracy and minimum loss.",
    "trial-08": "Highest validation accuracy; selected checkpoint has high NLL.",
    "trial-09": "Batch 64 lowers NLL versus batch 16, but has lower peak accuracy.",
    "trial-10": "95 versus 100 correct for the winner; selected NLL still high (1.3004).",
    "trial-11": "Doubling LR to 2e-3 worsens both peak accuracy and minimum loss.",
    "trial-12": "99 versus 100 correct: removing weight decay gives no clear gain.",
    "trial-13": "Stronger decay lowers NLL at the selected accuracy checkpoint, but loses accuracy.",
    "trial-14": "Final-answer weight 2 reduces accuracy; little change in minimum NLL.",
    "trial-15": "Final-answer weight 4 still trails the default's accuracy.",
    "trial-16": "Longer warmup lowers peak accuracy; minimum NLL changes little.",
    "trial-17": "Lower terminal LR does not beat the default accuracy.",
    "trial-18": "20-epoch schedule stops at epoch 10; it does not beat the 12-epoch schedule.",
    "loss-probe-01": "Smoothing raises peak accuracy versus trial 04, but worsens loss; 6-epoch cap.",
}


def changed_setting(configuration: dict[str, Any]) -> str:
    """Describe only confirmed changes from the common optimizer defaults."""
    differences = []
    if configuration["weight_decay"] != 0.01:
        differences.append(f"weight decay {configuration['weight_decay']:g}")
    if configuration["final_answer_weight"] != 1.0:
        differences.append(f"answer weight {configuration['final_answer_weight']:g}×")
    if configuration["warmup_fraction"] != 0.03:
        differences.append(f"warmup {100 * configuration['warmup_fraction']:g}%")
    if configuration["final_learning_rate_ratio"] != 0.1:
        differences.append(f"terminal LR {100 * configuration['final_learning_rate_ratio']:g}%")
    if configuration["schedule_epochs"] != 12:
        differences.append(f"schedule {configuration['schedule_epochs']} epochs")
    if configuration.get("label_smoothing", 0.0) != 0.0:
        differences.append(f"smoothing {configuration['label_smoothing']:g}; cap {configuration['epochs']} epochs")
    return "; ".join(differences) if differences else "defaults"


def main() -> None:
    """Create immutable tables and labeled plots from the saved scalar summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = json.loads(arguments.summary.read_text())
    assert isinstance(summary, dict)
    trials = summary["trials"]
    if len(trials) != 19 or {trial["trial_id"] for trial in trials} != set(NOTES):
        raise ValueError("Expected the completed study's exact 19 candidate identities.")
    rows = []
    for trial in trials:
        configuration = trial["configuration"]
        metrics = trial["best_metrics"]
        if metrics["validation/examples"] != 747 or configuration["seed"] != 1234:
            raise ValueError("The comparison requires the same validation partition and training seed.")
        rows.append({"trial": trial["trial_id"], "learning_rate": configuration["learning_rate"],
            "batch_size": configuration["batch_size"], "changed_setting": changed_setting(configuration),
            "validation_correct": metrics["validation/strict_correct"], "validation_examples": 747,
            "validation_accuracy_percent": 100 * metrics["validation/strict_correct"] / 747,
            "accuracy_epoch": trial["best_epoch"], "nll_at_accuracy_epoch": metrics["validation/loss"],
            "minimum_validation_nll": trial["best_loss"], "minimum_nll_epoch": trial["best_loss_epoch"],
            "completed_epochs": trial["completed_epochs"], "interpretation": NOTES[trial["trial_id"]],
            "configuration": configuration})
    arguments.output.mkdir(parents=True, exist_ok=False)
    source_digest = hashlib.sha256(arguments.summary.read_bytes()).hexdigest()
    data = {"source_summary_sha256": source_digest, "validation_examples": 747,
            "loss_units": "nats per query-answer token, including rationale and EOS", "trials": rows,
            "warning": "Peak accuracy and minimum loss may occur at different epochs. Test results did not select trials."}
    (arguments.output / "settings.json").write_text(json.dumps(data, indent=2) + "\n")
    csv_fields = [key for key in rows[0] if key != "configuration"]
    with (arguments.output / "settings.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in rows)

    figure, axis = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    scatter = axis.scatter([row["nll_at_accuracy_epoch"] for row in rows],
        [row["validation_accuracy_percent"] for row in rows],
        c=[row["learning_rate"] for row in rows], cmap="viridis",
        norm=LogNorm(1e-5, 2e-3), s=100, edgecolors="#263248", linewidths=0.7)
    offsets = {"trial-02": (-15, -15), "trial-03": (8, -4), "trial-06": (-25, -15),
               "trial-08": (-5, 11), "trial-12": (10, -4), "trial-14": (-20, -17),
               "trial-16": (10, -1), "trial-17": (8, 5), "trial-11": (8, -9)}
    for row in rows:
        label = row["trial"].replace("trial-", "T").replace("loss-probe-01", "S05")
        axis.annotate(label, (row["nll_at_accuracy_epoch"], row["validation_accuracy_percent"]),
                      xytext=offsets.get(row["trial"], (8, 6)), textcoords="offset points", fontsize=9)
    axis.set(title="SmolLM2-135M · GSM8K validation: accuracy versus loss",
             xlabel="NLL at the accuracy-selected checkpoint (nats / answer token; lower is better)",
             ylabel="Zero-shot strict accuracy on 747 validation questions (%; higher is better)",
             xlim=(0.55, 1.72), ylim=(4.5, 14.2))
    axis.grid(alpha=0.20)
    axis.text(0.02, 0.97, "Each dot uses accuracy and NLL from the SAME checkpoint.\n"
              "T01–T18 = main trials; S05 = smoothing 0.05.\n"
              "Upper-left is preferable. All trials use seed 1234.",
              transform=axis.transAxes, va="top", fontsize=10,
              bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9})
    figure.colorbar(scatter, ax=axis, label="Peak learning rate (logarithmic color scale)")
    figure.savefig(arguments.output / "accuracy-loss.svg")
    figure.savefig(arguments.output / "accuracy-loss.png", dpi=170)
    plt.close(figure)

    markdown = ["| Trial | LR | Batch | Other changes | Best validation accuracy | Lowest validation NLL | Observation |",
                "|---|---:|---:|---|---:|---:|---|"]
    html_rows = []
    for row in rows:
        values = [row["trial"], f"{row['learning_rate']:g}", str(row["batch_size"]), row["changed_setting"],
                  f"{row['validation_accuracy_percent']:.2f}% (epoch {row['accuracy_epoch']})",
                  f"{row['minimum_validation_nll']:.4f} (epoch {row['minimum_nll_epoch']})", row["interpretation"]]
        markdown.append("| " + " | ".join(values) + " |")
        display = values[:5] + [f"{row['nll_at_accuracy_epoch']:.4f}"] + values[5:]
        html_rows.append("<tr>" + "".join("<td>" + escape(value) + "</td>" for value in display) + "</tr>")
    (arguments.output / "table.md").write_text("\n".join(markdown) + "\n")
    style = "body{font:16px system-ui;color:#172033;max-width:1500px;margin:36px auto;padding:0 20px}img{width:100%;max-width:1200px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dce1e9;text-align:left;vertical-align:top}thead{background:#edf2f8}th{cursor:pointer;white-space:nowrap}tbody tr:hover{background:#f5f8fc}.table{overflow-x:auto}.note{background:#fff5d9;padding:14px;border-radius:7px}"
    columns = ["Trial", "Peak LR", "Batch", "Other changes", "Best validation accuracy", "NLL at accuracy peak", "Lowest validation NLL", "Observation"]
    page = "<!doctype html><html lang='en'><meta charset='utf-8'><title>SmolLM2 GSM8K settings comparison</title><style>" + style + "</style><h1>What worked in the 19-trial GSM8K search</h1>"
    page += "<p>SmolLM2-135M; full-parameter fine-tuning; 6,726 optimization and 747 validation examples. Same pinned revisions and split; one seed (1234).</p>"
    page += "<p>Defaults: AdamW, weight decay 0.01, warmup 3%, cosine schedule over 12 epochs ending at 10% of peak LR, answer weight 1, smoothing 0, sequence length 640, float32. Main trials stop after three epochs without accuracy or NLL improvement, after at least three epochs. The smoothing probe uses a six-epoch cap and patience two.</p>"
    page += "<p class='note'><strong>Read the epochs:</strong> best accuracy and lowest NLL often come from different checkpoints. The plot pairs metrics at the same accuracy-selected checkpoint. These are validation observations, not per-trial test scores or statistically established causal effects.</p>"
    page += "<img src='accuracy-loss.svg' alt='Validation zero-shot accuracy versus same-checkpoint answer-token NLL for all 19 trials, colored by learning rate'>"
    page += "<p>Click a column heading to sort. <a href='settings.csv'>CSV</a> · <a href='settings.json'>Exact metrics and full configurations</a> · <a href='../'>Original study plots and test results</a></p>"
    page += "<div class='table'><table><thead><tr>" + "".join("<th>" + column + "</th>" for column in columns) + "</tr></thead><tbody>" + "".join(html_rows) + "</tbody></table></div>"
    page += "<p>Source summary SHA-256: <code>" + source_digest + "</code>. Display rounding only; JSON retains recorded precision. No new experiment was run to make this page.</p>"
    page += "<script>document.querySelectorAll('th').forEach((header,index)=>{let ascending=true;header.onclick=()=>{const body=document.querySelector('tbody');const rows=Array.from(body.rows);rows.sort((a,b)=>{const x=a.cells[index].textContent,y=b.cells[index].textContent;const numeric=[1,2,4,5,6].includes(index);return (numeric?parseFloat(x)-parseFloat(y):x.localeCompare(y,undefined,{numeric:true}))*(ascending?1:-1)});rows.forEach(row=>body.appendChild(row));ascending=!ascending}})</script></html>"
    (arguments.output / "index.html").write_text(page)
    manifest = [{"name": path.name, "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(arguments.output.iterdir()) if path.is_file()]
    (arguments.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(arguments.output), "source_sha256": source_digest}))


if __name__ == "__main__":
    main()
