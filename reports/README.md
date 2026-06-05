# Reports

Report-related artifacts are consolidated here:

- `*.md` and `*.json`: generated report outputs.
- `figures/`: charts referenced by report markdown.
- `figures/archive/`: preserved older duplicate chart outputs.
- `scripts/`: ad hoc report and experiment runners.

Run the main report through the package CLI:

```bash
python -m aoe4_predict report
```

Ad hoc runners can be executed from the repository root, for example:

```bash
python reports/scripts/run_report_only.py
```
