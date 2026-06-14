# Reports

Report-related artifacts are consolidated here:

- `generated/`: generated report markdown and JSON outputs.
- `figures/`: charts referenced by report markdown.
- `figures/archive/`: preserved older duplicate chart outputs.

Run the main report through the package CLI:

```bash
python -m aoe4_predict report
```

Ad hoc runners can be executed from the repository root, for example:

```bash
python scripts/reports/run_report_only.py
```
