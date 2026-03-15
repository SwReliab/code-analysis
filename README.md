# code-analysis

code-analysis is a local analysis pipeline that unifies module aggregation across embed, commits, lizard, and faultdata steps.

## Features

- Single configuration file: analysis.yaml
- Unified module key logic via src/core/grouping.py
- Step execution: embed, commits, lizard, faultdata
- Full pipeline execution with run
- Pre-execution validation with --check

## Installation

```bash
pip install -e .
```

After installation, the CLI command is:

```bash
code-analysis
```

## Configuration

Use analysis.yaml at the repository root.

Example:

```yaml
module:
  root_src: src
  git_root: .
  git_subdir: src
  depth: 1

rle:
  start: 2025-W01
  until: 2025-W52

embedding:
  model: text-embedding-3-small
  weight_mode: loc

lizard:
  csv: lizard.csv
  languages: []
  extra_args: []

outputs:
  embeddings_jsonl: group_embeddings.jsonl
  commits_by_module_week_csv: commits_by_module_week.csv
  module_metrics_csv: module_metrics.csv
  faultdata_dir: faultdata
```

## Usage

Dry-run a step:

```bash
code-analysis embed -c analysis.yaml --dry-run
```

Validate before execution:

```bash
code-analysis run -c analysis.yaml --check
```

Execute a single step:

```bash
code-analysis commits -c analysis.yaml --execute
```

Execute embed in test mode (no OpenAI API key required):

```bash
code-analysis embed -c analysis.yaml --execute --test --test-dim 256
```

Execute commits in test mode (no git history required):

```bash
code-analysis commits -c analysis.yaml --execute --test --test-seed 42 --test-max-count 5
```

Execute all steps:

```bash
code-analysis run -c analysis.yaml --execute
```

## Output Files (Examples)

embed:

- group_embeddings.jsonl

Example (first record, shortened):

```json
{"folder":"__init__.py","group_type":"file","n_files":1,"total_loc":1,"weight_mode":"loc","embedding":[0.0894,...],"files":["__init__.py"]}
```

commits:

- commits_by_module_week.csv

Example:

```csv
bucket,module,count
2025-W01,cli.py,1
2025-W01,config.py,1
2025-W02,core,1
```

lizard:

- lizard.csv
- module_metrics.csv

Example:

```csv
group_key,group_type,files,functions,sum_nloc,sum_ccn,sum_token,avg_nloc,avg_ccn,avg_token,max_nloc,max_ccn,max_token,avg_params,max_params
core,folder,6,51,628,211,5254,12.314,4.137,103.020,60,18,474,14.196,72
plan.py,file,1,7,117,42,983,16.714,6.000,140.429,43,24,407,18.857,52
```

faultdata:

- faultdata/folder_hash_map.json
- faultdata/folder_hash_map.jsonl
- faultdata/94a0426e8d3203da5468ccf0c624f93cb37601e2.csv

Example hash mapping:

```json
{"core":"94a0426e8d3203da5468ccf0c624f93cb37601e2"}
```

Example RLE CSV (core):

```csv
week,counts
1,0
1,1
1,0
1,4
```

## Commands

- embed
- commits
- lizard
- faultdata
- run

All commands support:

- -c / --config
- --dry-run
- --check
- --execute
- --format yaml|json
