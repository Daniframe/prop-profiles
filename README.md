# propensities-incite

Reproduction pipeline for **"Capabilities Ain't All You Need: Measuring
Propensities in AI"** (arXiv 2602.18182): a framework for inferring propensity
profiles of AI systems from instance-level success/failure results and
LLM-annotated demand-interval rubrics.

The pipeline has 5 sectors, each a subpackage of `src/propensity/`:

| # | Sector | Package | Docs | Status |
|---|---|---|---|---|
| 1 | Dataset generation | `src/propensity/generation/` | [docs/sector1_generation.md](docs/sector1_generation.md) | Implemented |
| 2 | Annotation (AzureOpenAI rubric-based demand intervals) | `src/propensity/annotation/` | -- | Not yet implemented |
| 3 | Model inference (instance-level outcomes across incitement levels) | `src/propensity/inference/` | -- | Not yet implemented |
| 4 | Propensity surfaces (curve/surface fitting) | `src/propensity/surfaces/` | [docs/sector4_surfaces.md](docs/sector4_surfaces.md) | Implemented |
| 5 | Predictability (does propensity improve outcome prediction?) | `src/propensity/predictability/` | -- | Not yet implemented |

## Setup

```bash
conda env create -f environment.yaml   # or: pip install -r requirements.txt
```

Sectors that call Azure OpenAI (annotation, part of generation, part of
inference) read `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT` from the
environment (via `.env`, `python-dotenv`) -- see each sector's docs for which
functionality needs them. Sector 4 needs no credentials at all.

## Repository layout

```
src/propensity/{common,generation,annotation,inference,surfaces,predictability}/
scripts/            # one CLI per pipeline stage
config/             # one YAML per sector
benchmarks/         # Sector 1 declarative benchmark specs
rubrics/            # canonical propensity rubrics (RA, BR, Ex, Ul)
prompts/incitement/ # canonical incitement system prompts (RA, BR, Ex, Ul)
data/               # small/final artifacts checked in; data/inference/raw/ gitignored (regenerable dumps)
tests/              # pytest, one folder per sector once implemented
docs/               # per-sector documentation + usage examples
```

## Testing

```bash
pytest tests/ -q
```

Every test is synthetic-fixture-based -- no network calls, no GPU, no
credentials required to run the suite (functionality that needs Azure OpenAI
is exercised against a mocked client instead).
