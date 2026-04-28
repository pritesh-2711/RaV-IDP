## Reconstruction as Validation - Intelligent Document Processor

RaV-IDP is a local CLI pipeline for document extraction. It detects document
regions, extracts tables/images/text, reconstructs each extraction, scores
fidelity against the original crop, and can use GPT fallback for low-confidence
regions.

### Quick Start With Docker

```bash
cp .env.example .env
mkdir -p input artifacts data
docker compose build rav-idp
```

Place a PDF in `input/`, then run:

```bash
docker compose run --rm rav-idp \
  python -m rav_idp.cli /workspace/input/my-document.pdf
```

Outputs are written to `artifacts/`. Add `OPENAI_API_KEY` to `.env` if you want
GPT fallback and image enrichment. See `DOCKER.md` for more Docker usage notes.

### Paper

- arXiv: https://arxiv.org/abs/2604.23644
- HF: https://huggingface.co/papers/2604.23644
- DOI: https://doi.org/10.5281/zenodo.19694316

## License

See LICENSE for details.
