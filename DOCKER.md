# Docker CLI Usage

RaV-IDP is packaged as a local Dockerized CLI. It is not an HTTP service.

## First Run

```bash
cd /home/pritesh-jha/projects/rav-idp/experiment
cp .env.example .env
mkdir -p input artifacts data
docker compose build
```

Add `OPENAI_API_KEY` to `.env` if you want GPT fallback and image enrichment.

## Run On A PDF

Place the PDF under `input/`, then run:

```bash
docker compose run --rm rav-idp \
  python -m rav_idp.cli /workspace/input/my-document.pdf
```

Outputs are written under `artifacts/` on the host.

## Interactive Shell

```bash
docker compose run --rm rav-idp bash
```

From inside the container:

```bash
python -m rav_idp.cli /workspace/input/my-document.pdf
```

## Useful Cleanup

```bash
docker compose down
docker compose build --no-cache
```
