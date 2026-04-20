"""Step 0: sanity check gpt-4.1 fallback on 10 failed table samples."""
import io, json, tarfile
from pathlib import Path
from PIL import Image
from rav_idp.components.fallback_extractor import call_vision_fallback
from rav_idp.components.comparators.table import compare_table
from rav_idp.components.reconstructors.table import reconstruct_table
from rav_idp.config import get_settings
from rav_idp.models import BoundingBox, DetectedRegion, EntityType

settings = get_settings()
print(f"Model: {settings.openai_model}")

artifact = json.loads(Path("artifacts/stage3a_tables_500_val_fixed.json").read_text())
failed = [r for r in artifact["records"] if not r.get("passed_threshold", True)][:10]
print(f"Using {len(failed)} failed samples")

archive = Path("data/raw/pubtabnet/pubtabnet.tar.gz")
filenames = {r["filename"] for r in failed}
img_map = {}
with tarfile.open(archive, "r:gz") as tf:
    for m in tf:
        for fn in list(filenames):
            if fn in m.name:
                img_map[fn] = tf.extractfile(m).read()
                filenames.discard(fn)
                break
    if not filenames:
        pass

results = []
for rec in failed:
    fn = rec["filename"]
    orig_fidelity = rec["fidelity_score"]
    img = img_map.get(fn)
    if not img:
        print(f"  SKIP {fn} — not found"); continue

    w, h = Image.open(io.BytesIO(img)).size
    region = DetectedRegion(
        region_id=rec["sample_id"], entity_type=EntityType.TABLE,
        bbox=BoundingBox(x0=0, y0=0, x1=w, y1=h, page=0),
        original_crop=img, processed_crop=img,
        raw_docling_record={}, page_index=0,
    )
    try:
        fallback = call_vision_fallback(region, context_text="")
        recon = reconstruct_table(fallback, region)
        fid = compare_table(recon.content, region, settings.threshold_table,
                            skip_visual=True, detected_col_count=rec.get("predicted_cols"))
        fb_fidelity = fid.fidelity_score
        recovered = fb_fidelity >= settings.threshold_table
        err = None
    except Exception as e:
        fb_fidelity = 0.0; recovered = False; err = str(e)[:80]

    results.append(dict(orig=orig_fidelity, fb=fb_fidelity, recovered=recovered, err=err))
    status = "RECOVERED" if recovered else ("ERROR: " + str(err) if err else "failed")
    print(f"  {rec['sample_id']}: orig={orig_fidelity:.3f} → fb={fb_fidelity:.3f} [{status}]")

n = len(results)
n_rec = sum(1 for r in results if r["recovered"])
print(f"\ngpt-4.1 on 10 samples: {n_rec}/{n} = {n_rec/n:.1%}")
print(f"Reference (gpt-4o, 194 samples): 74/194 = 38.1%")
delta = n_rec/n - 0.381
print(f"Delta: {delta:+.1%} — {'OK (< 5%)' if abs(delta) < 0.05 else 'LARGE — note in paper'}")
