# Additive-ablation InternVL prompt-grounding judge summary

Eval set: `data/eval_subset` (30 images, ids 0801-0830). Judge: InternVL3-8B via `evaluate.py --vlm-judge`. Primary metric is `vlm_grounding_deep` (deepest two prompt/image pairs). CSV scoring writes rows incrementally.

## Core comparison (mean over n=30; deltas vs BASE)

| arm | deep ↑ | Δdeep | grounding_all ↑ | Δall | unique_token_ratio ↑ | Δuniq | MUSIQ ↑ | NIQE ↓ | CLIPIQA ↑ | vlm_quality ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE | 7.917 | +0.000 | 8.067 | +0.000 | 0.609 | +0.000 | 50.93 | -7.61 | 0.616 | 5.60 |
| +R_anc | 7.700 | -0.217 | 7.958 | -0.108 | 0.629 | +0.020 | 50.89 | -7.55 | 0.599 | 5.47 |
| +R_rep | 7.900 | -0.017 | 7.925 | -0.142 | 0.594 | -0.015 | 52.02 | -7.42 | 0.623 | 5.43 |
| ALL | 8.183 | +0.267 | 8.192 | +0.125 | 0.611 | +0.002 | 51.85 | -7.38 | 0.621 | 5.70 |

## Optional reference rows (same n=30)

| arm | deep ↑ | Δdeep vs BASE | grounding_all ↑ | unique_token_ratio ↑ | MUSIQ ↑ | NIQE ↓ | CLIPIQA ↑ | vlm_quality ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A2-ref | 6.567 | -1.350 | 6.792 | 0.698 | 51.01 | -7.79 | 0.597 | 5.17 |
| w4v2-ref | 7.283 | -0.633 | 7.525 | 0.672 | 51.03 | -7.78 | 0.605 | 5.60 |

## Verdict

- **Grounding is the discriminative axis.** `ALL` is the only core arm with a clear aggregate win: `vlm_grounding_deep` = 8.183 (+0.267 vs BASE) and `vlm_grounding_all` = 8.192 (+0.125).
- **Single-arm components are mixed.** `+R_rep` is the better individual grounding arm on drift-recovery examples (e.g. 0812/0827/0817), but its mean deep grounding is essentially tied/slightly below BASE (-0.017). `+R_anc` is lower on mean deep grounding (-0.217); this matches the training caveat that `R_anc` had near-zero reward variance / saturation.
- **Diversity/convergence:** `+R_anc` has the highest unique-token ratio among the three new ablations (0.629, +0.020 vs BASE), while `ALL` is basically tied (+0.002) and `+R_rep` is lower (-0.015). So the simple unique-token-ratio proxy does **not** show an isolated `+R_rep` diversity win on this n=30 screen.
- **IQA is approximately tied.** MUSIQ/NIQE/CLIPIQA move only modestly; `+R_rep` and `ALL` have slightly higher MUSIQ/CLIPIQA, while BASE has slightly better NIQE in this run. VLM quality is also close, with `ALL` highest by +0.10.
- `+R_fb` was dropped for time and is not included in this additive-ablation table.

## Per-image grounding deltas used for trajectory examples

| image | BASE deep | +R_anc | +R_rep | ALL | note |
|---|---:|---:|---:|---:|---|
| 0812 | 5.5 | 5.0 | 9.5 | 9.0 | statue crop: baseline/anchor drift to medical mammogram terms; +R_rep/ALL score much higher on deep grounding |
| 0827 | 5.0 | 5.0 | 8.5 | 8.5 | highway crop: +R_rep and ALL keep road/car vocabulary and recover +3.5 to +4.0 deep points |
| 0824 | 8.0 | 1.0 | 4.5 | 5.5 | ship wheel crop: persistent drift/failure case; all variants wander through propeller/camera/ring/space |

## Prompt trajectory examples (txt/*.txt)

### Image 0812
- **BASE**
  - s0: statue, sculpture, statue of woman, classical architecture, grand building, outdoor scene, clear sky, greenery, pedestal, detailed features, historical monument,
  - s1: statue, black, close-up, details, legs, torso, statue, figure, sculpture, historical, art, architecture, building, outdoors, outdoor,
  - s2: Breast, mammogram, mammography, breast cancer, dense tissue, texture, density, nodules, abnormality, imaging, medical, scan, X
  - s3: Breast, Mammogram, X-ray, Medical, Diagnostic, Imaging, Scan, Radiology, Healthcare, Female, Anatomy, Tissue, Texture,
- **+R_anc**
  - s0: statue, sculpture, classical architecture, grand building, outdoor, daytime, clear sky, greenery, pedestal, detailed, intricate, historical, cultural, monument
  - s1: statue, black, ancient, historical, sculpture, detailed, craftsmanship, art, monument, outdoor, architecture, background, close-up, details, texture,
  - s2: Breast, mammogram, breast cancer, dense tissue, nodules, abnormality, medical imaging, X-ray, scan, mammography, breast health,
  - s3: Breast, mammogram, tissue, density, calcifications, nodules, abnormality, imaging, medical, scan, X-ray, radiology, diagnostic
- **+R_rep**
  - s0: statue, sculpture, statue of woman, classical architecture, grand building, outdoor scene, blue sky, white clouds, greenery, pedestal, detailed features,
  - s1: statue, black, close-up, details, legs, torso, statue, public, building, architecture, historical, sculpture, art, craftsmanship, texture,
  - s2: Black, Close-up, Breast, Female, Skin, Texture, Dark, Soft, Smooth, Softness, Detail, Closeup, Close-up, Details,
  - s3: Dark, texture, close-up, surface, details, pattern, dark blue, black, grainy, rough, uneven, intricate, close-up view, detailed
- **ALL**
  - s0: statue, sculpture, classical architecture, grand building, outdoor, daytime, clear sky, greenery, pedestal, detailed, intricate, historical, cultural, monument
  - s1: statue, black, ancient, historical, sculpture, detailed, craftsmanship, art, monument, outdoor, architecture, background, close-up, details, texture,
  - s2: Breast, mammogram, breast cancer, tissue, texture, scan, medical, imaging, X-ray, mammography, mammogram, breast tissue, breast
  - s3: Breast, mammogram, scan, tissue, density, texture, imaging, medical, diagnostic, X-ray, mammography, breast cancer, abnormality,

### Image 0827
- **BASE**
  - s0: highway, cars, traffic, aerial view, urban landscape, road, vehicles, transportation, infrastructure, evening, dusk, lights, movement, flow, density
  - s1: highway, aerial view, vehicles, cars, trucks, lanes, road, traffic, transportation, movement, infrastructure, urban, highway, transportation, infrastructure,
  - s2: car, road, aerial view, highway, blue, vehicle, transportation, perspective, top-down, straight, smooth, empty, wide, lanes, asphalt,
  - s3: car, road, blue, top view, asphalt, street, vehicle, aerial, perspective, urban, transportation, modern, sleek, shiny, clean, parked
- **+R_anc**
  - s0: highway, traffic, cars, trucks, lights, dusk, evening, aerial view, urban, infrastructure, transportation, movement, road, vehicles, landscape,
  - s1: highway, aerial view, vehicles, cars, trucks, lanes, road, traffic, transportation, urban, infrastructure, movement, speed, direction, perspective,
  - s2: car, road, aerial view, blue, highway, vehicle, transportation, perspective, top-down, asphalt, lane, straight, modern, urban, movement,
  - s3: car, road, blue, aerial view, top-down, street, asphalt, vehicle, parked, close-up, detail, texture, lines, shadows, reflections
- **+R_rep**
  - s0: highway, cars, traffic, dusk, evening, aerial view, urban, road, vehicles, lights, movement, landscape, infrastructure, transportation, commute,
  - s1: highway, cars, aerial view, traffic, road, vehicles, transportation, movement, urban, infrastructure, highway, lanes, vehicles, cars, traffic,
  - s2: car, road, aerial view, blue, highway, vehicle, transportation, asphalt, top-down, perspective, urban, movement, speed, travel, journey,
  - s3: car, road, asphalt, blue, top view, aerial, street, vehicle, parked, close-up, detail, texture, lines, shadow, reflection,
- **ALL**
  - s0: highway, cars, traffic, dusk, aerial view, urban landscape, road, vehicles, evening, lights, motion, movement, infrastructure, transportation, city
  - s1: highway, cars, traffic, aerial view, road, vehicles, transportation, movement, urban, infrastructure, highway, lanes, speed, direction, perspective,
  - s2: car, road, aerial view, highway, blue, vehicle, transportation, asphalt, straight, perspective, top-down, urban, movement, speed, journey,
  - s3: car, road, asphalt, blue, top view, street, vehicle, detail, texture, urban, close-up, perspective, angle, view, overhead,

### Image 0824
- **BASE**
  - s0: ship wheel, steering wheel, nautical, vintage, wooden, metal, ropes, ropes, deck, railing, sunlight, shadows, reflection, blue sky,
  - s1: propeller, close-up, metal, rust, vintage, industrial, detail, texture, weathered, old-fashioned, mechanical, aviation, boat, ship,
  - s2: camera, lens, close-up, detail, vintage, metal, glass, black, round, texture, reflection, vintage-camera, antique, photography, lens-cap
  - s3: space, planet, moon, ring, space exploration, astronomy, celestial body, outer space, cosmic, galaxy, universe, distant, distant view, close-up
- **+R_anc**
  - s0: ship, steering wheel, deck, railing, sunlight, shadow, metal, rust, nautical, vintage, control, navigation, outdoors, sky, clouds,
  - s1: propeller, close-up, metal, rusted, vintage, industrial, detail, texture, weathered, old-fashioned, mechanical, engineering, craftsmanship, antique
  - s2: camera, lens, close-up, detail, metal, black, vintage, antique, photography, lens cap, glass, focus, aperture, shutter, lens mount
  - s3: ring, close-up, texture, detail, black background, metallic, circular, high-resolution, zoomed-in, macro, close-up view, detailed, intricate
- **+R_rep**
  - s0: ship wheel, steering wheel, ship, boat, nautical, vintage, old, rusty, metal, wooden, ropes, handles, spokes, center, black
  - s1: propeller, close-up, metal, rusted, vintage, industrial, detail, texture, weathered, old-fashioned, mechanical, engineering, craftsmanship, antique
  - s2: camera, lens, macro, close-up, detail, metal, black, texture, reflection, vintage, old-fashioned, antique, photography, lens cap, glass
  - s3: ring, space, planet, Saturn, view, close-up, detail, atmosphere, surface, rings, planet, Saturn, view, close-up, detail,
- **ALL**
  - s0: ship wheel, steering wheel, nautical, vintage, metal, wooden, ropes, ropes, deck, railing, sunlight, shadows, blue sky, clouds,
  - s1: propeller, close-up, metal, rusted, vintage, industrial, detail, texture, weathered, old-fashioned, mechanical, craftsmanship, engineering, antique
  - s2: camera, lens, close-up, detail, metal, texture, vintage, photography, equipment, macro, focus, lens cap, shutter, aperture, lens mount
  - s3: ring, close-up, texture, detail, metallic, curvature, edge, dark background, macro, zoom-in, close-up view, detailed, metallic surface,

## Output paths

- `results/abl_base.csv`
- `results/abl_anc.csv`
- `results/abl_rep.csv`
- `results/abl_all.csv`
- `results/abl_A2.csv`
- `results/abl_w4v2.csv`
- `results/abl_grounding_table.csv`
- `results/abl_summary.md`
- `results/abl_anc_sr/per-sample`
- `results/abl_rep_sr/per-sample`
- `results/abl_all_sr/per-sample`

## Per-image paired analysis

See `results/abl_paired_analysis.md` for the per-image PAIRED stats (win-rate + t-read) and the full A2->w4v2->BASE->+R_anc->+R_rep->ALL ladder.
Headline (n=30, paired vs BASE): ALL deep grounding +0.267, 13/30 wins, t-read +0.77 (directional, within noise); +R_anc wins diversity (uniqtok +0.020, t-read +1.80); confirm at n=100.
