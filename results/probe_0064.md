# 0064 drift/convergence probe

Setup: `samples/0064.png`, `recursive_multiscale`, `--save_prompts`, `.venv`, GPU 0. A3 used author LoRA `ckpt/VLM_LoRA/checkpoint-10000`; w4v2 used full-FT `ckpt/VLM_FT/coz_w4v2` through `--vlm_model_path`.

## Per-scale prompts

| scale | A3 author prompt | w4v2 tuned prompt |
|---:|---|---|
| 1 | animal, red panda, face, close-up, eyes, nose, whiskers, fur, leaves, nature, wildlife, cute, adorable, fluffy, brown | Red panda, animal, mammal, fur, face, eyes, nose, whiskers, leaf, close-up, zoom-in, detail, texture, nature |
| 2 | dog, close-up, eye, nose, whiskers, fur, texture, detail, macro, animal, pet, eyes, nose, whiskers, fur | dog |
| 3 | fur, texture, close-up, animal, soft, fluffy, white, brown, detail, macro, natural, fur texture, animal fur, close-up shot | Fur texture, Close-up, Animal fur, Soft, Fluffy, Detail, Macro shot, Natural, Warm tones, Cozy, Softness, Texture |
| 4 | hair texture, close-up, detailed, strands, natural, soft, light brown, white, pattern, background, close-up view, detailed texture, natural hair | Hair texture, Fur texture, Animal fur, Furry background, Natural pattern, Textured backdrop, Close-up view, Soft lines, Coarse strands, F |

## Checks

- Subject-token retention at deep scales (`eye|fur|animal|dog`): **PASS**. w4v2 scale 3 contains `Fur`/`Animal`; w4v2 scale 4 contains `Fur`/`Animal`. A3 scale 4 loses this subject-token set.
- Neural drift terms (`neuron|synapse|dendrite|axon`, singular/plural): **PASS for no neural terms, tie vs A3**. w4v2 count = 0; A3 rerun count = 0. Historical fail-case drift had `Neurons, dendrites, axons, synapses...`, but neither rerun emitted those terms.
- Cross-scale unique-token ratio: **PASS**. A3 = 34 unique / 72 tokens = 0.4722; w4v2 = 38 unique / 55 tokens = 0.6909; delta = +0.2187.

Verdict: w4v2 retains animal/fur subject tokens through scales 3-4, emits no neuron/synapse/dendrite/axon drift terms, and has a higher cross-scale unique-token ratio than A3. Caveat: w4v2 scale 2 is terse (`dog`), and the neural-token check is a no-drift tie against this A3 rerun rather than a differential win.
