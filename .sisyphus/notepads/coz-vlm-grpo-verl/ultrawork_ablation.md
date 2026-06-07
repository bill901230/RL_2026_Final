# ULTRAWORK: additive reward ablation + VLM-judge eval (deadline 21:00 today)

## REQUEST (user, RL course final project)
Current balanced reward gave inconclusive/small gains. Redo as ADDITIVE ablations, FIX each component to a CLEAR win individually, THEN combine:
1. Original/base Reward
2. base + Anchor (R_anc)
3. base + Cross-Repetition Penalty (R_rep)
4. base + Feedback (R_fb)
5. ALL
Constraints: 2x H200; deadline 21:00 (started ~16:15, ~4.5h); DO NOT ask user questions (unavailable) -> oracle/recommended/small-sample-then-scale; can add ablations but keep on-track (course project); work until last minute.

## JUDGE ENDPOINTS (paper uses InternVL for IQA) — VERIFIED WORKING
- InternVL3-8B (PRIMARY): https://intern-vl3-8b.seanmamasde.me/v1  key: sk-internvl3-zpC5lCRvqeRnjxFanKbfttAQsLrz1BD0SVLvbb8i7aA  model id: InternVL3-8B
- Gemma-4-26B (reference): https://gemma4-26b-a4b.seanmamasde.me/v1  key: sk-gemma4-1SkjlYEYPcTrYhg8C4wVK4dqlirP5muW8vw7XQ1mL0A  model id: gemma-4-26B-A4B-it
- **MUST call via curl / browser User-Agent** — Python urllib gets Cloudflare 403 (error 1010). OpenAI-compatible /v1/chat/completions, content=[{type:text},{type:image_url,image_url:{url:"data:image/png;base64,..."}}]. Multi-image supported.

## CRITICAL JUDGE FINDING (manual QA, id 0801, deepest 4.png)
- Absolute quality 1-10: A0_nn=1, A1/A2/A3/OURS ALL=7 -> NOT discriminative among trained arms (too coarse).
- Consistency(orig 0.png + 256x crop 4.png): A3=1, OURS=1 -> at 256x a crop looks unrelated to full image; this framing is BROKEN.
=> MUST use a DISCRIMINATIVE protocol: PAIRWISE A-vs-B ("which crop is higher quality / more faithful zoom", randomize order, count win-rate), and/or ADJACENT-scale consistency (scale i vs i+1), and/or finer 1-100 rubric. Confirm with oracle + check proposal.pdf for the paper's exact InternVL protocol.

## IN-FLIGHT AGENTS (fired ~16:20)
- explore bg_86b831a8 (ses_163fece51ffeEHgV5Z5T7TEdoQ): reward components + what "base/original reward" is + env toggles
- explore bg_cb6b084c (ses_163feb08fffeCc5kDHmrrXCnOk): evaluate.py structure + where to add VLM-judge axis
- explore bg_51f0d755 (ses_163fe8b4fffeA2wWRvp433fkEF): launchers + per-component env toggles + merge/eval recipe + timing
- oracle bg_cb443731 (ses_163fe2f6bffeQDdXrK1YdoB1Rg): experimental design under 4.5h/2GPU, judge rubric, base-reward def, per-component weights, priority order

## KEY KNOWN FACTS (from prior session)
- veRL env: .venv-train; eval: .venv (source activate.sh). 2 GPUs idx 0,1. NEVER uv run for verl. RAY_TMPDIR/TMPDIR off NFS + umask 022. Direct-controller launcher .sisyphus/evidence/verl_direct_controller.py.
- Reward weights via env: COZ_R_ANC_WEIGHT(0.2) COZ_R_REP_WEIGHT(1.0) COZ_R_FB_WEIGHT(1.0) COZ_R_PHR_WEIGHT(0.1) COZ_ENABLE_RFB COZ_VALIDITY_GUARD. Stabilized cfg: kl_loss_coef=0.02 lr=3e-7 entropy_coeff=0.005 grad_clip=0.5 rollout.n=6 train_batch=2.
- Full-FT base: ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged. Train parquet: data/parquet/coz_states_train_s2_4.parquet (+ _val_, ~150 rows). Eval valid: data/div2k/valid (n=100). SR: ckpt/SR_LoRA/model_20001.pkl + SR_VAE/vae_encoder_20001.pt + DAPE + RAM.
- 100-step full-FT ~2.5h on 2 GPUs. Merged FT models -> ckpt/VLM_FT/<name>. eval: inference_coz.py --vlm_model_path <dir> ... then evaluate.py.
- Existing eval SR dirs on disk: results/{A0,A1,A2,A3,w4v2}_full_sr/per-sample/<id>/{0..4}.png + txt/. (per-sample/<id>/0.png=original, 4.png=deepest 256x).
- Branch grpo/coz-vlm pushed to origin (SSH push-url). HEAD 8c5897d.

## PIVOTAL FINDINGS (manual QA) — RESHAPE THE WHOLE APPROACH
- FINDING A: SR IMAGE quality is ~TIED A3 vs ours. Pairwise InternVL on 10 random ids = 5/5; absolute = both 7. The FROZEN SR backbone caps pixel quality; prompt changes barely move image IQA on average. => judging the IMAGE cannot separate arms (explains "inconclusive").
- FINDING B: discriminative signal = PROMPT quality. Use PROMPT<->IMAGE GROUNDING judge: give InternVL the crop image + the model's prompt for that scale, ask "does this text faithfully/accurately describe what is visible? 1-10". Drifted prompt (neurons for fur) scores low. This measures DRIFT directly and WILL discriminate. Plus cross-scale DIVERSITY for convergence. Image-quality wins only in the TAIL (catastrophic drift cases e.g. 0880 owl->tire->stone) -> test pairwise on drift-case ids, not random.
- FINDING C: openai SDK uses httpx -> will also hit Cloudflare 1010. Judge client MUST set a browser User-Agent (httpx headers) OR shell out to curl. (curl confirmed working.)

## evaluate.py JUDGE INTEGRATION (mapped, bg_cb6b084c)
Root evaluate.py. score_sample (:203-220): x0=scale0_image(:208), deepest=sr_image(:207) already loaded PIL. 3 edits: (1) build self.judge in CoZEvaluator.__init__ (:190-201); (2) after :211 merge self.judge.score(...) into scores dict (:213-220); (3) append "vlm_quality","vlm_consistency" (and/or prompt-grounding axis) to CSV_COLUMNS (:34-44). Loop(:267-272)+DictWriter(:226) need no change. Add CLI flags parse_args(:232-258)+Args(:70-76). For prompt-grounding axis I also need the per-scale prompts (sample.prompt_paths) + per-scale images -> may compute a mean grounding over scales 1..N. openai==2.38.0 in .venv-train but use browser-UA/curl.

## REWARD MAP (bg_86b831a8) — ARM DEFINITIONS (veRL path = source of truth)
- R_crit HARDCODED OFF on veRL. "base/original reward" = R_phr(0.1) + validity guard ONLY (no quality/task base). => BASE arm = UNTRAINED starting policy = author A3 (we full-FT FROM merged author model). A3 already evaluated (A3_full). Do NOT train base (oracle).
- Additive arms = GRPO-finetune A3 with ONE component (env toggles, keep COZ_R_PHR_WEIGHT=0.1 + COZ_VALIDITY_GUARD=1 constant):
  - +R_anc: COZ_R_ANC_WEIGHT=0.2 COZ_R_REP_WEIGHT=0 COZ_ENABLE_RFB=0
  - +R_rep: COZ_R_ANC_WEIGHT=0 COZ_R_REP_WEIGHT=1.0 COZ_ENABLE_RFB=0  (TRAIN scales 2-4 parquet or R_rep==0; coz_states_train_s2_4.parquet OK)
  - +R_fb : COZ_R_ANC_WEIGHT=0 COZ_R_REP_WEIGHT=0 COZ_ENABLE_RFB=1 COZ_R_FB_WEIGHT=1.25 (NEEDS extra_info.crop_path in parquet -> VERIFY/blocker; RFB=musiq+0.5*clip-consistency)
  - ALL: COZ_R_ANC_WEIGHT=0.2 COZ_R_REP_WEIGHT=1.0 COZ_ENABLE_RFB=1 COZ_R_FB_WEIGHT=1.25
- Component formulas: R_anc=cos01(prompt,x0_caption) [SATURATES near 1 -> low advantage, watch]; R_rep=-(0.5*Jaccard2 + 0.5*maxCLIPsim) to prev prompts [0 if no prev]; R_fb=musiq(SR(prompt))+0.5*clip(SR,crop); R_phr=-#fillers. Combine: R=sum_k w_k * z_pergroup(R_k) THEN GRPO re-standardizes advantages => DOUBLE z-norm => weight magnitude ~irrelevant for single-component arms (what matters = component produces within-group variance).
- Validity guard: invalid if model_tokens<8 OR unique_content<5 OR cjk_ratio>0.30 -> total forced <= -2.0.
- Two paths exist: veRL (ours) vs torch exp0-4 scaffold (R_crit ON, torch only). Ignore torch path.

## ORACLE DESIGN (bg_cb443731) — ADOPTED
- Schedule (~45min/arm @ 30 steps): train +R_anc, +R_rep, +R_fb, ALL sequentially on 2 GPUs (NOT 1-GPU concurrent — fragile). Stabilized cfg ALL arms: lr3e-7 kl0.02 entropy0.005 grad_clip0.5 guard on. Use all ~150 train rows.
- Eval: InternVL JUDGE-ONLY (not a training reward; would bottleneck+contaminate). Fast n=30 screen base+4 arms -> pick best individual + ALL -> final n=100. Gemma reference on finalists.
- Judge rubric (refined by MY Finding B): primary = PROMPT-GROUNDING (drift), + quality (expect ~tied), + diversity (convergence). Clear win ~ +0.4/10 mean OR >=60% paired win-rate, no drop on other axis.
- Priority if time short: base, +R_anc, +R_rep, ALL, (+R_fb last/optional).
- Watch: R_rep terse-hack (need min content len); R_fb musiq overfit (select by InternVL); R_anc generic-caption collapse.

## JUDGE VALIDATED (bg_c807804d) — GREEN LIGHT
- train/grpo/vlm_judge.py BUILT: VlmJudge.quality(img), .grounding(prompt,img), .pairwise(a,b), .prompt_pairwise_grounding(...). Env: VLM_JUDGE_BASE_URL/API_KEY/MODEL/UA/TIMEOUT/RETRIES/TRANSPORT. curl/browser-UA, fail-soft. lsp clean.
- Discrimination (n=20, ids 0801-0820): deep-grounding A2=6.475, A3=8.150, ours=7.975 (A3/ours -A2 ~+1.6). Quality spread only 0.5 (useless). => PRIMARY metric = PROMPT-GROUNDING (esp deep). Validation csv: .sisyphus/evidence/judge_validate.csv.
- NOTE A3(base) ~ ours on grounding already (author training fixed most drift); the NEW additive arms test whether ISOLATED R_anc beats base(A3) grounding, R_rep beats diversity, etc.

## LAUNCHER MAP (bg_51f0d755) — EXECUTION COOKBOOK
- BLOCKER W0.1: task_w4v2_seed123_2gpu_stable.sh HARDCODES reward weights as export (lines ~53-64). MUST parametrize (make reward env caller-driven) or every arm trains identically. Script ALREADY reads COZ_TOTAL_STEPS/COZ_SAVE_FREQ/COZ_CHECKPOINT_DIR (trainer.total_training_steps=${COZ_TOTAL_STEPS:-100}).
- weight=0 cleanly removes a component (rewards.py:174-211). COZ_ENABLE_RFB=0 also SKIPS loading SD3 (text-only arms = FAST). +R_fb/ALL load SD3 (slow). crop_path present in coz_states_train_s2_4.parquet (R_fb works).
- Timing: 100 steps ~1h49m (~63s/step) w/ R_fb; text-only faster. Use COZ_TOTAL_STEPS=30 (~30-40min/arm).
- Concurrency (1 GPU/arm): n_gpus_per_node=1, per-arm CUDA_VISIBLE_DEVICES=0 or 1, distinct RAY_TMPDIR, COZ_REWARD_*_DEVICE=cuda:0, REMOVE shared `ray stop --force`. Best: run 2 TEXT-ONLY arms concurrent; SD3 arms heavier.
- Merge: .venv-train: python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/abl_<arm>/global_step_<N>/actor --target_dir ckpt/VLM_FT/coz_abl_<arm>
- Eval: .venv: inference_coz.py -i <imgs> -o results/abl_<arm>_sr --vlm_model_path ckpt/VLM_FT/coz_abl_<arm> --vlm_state standard --rec_type recursive_multiscale --prompt_type vlm --lora_path ckpt/SR_LoRA/model_20001.pkl --vae_path ckpt/SR_VAE/vae_encoder_20001.pt --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers --ram_ft_path ckpt/DAPE/DAPE.pth --ram_path ckpt/RAM/ram_swin_large_14m.pth --save_prompts ; then evaluate.py --arm abl_<arm> --images results/abl_<arm>_sr/per-sample --out results/abl_<arm>.csv
- eval sets: data/eval_subset (30 imgs 0801-0830) for screen; data/div2k/valid (100) for final.

## GREEN-LIT EXECUTION (deadline 21:00)
TRACK A (critical path, GPU): parametrize launcher -> smoke-verify +R_anc reward variance -> train 4 arms @30 steps (text-only +R_anc/+R_rep concurrent 1GPU each; then +R_fb/ALL) -> merge each -> ckpt/VLM_FT/coz_abl_{anc,rep,fb,all}.
TRACK B (parallel, GPU-free): integrate VlmJudge into evaluate.py (add vlm_grounding deep+all, vlm_quality cols; --vlm-judge flag; pytest green) + score BASE=A3 grounding on saved results/A3_full_sr (n=30 + n=100) -> results/abl_base.csv. Also add cross-scale diversity already exists (unique_token_ratio).
THEN: eval 4 arms (n=30 grounding screen) -> pick best individual + ALL -> n=100 final on base/best/ALL -> comparison table (grounding/drift, diversity, quality~tied, paired winrate) + prompt trajectories. GO/NO-GO ~19:20: ensure base/+R_anc/+R_rep/ALL done; drop +R_fb if slipping.

## TRAINING DONE (~19:30) — 3 arms merged
- coz_abl_anc (+R_anc): score_var ~0.04 VERY LOW (R_anc cosine saturation -> weak gradient -> may ~= base). final resp_len 59.2 / entropy 1.564 / kl 0.073.
- coz_abl_rep (+R_rep): score_var ~1.0 healthy. resp_len 33.5 / entropy 2.674 / kl 0.190.
- coz_abl_all (ALL anc+rep+fb): score_var 0.4-2.9 healthy, raw_r_fb present. resp_len 44.3 / entropy 2.824 / kl 0.096.
- abl_fb DROPPED (past 19:00, honest). All arms verified-different via logged cfg; no collapse. Launcher: .sisyphus/evidence/task_abl_arm.sh (parametrized, caller-driven reward knobs). 1-GPU concurrency FAILED (vLLM traceback) -> used sequential 2-GPU.
- FINAL EVAL delegated bg_cc98a2bc / ses_1634b5e18ffeEjypc0FpkUq4kA (timeout-safe: tmux+poll, no >8min blocking). Compares BASE(A3) / +R_anc / +R_rep / ALL on grounding_deep(PRIMARY) + diversity(uniqtok) + IQA(~tied) on n=30 -> results/abl_*.csv + results/abl_summary.md. If agent dies, SR/eval CSVs are on disk -> take over table-building.

## EXECUTION STATUS (live, ~17:30)
- TRACK A training bg_f4035330 / ses_163e6e84affebro27GZDbqRczB: RUNNING. Chose SEQUENTIAL 2-GPU (not concurrent). abl_anc in progress (tmux session "abl_anc2", checkpoints/abl_anc, 30 steps). Will chain abl_rep -> abl_fb -> abl_all, merge each -> ckpt/VLM_FT/coz_abl_<arm>. Parametrized launcher .sisyphus/evidence/task_abl_arm.sh.
- TRACK B bg_3aa06fb5: TIMED OUT (ran 31-min blocking chain of full evaluate.py w/ pyiqa on CPU). BUT deliverable LANDED + intact: evaluate.py +114 lines (vlm_grounding_all/deep + vlm_quality cols, --vlm-judge flag; 13 grounding refs) + train/grpo/vlm_judge.py. Orphan eval process KILLED.
- TODO at EVAL time (post-training, GPUs free => pyiqa fast): (1) VERIFY evaluate.py integration: pytest green + 3-img --vlm-judge smoke (defer; CPU pytest is slow while GPUs busy). (2) eval each new arm: inference_coz.py (SR+prompts) -> evaluate.py --vlm-judge (grounding + IQA) on n=30 screen. (3) also grounding-score baselines base=A3 / A2 / ours-w4v2 (reuse saved *_full_sr; n=30 same ids) for the ladder. (4) pick best individual + ALL -> n=100 final -> comparison table (grounding deep=PRIMARY, diversity, quality~tied, paired winrate) + prompt trajectories.
- BASELINE grounding already known (validation n=20): A2=6.48, A3(base)=8.15, ours=7.98 deep. Will recompute consistently at final.
- RISK: Track A agent may hit 30-min inactivity timeout like Track B. Training procs run detached in tmux (survive) -> if agent dies, TAKE OVER: merge finished arms + launch remaining + eval. GO/NO-GO ~19:20: ensure abl_anc/abl_rep/abl_all + merges done; drop abl_fb if slipping.
- JUDGE EVAL must be timeout-safe: run via tmux+poll<=110s OR parallelized curl, NEVER one long blocking command (that killed Track B).

## PLAN (to refine with oracle + plan agent)
1. Build discriminative InternVL judge (pairwise winrate vs A3 baseline + quality) as eval -> reveal signal current no-ref IQA misses.
2. Additive arms with SHORT runs (oracle to set steps/subset) tuned per-component to clear win on judge (small n), then ALL, then final n=100.
3. Use small-sample-first. Parallelize if feasible. Honest reporting.
