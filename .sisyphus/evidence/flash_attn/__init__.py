# W3.T1 smoke shim: veRL imports flash_attn.bert_padding unconditionally even
# when actor_rollout_ref.model.use_remove_padding=False.  Do not use for real
# remove-padding training.
