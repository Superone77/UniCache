"""Shared constants for UniCache operator and scheduler contracts."""

SUPPORTED_FAKE_QUANT_BITS = frozenset({2, 3, 4, 8, 16})

# BAGEL adapter token-type contract. Keeping the ID here avoids importing the
# model package from generic operators while making the mechanical active-state
# exclusion explicit.
BAGEL_KV_TYPE_CURRENT_VAE = 4
