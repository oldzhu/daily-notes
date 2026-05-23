The current implementation already handles already-patched environments — let me confirm by walking through the guards in _patch_modeling_init_rope_inplace:

1. **Marker check (primary guard)** — at the top of the function:
   ```python
   if _INIT_ROPE_PATCH_MARKER in text:
       print(f"[preprocess][init-rope-patch] {label}: ... already patched; skip")
       return
   ```
   The marker string is `"transformers>=4.43 standardizes rope_scaling"`, which is **byte-identical to the comment text upstream PR #10 inserts**. So this catches:
   - files we patched ourselves (our patcher writes the same comment), AND
   - files OpenBMB updated upstream (e.g. if HF re-downloaded a newer revision that already contains the fix).

2. **Pattern-count guard (secondary)** — if old patterns are missing for any reason (already patched by a different tool, modeling file refactored upstream, etc.):
   ```python
   header_count = text.count(_INIT_ROPE_OLD_HEADER)
   else_count  = text.count(_INIT_ROPE_OLD_ELSE)
   if header_count == 0 or else_count == 0:
       print(... "expected patterns not found ... skip")
       return
   ```

3. **No-op guard (tertiary)** — final safety net before write:
   ```python
   if patched == text:
       print(... "no changes computed ... skip")
       return
   ```

I already verified both code paths in the local round-trip test:

- Run on already-patched (post-fix) file → log `already patched; skip`, file untouched ✓
- Run on synthetic pre-fix file → produces byte-exact upstream output ✓
- Second run on the just-patched file → skips ✓

So no code change needed — the patcher is safe to invoke unconditionally on any environment, including ones that already have the upstream fix or were patched by a previous preprocess run.
