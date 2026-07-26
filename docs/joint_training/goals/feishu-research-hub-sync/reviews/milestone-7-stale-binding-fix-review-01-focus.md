Review only the IN_SCOPE repair for `F-M7-R02` at local Hub candidate `e354f950338bed8750e77653395efec4dc473907`, based on `7078fddab173cdfcfcfb8817bf9f5aeb98b1cd89`.

Required behavior:

- A guarded `binding-refresh` path may update only runtime-private CAS revision/editor metadata after fetching the existing approved object.
- It must require the runtime-private token, run the normal writer/root gate and adapter-version check, and emit no token/revision/editor value.
- It may refresh only when canonical remote content equals the recorded common edition. A remote semantic change must fail `E_BINDING_DIVERGED` without changing binding state, tracked files, common snapshot, Git, or Feishu.
- The command itself performs no Feishu write; it is a read plus protected local metadata refresh.
- Verify the positive and negative tests genuinely exercise those branches and that the complete suite/root gate remain green.
- Verify commit attribution is `Codex Agent <codex-agent@example.invalid>` with `Co-authored-by: GongxunLi <lgxma01@buaa.edu.cn>`.
- Do not call Feishu, push, modify ledgers, update the parent gitlink, or implement fixes. Use `PASS` only if the guarded refresh closes the stale-binding readiness defect without weakening CAS or privacy.
