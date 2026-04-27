---
name: Subagent model selection
description: Subagents should use Haiku model for cost efficiency, not Opus or Sonnet
type: feedback
originSessionId: 504dc4c8-b3f1-4ad0-bf9e-6e337aecb68d
---
When spawning subagents (Agent tool), always use `model: "haiku"` unless the task specifically requires strong reasoning capabilities. Subagents are for exploratory/independent work and context offloading — they don't need expensive models.

**Why:** Cost control. Subagent work is typically file exploration, log parsing, or simple searches that don't need Opus-level reasoning.

**How to apply:** Every Agent tool call should include `model: "haiku"` by default. Only escalate to Sonnet/Opus if the subagent task involves complex code generation, architectural reasoning, or multi-step problem solving.
