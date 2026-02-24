# Ray Temp Directory Configuration

## Summary
Ray temp directory has been successfully moved from `/tmp/ray/` to `/data-1/.cache/ray/` to keep the system /tmp clean.

## Configuration Details

### Location
- **New Ray temp directory:** `/data-1/.cache/ray/`
- **Old location (cleaned):** `/tmp/ray/` ✓ Removed

### Automatic Configuration
The verl07 conda environment is configured to automatically set `RAY_TMPDIR` when activated.

**Activation script:** `/data-1/.cache/conda/envs/verl07/etc/conda/activate.d/env_vars.sh`
```bash
#!/bin/sh
export RAY_TMPDIR=/data-1/.cache/ray
```

**Deactivation script:** `/data-1/.cache/conda/envs/verl07/etc/conda/deactivate.d/env_vars.sh`
```bash
#!/bin/sh
unset RAY_TMPDIR
```

## Usage

### Normal Usage
Simply activate the environment and Ray will use the correct directory:
```bash
conda activate verl07
# RAY_TMPDIR is automatically set to /data-1/.cache/ray

# Run your verl training
cd /data-1/verl07/verl
bash examples/grpo_trainer/run_deepseek7b_llm.sh
```

### Verify Configuration
```bash
conda activate verl07
echo $RAY_TMPDIR
# Output: /data-1/.cache/ray
```

### Test Ray
```bash
conda activate verl07
python -c "import ray; ray.init(); print('Ray session:', ray._private.worker.global_worker.node.get_session_dir_path()); ray.shutdown()"
# Should show path under /data-1/.cache/ray/
```

## Directory Structure
```
/data-1/.cache/ray/
└── ray/
    ├── session_latest -> (symlink to current session)
    ├── session_YYYY-MM-DD_HH-MM-SS_XXXXXX_XXXXXX/
    │   ├── sockets/          # Ray internal communication
    │   ├── logs/             # Ray logs and debug info
    │   ├── runtime_resources/ # Runtime resource tracking
    │   └── metrics/          # Performance metrics
    ├── prom_metrics_service_discovery.json
    └── tmp_prom_metrics_service_discovery.json
```

## Maintenance

### Check Ray Sessions
```bash
conda activate verl07
ray status
```

### Stop All Ray Processes
```bash
conda activate verl07
ray stop --force
```

### Clean Old Sessions
```bash
# Remove old session directories
rm -rf /data-1/.cache/ray/ray/session_*

# Or clean everything
rm -rf /data-1/.cache/ray/ray/
```

### Monitor Disk Usage
```bash
du -sh /data-1/.cache/ray/
```

## Benefits
- ✓ System `/tmp` stays clean
- ✓ Ray temp files in dedicated cache location
- ✓ Easier to manage and monitor Ray sessions
- ✓ Automatic configuration per conda environment
- ✓ No manual export needed each time
- ✓ Persistent across terminal sessions

## Notes
- Configuration applies only when verl07 environment is activated
- Other conda environments are not affected
- Ray will create subdirectories automatically as needed
- Sessions persist until explicitly stopped or cleaned

## Troubleshooting

### If Ray still uses /tmp/ray
1. Make sure you activated the environment: `conda activate verl07`
2. Verify the variable is set: `echo $RAY_TMPDIR`
3. If not set, manually export: `export RAY_TMPDIR=/data-1/.cache/ray`

### If activation scripts don't work
The scripts are sourced automatically by conda. If they don't work:
1. Check they exist and are executable
2. Deactivate and reactivate the environment
3. Start a new terminal session

### Manual override (if needed)
You can still override per session:
```bash
export RAY_TMPDIR=/your/custom/path
```

## Installation Date
2026-02-24

## Related Files
- Activation script: `/data-1/.cache/conda/envs/verl07/etc/conda/activate.d/env_vars.sh`
- Deactivation script: `/data-1/.cache/conda/envs/verl07/etc/conda/deactivate.d/env_vars.sh`
- Ray cache directory: `/data-1/.cache/ray/`
- verl installation: `/data-1/verl07/verl/`
