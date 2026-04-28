from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


METRIC_KEY = "val-core/HuggingFaceH4/MATH-500/acc/mean@1"


def _make_checkpoint(root, step, with_optimizer=True):
    actor_dir = root / f"global_step_{step}" / "actor"
    actor_dir.mkdir(parents=True)
    (actor_dir / "model_world_size_8_rank_0.pt").write_text("model")
    if with_optimizer:
        (actor_dir / "optim_world_size_8_rank_0.pt").write_text("optim")
    return actor_dir.parent


def _make_trainer(root):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(root),
                "keep_best_ckpt": True,
                "best_ckpt_metric_key": METRIC_KEY,
                "best_ckpt_metric_mode": "max",
                "best_ckpt_strip_optimizer": True,
            }
        }
    )
    trainer.best_ckpt_step = None
    trainer.best_ckpt_metric_value = None
    return trainer


def test_best_latest_retention_keeps_latest_full_and_best_model_only(tmp_path):
    _make_checkpoint(tmp_path, 25)
    _make_checkpoint(tmp_path, 50)
    _make_checkpoint(tmp_path, 75)

    trainer = _make_trainer(tmp_path)

    trainer.global_steps = 25
    trainer._maybe_update_best_checkpoint({METRIC_KEY: 0.7})

    trainer.global_steps = 50
    trainer._maybe_update_best_checkpoint({METRIC_KEY: 0.6})
    trainer._cleanup_best_latest_checkpoints(latest_step=50)

    assert (tmp_path / "global_step_25").is_dir()
    assert (tmp_path / "global_step_50").is_dir()
    assert not (tmp_path / "global_step_75").exists()
    assert not (tmp_path / "global_step_25/actor/optim_world_size_8_rank_0.pt").exists()
    assert (tmp_path / "global_step_50/actor/optim_world_size_8_rank_0.pt").exists()


def test_new_best_removes_previous_best(tmp_path):
    _make_checkpoint(tmp_path, 25)

    trainer = _make_trainer(tmp_path)

    trainer.global_steps = 25
    trainer._maybe_update_best_checkpoint({METRIC_KEY: 0.7})
    trainer._cleanup_best_latest_checkpoints(latest_step=25)

    trainer._cleanup_best_latest_checkpoints(latest_step=None)
    _make_checkpoint(tmp_path, 50)
    trainer.global_steps = 50
    trainer._maybe_update_best_checkpoint({METRIC_KEY: 0.8})
    trainer._cleanup_best_latest_checkpoints(latest_step=50)

    assert not (tmp_path / "global_step_25").exists()
    assert (tmp_path / "global_step_50").is_dir()
    assert (tmp_path / "global_step_50/actor/optim_world_size_8_rank_0.pt").exists()
