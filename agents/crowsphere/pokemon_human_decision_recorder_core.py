from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .obs_text_preproc import preprocess_obs_text_for_model
from .pokemon_tools import (
    fallback_action_str_for_pokemon,
    pokemon_decision_support,
    pokemon_milestone_progress,
)

from .pokemon_human_decision_recorder_io import load_checkpoint, save_checkpoint, save_screenshot
from .pokemon_human_decision_recorder_reports import write_highlights_md, write_summary_md


class HumanDecisionRecorder:
    """
    Pokemon Red 人类决策轨迹记录器

    - 调用 pokemon_decision_support 生成候选动作
    - 选择最高优先级候选（确定性基线）
    - 记录完整决策上下文、截图、checkpoint、summary/highlights
    """

    def __init__(self, output_dir: str, config: dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.trajectory_file = self.output_dir / f"trajectory_{self.timestamp}.jsonl"
        self.highlights_file = self.output_dir / f"highlights_{self.timestamp}.md"
        self.summary_file = self.output_dir / f"summary_{self.timestamp}.md"

        self.screenshots_dir = self.output_dir / f"screenshots_{self.timestamp}"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / f"checkpoints_{self.timestamp}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 截图策略：每 N 步保存一次 + 特殊场景 + 里程碑
        self.screenshot_interval = 10
        self.save_all_screenshots = False

        # Checkpoint 策略
        self.checkpoint_interval = 0
        self.auto_checkpoint_on_milestone = True

        # 轨迹与高亮
        self.trajectory_log: list[dict[str, Any]] = []
        self.highlights: dict[str, Any] = {
            "dialog_sequences": [],
            "battle_sequences": [],
            "stuck_recoveries": [],
            "milestone_moments": [],
        }

        # 内存状态（尽量对齐 CrowSphere 引擎的字段命名）
        self.memory: dict[str, Any] = {
            "last_action": "",
            "note": None,
            "score_estimate": 0,
            "prev_state": "",
            "subtask_step_count": 0,
            "pos": None,
            "pos_hist": [],
        }

        # next step index（从 0 开始；load checkpoint 后会跳到 step+1）
        self.step_count = 0
        self.prev_obs_text = ""
        self.start_time = time.time()
        self.final_score_estimate: int | None = None

        # 决策统计
        self.stats: dict[str, Any] = {
            "env_tool_count": 0,
            "keys_count": 0,
            "fallback_count": 0,
            "by_state": {},
        }

    def _extract_key_info(self, obs_text: str) -> dict[str, Any]:
        state_match = re.search(r"State:\s*(\w+)", obs_text)
        state = state_match.group(1) if state_match else "UNKNOWN"

        map_match = re.search(r"Map Name:\s*([^,\n]+)", obs_text)
        map_name = map_match.group(1).strip() if map_match else "N/A"

        pos_match = re.search(r"Your position \(x, y\):\s*\((\d+),\s*(\d+)\)", obs_text)
        pos = f"({pos_match.group(1)},{pos_match.group(2)})" if pos_match else "N/A"

        party_match = re.search(r"\[Current Party\]\n(.*?)(?=\[Badge List\]|\[Bag\]|$)", obs_text, re.DOTALL)
        has_pokemon = bool(party_match) and ("Name" in party_match.group(1))

        return {
            "state": state,
            "map_name": map_name,
            "position": pos,
            "has_pokemon": has_pokemon,
        }

    def _candidate_to_action_str(self, candidate: dict[str, Any]) -> str:
        if candidate.get("env_tool"):
            tool = candidate["env_tool"]
            tool_name = tool["name"]
            tool_args = tool.get("args", {})

            if not tool_args:
                return f"use_tool({tool_name}, ())"

            args_str = ", ".join(f"{k}={repr(v)}" for k, v in tool_args.items())
            return f"use_tool({tool_name}, ({args_str}))"

        keys = candidate.get("keys")
        if isinstance(keys, list) and keys:
            return " ".join(keys) if keys else "pass"
        return "pass"

    def decide_action(self, obs_text: str) -> tuple[str, dict[str, Any]]:
        """
        确定性决策：调用 pokemon_decision_support，并选择最高优先级候选。
        """
        try:
            ds = pokemon_decision_support(
                obs_text=obs_text,
                memory=self.memory,
                config=self.config,
                proposed_subtask=None,
            )
            decision_context = ds.to_dict()

            if not ds.candidates:
                action_str = fallback_action_str_for_pokemon(obs_text)
                decision_context["selection"] = {
                    "method": "fallback_no_candidates",
                    "action_str": action_str,
                    "chosen_candidate": None,
                }
                self.stats["fallback_count"] += 1
                return action_str, decision_context

            top_candidate = ds.candidates[0]
            action_str = self._candidate_to_action_str(top_candidate)
            decision_context["selection"] = {
                "method": "top_priority_candidate",
                "action_str": action_str,
                "chosen_candidate": top_candidate,
            }

            if top_candidate.get("env_tool"):
                self.stats["env_tool_count"] += 1
            else:
                self.stats["keys_count"] += 1

            return action_str, decision_context
        except Exception as e:
            action_str = fallback_action_str_for_pokemon(obs_text)
            decision_context = {
                "error": str(e),
                "selection": {
                    "method": "fallback_on_error",
                    "action_str": action_str,
                },
            }
            self.stats["fallback_count"] += 1
            return action_str, decision_context

    def record_step(
        self,
        *,
        obs_text: str,
        decision_context: dict[str, Any],
        action_str: str,
        next_obs_text: str,
        post_progress: dict[str, Any] | None,
        milestone_achieved: bool,
        obs: Any | None,
        next_obs: Any | None,
    ) -> None:
        key_info = self._extract_key_info(obs_text)
        next_key_info = self._extract_key_info(next_obs_text)

        # 状态统计（用 pre-state 更直观：当步是在什么状态下做决策）
        state = key_info["state"]
        by_state: dict[str, int] = self.stats["by_state"]
        by_state[state] = by_state.get(state, 0) + 1

        # 截图策略
        screenshot_path: str | None = None
        should_save = self.save_all_screenshots or (self.step_count % self.screenshot_interval == 0)

        if milestone_achieved and next_obs is not None:
            should_save = True
            score = int(post_progress.get("score_estimate", 0) or 0) if isinstance(post_progress, dict) else 0
            screenshot_path = save_screenshot(
                obs=next_obs,
                step=self.step_count,
                screenshots_dir=self.screenshots_dir,
                timestamp=self.timestamp,
                tag=f"milestone_{score}",
            )
        elif state in ["Battle", "Dialog"] or decision_context.get("anti_stuck", {}).get("note"):
            should_save = True
            tag = state.lower() if state in ["Battle", "Dialog"] else "stuck"
            if obs is not None:
                screenshot_path = save_screenshot(
                    obs=obs,
                    step=self.step_count,
                    screenshots_dir=self.screenshots_dir,
                    timestamp=self.timestamp,
                    tag=tag,
                )
        elif should_save and obs is not None:
            screenshot_path = save_screenshot(
                obs=obs,
                step=self.step_count,
                screenshots_dir=self.screenshots_dir,
                timestamp=self.timestamp,
            )

        record: dict[str, Any] = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "observation": {
                "key_info": key_info,
                "full_text": obs_text,
            },
            "decision": decision_context,
            "action": action_str,
            "result": {
                "key_info": next_key_info,
                "state_changed": key_info["state"] != next_key_info["state"],
                "map_changed": key_info["map_name"] != next_key_info["map_name"],
            },
        }
        if isinstance(post_progress, dict) and post_progress:
            record["result"]["progress"] = post_progress
        if screenshot_path:
            record["screenshot"] = screenshot_path

        with open(self.trajectory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.trajectory_log.append(record)
        self._detect_highlights(record, milestone_achieved=milestone_achieved)

    def _detect_highlights(self, record: dict[str, Any], *, milestone_achieved: bool) -> None:
        state = record.get("observation", {}).get("key_info", {}).get("state", "UNKNOWN")

        # Dialog 场景
        if state == "Dialog":
            if (not self.highlights["dialog_sequences"]) or (
                self.highlights["dialog_sequences"][-1][-1]["step"] != record["step"] - 1
            ):
                self.highlights["dialog_sequences"].append([])
            self.highlights["dialog_sequences"][-1].append(record)

        # Battle 场景
        if "Battle" in state:
            if (not self.highlights["battle_sequences"]) or (
                self.highlights["battle_sequences"][-1][-1]["step"] != record["step"] - 1
            ):
                self.highlights["battle_sequences"].append([])
            self.highlights["battle_sequences"][-1].append(record)

        # 卡顿恢复
        anti_stuck = record.get("decision", {}).get("anti_stuck", {})
        if isinstance(anti_stuck, dict) and anti_stuck.get("note"):
            self.highlights["stuck_recoveries"].append(record)

        # 里程碑达成（基于 post-progress 判断，避免 next_milestone 偏移与最终里程碑缺失）
        if milestone_achieved:
            self.highlights["milestone_moments"].append(record)

    def update_memory_after_step(
        self,
        *,
        obs_text: str,
        next_obs_text: str,
        action_str: str,
        decision_context: dict[str, Any],
        post_progress: dict[str, Any] | None,
    ) -> None:
        # last_action / prev_state：用于后续 anti-stuck 与 Battle->Field 里程碑判断
        self.memory["last_action"] = action_str
        m_state = re.search(r"State:\s*(\w+)", obs_text)
        self.memory["prev_state"] = m_state.group(1) if m_state else ""

        # note：本次动作是否导致观测不变（更贴合“卡住”定义）
        self.memory["note"] = "unchanged observation" if (obs_text == next_obs_text) else None

        # pos / map：用于子任务与候选生成（用 post-state 更合理）
        pos_match = re.search(r"Your position \(x, y\):\s*\((\d+),\s*(\d+)\)", next_obs_text)
        if pos_match:
            x, y = int(pos_match.group(1)), int(pos_match.group(2))
            self.memory["pos"] = (x, y)
            pos_hist = list(self.memory.get("pos_hist", []))
            pos_hist.append((x, y))
            self.memory["pos_hist"] = pos_hist[-4:]

        map_match = re.search(r"Map Name:\s*([^,\n]+)", next_obs_text)
        if map_match:
            self.memory["map"] = map_match.group(1).strip()

        # 里程碑进度：以 post-progress 为准（避免最后一步 break 时 summary 仍显示 6/7）
        if isinstance(post_progress, dict) and post_progress:
            self.memory["score_estimate"] = int(post_progress.get("score_estimate", 0) or 0)
            reached = post_progress.get("reached")
            if isinstance(reached, list):
                self.memory["reached_milestones"] = reached
        else:
            # 兜底：不下降
            try:
                self.memory["score_estimate"] = max(
                    int(self.memory.get("score_estimate", 0) or 0),
                    int(decision_context.get("progress", {}).get("score_estimate", 0) or 0),
                )
            except Exception:
                pass

        # 子任务状态（来自 decision_context）
        subtask = decision_context.get("subtask", {})
        if isinstance(subtask, dict):
            accepted = subtask.get("accepted")
            if isinstance(accepted, str) and accepted:
                prev_subtask = self.memory.get("current_subtask")
                if prev_subtask == accepted:
                    self.memory["subtask_step_count"] = int(self.memory.get("subtask_step_count", 0) or 0) + 1
                else:
                    self.memory["current_subtask"] = accepted
                    self.memory["subtask_step_count"] = 0

        self.prev_obs_text = next_obs_text

    def save_checkpoint(self, env: Any, *, step: int, tag: str) -> str | None:
        name = save_checkpoint(
            env=env,
            step=step,
            tag=tag,
            checkpoint_dir=self.checkpoint_dir,
            memory=self.memory,
            stats=self.stats,
            prev_obs_text=self.prev_obs_text,
            config=self.config,
        )
        if name:
            print(f"💾 Checkpoint saved: {name}")
        return name

    def load_checkpoint(self, env: Any, checkpoint_path: str) -> bool:
        metadata = load_checkpoint(env=env, checkpoint_path=checkpoint_path)
        if not metadata:
            return False

        self.step_count = int(metadata.get("step", 0) or 0) + 1
        self.memory = metadata.get("memory", self.memory)
        self.prev_obs_text = metadata.get("prev_obs_text", "")

        stats = metadata.get("stats", {})
        if isinstance(stats, dict):
            self.stats["env_tool_count"] = int(stats.get("env_tool_count", 0) or 0)
            self.stats["keys_count"] = int(stats.get("keys_count", 0) or 0)
            self.stats["fallback_count"] = int(stats.get("fallback_count", 0) or 0)
            self.stats["by_state"] = stats.get("by_state", {}) if isinstance(stats.get("by_state", {}), dict) else {}

        print(f"✅ Checkpoint loaded from step {self.step_count - 1}")
        print(f"   Memory score: {self.memory.get('score_estimate', 0)}/7")
        print(f"   Previous state: {self.memory.get('prev_state', 'N/A')}")
        return True

    def generate_reports(self) -> None:
        elapsed = time.time() - self.start_time
        total_steps = len(self.trajectory_log)
        final_score = int(self.final_score_estimate if self.final_score_estimate is not None else self.memory.get("score_estimate", 0) or 0)

        write_highlights_md(
            highlights_file=self.highlights_file,
            timestamp=self.timestamp,
            total_steps=total_steps,
            highlights=self.highlights,
        )
        write_summary_md(
            summary_file=self.summary_file,
            timestamp=self.timestamp,
            total_steps=total_steps,
            final_score=final_score,
            elapsed_time=elapsed,
            stats=self.stats,
            highlights=self.highlights,
        )

        screenshot_count = sum(1 for r in self.trajectory_log if "screenshot" in r)
        checkpoint_count = sum(1 for _ in self.checkpoint_dir.glob("step_*"))

        print("\n✅ 报告已生成:")
        print(f"  📄 完整轨迹: {self.trajectory_file} ({len(self.trajectory_log)} 步)")
        print(f"  🎯 特殊场景: {self.highlights_file}")
        print(f"  📈 总结报告: {self.summary_file}")
        print(f"  📸 游戏截图: {self.screenshots_dir} ({screenshot_count} 张)")
        if checkpoint_count > 0:
            print(f"  💾 Checkpoints: {self.checkpoint_dir} ({checkpoint_count} 个)")

    def run(self, env: Any, *, max_steps: int) -> None:
        obs = env.initial_obs()
        print("\n🎮 开始记录 Pokemon Red 决策轨迹...")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"⏱️  最大步数: {max_steps}")
        print("─" * 60)

        for _ in range(max_steps):
            step = self.step_count

            raw_obs_text = obs.state_text
            obs_text = preprocess_obs_text_for_model(game="pokemon_red", obs_text=raw_obs_text).text
            prev_score_estimate = int(self.memory.get("score_estimate", 0) or 0)
            current_state = self._extract_key_info(obs_text).get("state", "")

            action_str, decision_context = self.decide_action(obs_text)

            try:
                from mcp_game_servers.pokemon_red.game.pokemon_red_env import PokemonRedAction

                action = PokemonRedAction(action=action_str)
                next_obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"\n❌ Step {step} 执行失败: {e}")
                break

            raw_next_obs_text = next_obs.state_text
            next_obs_text = preprocess_obs_text_for_model(game="pokemon_red", obs_text=raw_next_obs_text).text

            # post-progress：用于里程碑记录/summary（避免最后一步 break 时缺失 7/7）
            mem_for_progress = dict(self.memory)
            mem_for_progress["prev_state"] = current_state
            mem_for_progress["score_estimate"] = prev_score_estimate
            post_progress_obj = pokemon_milestone_progress(obs_text=next_obs_text, memory=mem_for_progress)
            post_progress = post_progress_obj.to_dict()
            post_score = int(post_progress_obj.score_estimate)
            milestone_achieved = post_score > prev_score_estimate

            self.record_step(
                obs_text=obs_text,
                decision_context=decision_context,
                action_str=action_str,
                next_obs_text=next_obs_text,
                post_progress=post_progress,
                milestone_achieved=milestone_achieved,
                obs=obs,
                next_obs=next_obs,
            )

            self.update_memory_after_step(
                obs_text=obs_text,
                next_obs_text=next_obs_text,
                action_str=action_str,
                decision_context=decision_context,
                post_progress=post_progress,
            )

            # 自动保存 checkpoint
            should_save_checkpoint = False
            checkpoint_tag = ""
            if self.checkpoint_interval > 0 and step > 0 and step % self.checkpoint_interval == 0:
                should_save_checkpoint = True
                checkpoint_tag = "auto"
            if self.auto_checkpoint_on_milestone and milestone_achieved:
                should_save_checkpoint = True
                checkpoint_tag = f"milestone_{post_score}"
            if should_save_checkpoint:
                self.save_checkpoint(env, step=step, tag=checkpoint_tag)

            # 进度显示（用 post-score 更直观）
            if step % 10 == 0:
                map_name = self._extract_key_info(next_obs_text).get("map_name", "?")
                state = self._extract_key_info(next_obs_text).get("state", "?")
                print(f"Step {step:3d}: 得分 {post_score}/7 | 状态: {state:10s} | 地图: {map_name}")

            # 检查结束条件
            try:
                score, done = env.evaluate(next_obs)
                if done:
                    self.final_score_estimate = int(score)
                    print(f"\n✅ 完成! 最终得分: {score}/7")
                    break
            except Exception:
                pass

            obs = next_obs
            self.step_count += 1

        self.generate_reports()
