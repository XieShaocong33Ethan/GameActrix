from __future__ import annotations

from pathlib import Path
from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _milestone_progress_from_record(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result", {})
    post_progress = result.get("progress")
    if isinstance(post_progress, dict) and post_progress:
        return post_progress
    decision = record.get("decision", {})
    pre_progress = decision.get("progress")
    if isinstance(pre_progress, dict) and pre_progress:
        return pre_progress
    return {}


def _milestone_achieved_id(progress: dict[str, Any]) -> str | None:
    reached = progress.get("reached")
    if isinstance(reached, list) and reached:
        last = reached[-1]
        return str(last) if isinstance(last, str) and last else None
    return None


def _best_key_info_for_milestone(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result", {})
    key_info = result.get("key_info")
    if isinstance(key_info, dict) and key_info:
        return key_info
    obs = record.get("observation", {})
    key_info = obs.get("key_info")
    if isinstance(key_info, dict) and key_info:
        return key_info
    return {}


def write_highlights_md(
    *,
    highlights_file: Path,
    timestamp: str,
    total_steps: int,
    highlights: dict[str, Any],
) -> None:
    with open(highlights_file, "w", encoding="utf-8") as f:
        f.write("# Pokemon Red 决策轨迹 - 特殊场景集锦\n\n")
        f.write(f"**记录时间**: {timestamp}\n\n")
        f.write(f"**总步数**: {total_steps}\n\n")
        f.write("---\n\n")

        dialog_sequences = highlights.get("dialog_sequences", [])
        battle_sequences = highlights.get("battle_sequences", [])
        stuck_recoveries = highlights.get("stuck_recoveries", [])
        milestone_moments = highlights.get("milestone_moments", [])

        # Dialog 场景
        f.write(f"## 🗨️ Dialog 场景 ({len(dialog_sequences)} 次)\n\n")
        for i, seq in enumerate(dialog_sequences[:5]):
            if not seq:
                continue
            f.write(f"### Dialog #{i+1} (Steps {seq[0]['step']}-{seq[-1]['step']})\n\n")
            if "screenshot" in seq[0]:
                f.write(f"![Dialog {i+1}]({seq[0]['screenshot']})\n\n")
            for record in seq:
                chosen = record.get("decision", {}).get("selection", {}).get("chosen_candidate", {})
                why = chosen.get("why", "N/A") if isinstance(chosen, dict) else "N/A"
                f.write(f"- **Step {record['step']}**: `{record.get('action', 'N/A')}`\n")
                f.write(f"  - 理由: {why}\n")
            f.write("\n")

        # Battle 场景
        f.write(f"## ⚔️ Battle 场景 ({len(battle_sequences)} 次)\n\n")
        for i, seq in enumerate(battle_sequences):
            if not seq:
                continue
            f.write(f"### Battle #{i+1} (Steps {seq[0]['step']}-{seq[-1]['step']})\n\n")
            if "screenshot" in seq[0]:
                f.write(f"![Battle {i+1}]({seq[0]['screenshot']})\n\n")
            for record in seq:
                chosen = record.get("decision", {}).get("selection", {}).get("chosen_candidate", {})
                why = chosen.get("why", "N/A") if isinstance(chosen, dict) else "N/A"
                f.write(f"- **Step {record['step']}**: `{record.get('action', 'N/A')}`\n")
                f.write(f"  - 理由: {why}\n")
            f.write("\n")

        # 卡顿恢复
        f.write(f"## 🔄 卡顿恢复 ({len(stuck_recoveries)} 次)\n\n")
        for record in stuck_recoveries:
            anti_stuck = record.get("decision", {}).get("anti_stuck", {})
            note = anti_stuck.get("note", "N/A") if isinstance(anti_stuck, dict) else "N/A"
            last_action = anti_stuck.get("last_action", "N/A") if isinstance(anti_stuck, dict) else "N/A"
            f.write(f"- **Step {record['step']}**: 检测到卡顿\n")
            f.write(f"  - 诊断: {note}\n")
            f.write(f"  - 上次动作: {last_action}\n")
            f.write(f"  - 恢复动作: `{record.get('action', 'N/A')}`\n\n")

        # 里程碑达成
        f.write(f"## 🎯 里程碑达成 ({len(milestone_moments)} 次)\n\n")
        for record in milestone_moments:
            progress = _milestone_progress_from_record(record)
            score = _safe_int(progress.get("score_estimate", 0), 0)
            achieved = _milestone_achieved_id(progress) or "N/A"
            next_goal = progress.get("next_milestone")
            next_goal_str = str(next_goal) if isinstance(next_goal, str) and next_goal else "COMPLETE"
            key_info = _best_key_info_for_milestone(record)
            map_name = key_info.get("map_name", "N/A")
            position = key_info.get("position", "N/A")

            f.write(f"### 里程碑 #{score} (Step {record['step']})\n\n")
            if "screenshot" in record:
                f.write(f"![Milestone {score}]({record['screenshot']})\n\n")
            f.write(f"- **得分**: {score}/7\n")
            f.write(f"- **达成里程碑**: {achieved}\n")
            f.write(f"- **下一目标**: {next_goal_str}\n")
            f.write(f"- **地图**: {map_name}\n")
            f.write(f"- **位置**: {position}\n\n")


def write_summary_md(
    *,
    summary_file: Path,
    timestamp: str,
    total_steps: int,
    final_score: int,
    elapsed_time: float,
    stats: dict[str, Any],
    highlights: dict[str, Any],
) -> None:
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# Pokemon Red 人类决策轨迹 - 总结报告\n\n")
        f.write(f"**记录时间**: {timestamp}\n\n")
        f.write("---\n\n")

        # 运行概览
        f.write("## 📊 运行概览\n\n")
        f.write(f"- **总步数**: {total_steps}\n")
        f.write(f"- **最终得分**: {final_score}/7\n")
        f.write(f"- **运行时间**: {elapsed_time:.2f} 秒\n")
        f.write(f"- **平均每步**: {elapsed_time/max(1, total_steps):.2f} 秒\n\n")

        # 决策统计
        env_tool_count = _safe_int(stats.get("env_tool_count", 0), 0)
        keys_count = _safe_int(stats.get("keys_count", 0), 0)
        fallback_count = _safe_int(stats.get("fallback_count", 0), 0)
        by_state = stats.get("by_state", {}) if isinstance(stats.get("by_state", {}), dict) else {}

        f.write("## 🎯 决策统计\n\n")
        f.write("| 类型 | 次数 | 占比 |\n")
        f.write("|------|------|------|\n")
        total = max(1, total_steps)
        f.write(f"| 环境工具 (env_tool) | {env_tool_count} | {env_tool_count/total*100:.1f}% |\n")
        f.write(f"| 按键序列 (keys) | {keys_count} | {keys_count/total*100:.1f}% |\n")
        f.write(f"| 降级方案 (fallback) | {fallback_count} | {fallback_count/total*100:.1f}% |\n\n")

        f.write("### 按游戏状态统计\n\n")
        f.write("| 状态 | 步数 |\n")
        f.write("|------|------|\n")
        for state, count in sorted(by_state.items(), key=lambda x: -_safe_int(x[1], 0)):
            f.write(f"| {state} | {_safe_int(count, 0)} |\n")
        f.write("\n")

        # 特殊场景统计
        dialog_sequences = highlights.get("dialog_sequences", [])
        battle_sequences = highlights.get("battle_sequences", [])
        stuck_recoveries = highlights.get("stuck_recoveries", [])
        milestone_moments = highlights.get("milestone_moments", [])

        f.write("## 🎬 特殊场景统计\n\n")
        f.write(f"- **Dialog 序列**: {len(dialog_sequences)} 次\n")
        f.write(f"- **Battle 序列**: {len(battle_sequences)} 次\n")
        f.write(f"- **卡顿恢复**: {len(stuck_recoveries)} 次\n")
        f.write(f"- **里程碑达成**: {len(milestone_moments)} 次\n\n")

        # 里程碑时间线
        f.write("## 🗺️ 里程碑时间线\n\n")
        f.write("| 步数 | 得分 | 达成里程碑 | 地图 |\n")
        f.write("|------|------|------------|------|\n")
        for record in milestone_moments:
            progress = _milestone_progress_from_record(record)
            score = _safe_int(progress.get("score_estimate", "?"), 0)
            achieved = _milestone_achieved_id(progress) or "N/A"
            map_name = _best_key_info_for_milestone(record).get("map_name", "N/A")
            f.write(f"| {record['step']} | {score}/7 | {achieved} | {map_name} |\n")
        f.write("\n")

