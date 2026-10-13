import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pogema
from pogema import GridConfig, pogema_v0


SCENARIOS = [
    ("same_cell", [[2, 1], [2, 3]], [(0, 1), (0, -1)]),
    ("swap_cells", [[2, 1], [2, 2]], [(0, 1), (0, -1)]),
    ("following", [[2, 1], [2, 2]], [(0, 1), (0, 1)]),
]

SYSTEMS = ["priority", "block_both", "soft"]


def run_check(scenario, starts, shifts, system):
    # Во всех проверках карта и цели одинаковые.
    config = GridConfig(
        map="\n".join(["....."] * 5),
        agents_xy=[xy.copy() for xy in starts],
        targets_xy=[[0, 0], [4, 4]],
        num_agents=2,
        obs_radius=1,
        seed=42,
        on_target="nothing",
        collision_system=system,
        max_episode_steps=1,
    )

    # Находим индексы команд по направлениям движения.
    action_ids = {
        tuple(move): idx
        for idx, move in enumerate(config.MOVES)
    }
    requested = [action_ids[shift] for shift in shifts]

    env = pogema_v0(grid_config=config)

    try:
        env.reset()

        before = [
            list(xy)
            for xy in env.unwrapped.get_agents_xy(ignore_borders=True)
        ]

        # Soft может изменять список, поэтому передаём копию.
        env.step(requested.copy())

        after = [
            list(xy)
            for xy in env.unwrapped.get_agents_xy(ignore_borders=True)
        ]
    finally:
        env.close()

    if before != starts:
        raise RuntimeError(f"Unexpected starting positions: {before}")

    if len({tuple(xy) for xy in after}) != len(after):
        raise RuntimeError("Agents occupy the same final cell")

    moved = [
        old != new
        for old, new in zip(before, after)
    ]

    blocked = sum(
        shift != (0, 0) and not done
        for shift, done in zip(shifts, moved)
    )

    print(
        f"{scenario:12} | {system:10} | "
        f"moved={sum(moved)} | blocked={blocked} | "
        f"{before} -> {after}"
    )

    return {
        "experiment": "01_collisions",
        "scenario": scenario,
        "collision_system": system,
        "seed": 42,
        "num_agents": 2,
        "steps": 1,
        "requested_actions": json.dumps(requested),
        "before": json.dumps(before),
        "after": json.dumps(after),
        "moved_agents": sum(moved),
        "blocked_move_commands": blocked,
        "pogema_version": pogema.__version__,
        "python_version": sys.version.split()[0],
    }


def main():
    print("Experiment 01: collision rules")
    print("POGEMA:", pogema.__version__)

    rows = []

    for scenario, starts, shifts in SCENARIOS:
        for system in SYSTEMS:
            rows.append(
                run_check(scenario, starts, shifts, system)
            )

    timestamp = datetime.now().strftime(
        "exp01_%Y%m%d_%H%M%S_%f"
    )

    folder = (
        Path(__file__).resolve().parent
        / "results"
        / timestamp
    )
    folder.mkdir(parents=True, exist_ok=False)

    output = folder / "results.csv"

    with output.open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nChecks completed:", len(rows))
    print("Results saved:", output)


if __name__ == "__main__":
    main()
