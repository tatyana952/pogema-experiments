import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import pogema
from pogema import BatchAStarAgent, GridConfig, pogema_v0


SCENARIOS = [
    (
        "offset",
        [[3, 1], [6, 2], [3, 12], [6, 11]],
        [[3, 12], [6, 11], [3, 1], [6, 2]],
    ),
    (
        "aligned",
        [[4, 1], [4, 3], [4, 12], [4, 10]],
        [[4, 12], [4, 10], [4, 1], [4, 3]],
    ),
    (
        "diagonal",
        [[1, 2], [8, 3], [8, 11], [1, 12]],
        [[8, 11], [1, 12], [1, 2], [8, 3]],
    ),
]


def make_map(width):
    # Две комнаты и проход длиной четыре клетки.
    grid = [[0] * 14 for _ in range(10)]

    for row in range(10):
        for col in range(5, 9):
            grid[row][col] = 1

    # Расширяем тот же проход дополнительной полосой.
    for row in range(4, 4 + width):
        for col in range(5, 9):
            grid[row][col] = 0

    return grid


def run_episode(name, starts, goals, width):
    config = GridConfig(
        map=make_map(width),
        agents_xy=[xy.copy() for xy in starts],
        targets_xy=[xy.copy() for xy in goals],
        num_agents=4,
        seed=42,
        obs_radius=5,
        observation_type="POMAPF",
        collision_system="soft",
        on_target="nothing",
        max_episode_steps=128,
    )

    env = pogema_v0(grid_config=config)

    agent = BatchAStarAgent()
    agent.reset_states()

    blocked = 0
    success = False
    metrics = {}

    try:
        observations, _ = env.reset()
        started = time.perf_counter()

        for steps in range(1, 129):
            before = [
                tuple(xy)
                for xy in env.unwrapped.get_agents_xy(
                    ignore_borders=True
                )
            ]

            actions = agent.act(observations)

            requested_moves = [
                tuple(config.MOVES[action]) != (0, 0)
                for action in actions
            ]

            observations, _, terminated, truncated, infos = (
                env.step(actions.copy())
            )

            after = [
                tuple(xy)
                for xy in env.unwrapped.get_agents_xy(
                    ignore_borders=True
                )
            ]

            blocked += sum(
                wanted and old == new
                for wanted, old, new in zip(
                    requested_moves, before, after
                )
            )

            if all(terminated) or all(truncated):
                success = bool(all(terminated))
                metrics = infos[0].get("metrics", {})
                break

        runtime = time.perf_counter() - started

    finally:
        env.close()

    # Стандартные метрики POGEMA.
    soc = metrics.get("SoC")
    makespan = metrics.get("makespan")

    if soc is None or makespan is None:
        raise RuntimeError(
            f"Missing SoC or makespan: {metrics}"
        )

    reached = sum(
        list(position) == goal
        for position, goal in zip(after, goals)
    )

    print(
        f"{name:8} | width={width} | "
        f"success={int(success)} | reached={reached}/4 | "
        f"steps={steps:3} | SoC={soc} | "
        f"makespan={makespan} | blocked={blocked}"
    )

    return {
        "experiment": "03_corridor",
        "scenario": name,
        "corridor_width": width,
        "seed": 42,
        "num_agents": 4,
        "success": int(success),
        "reached_at_end": reached,
        "steps": steps,
        "SoC": soc,
        "makespan": makespan,
        "blocked_move_commands": blocked,
        "runtime_seconds": round(runtime, 6),
        "obs_radius": 5,
        "max_episode_steps": 128,
        "on_target": "nothing",
        "collision_system": "soft",
        "algorithm": "BatchAStarAgent",
        "pogema_version": pogema.__version__,
        "python_version": sys.version.split()[0],
    }


def main():
    print("Experiment 03: corridor width")
    print("POGEMA:", pogema.__version__)

    rows = []

    for name, starts, goals in SCENARIOS:
        for width in [1, 2]:
            rows.append(
                run_episode(name, starts, goals, width)
            )

    timestamp = datetime.now().strftime(
        "exp03_%Y%m%d_%H%M%S_%f"
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

    print("\nEpisodes completed:", len(rows))
    print("Results saved:", output)


if __name__ == "__main__":
    main()
