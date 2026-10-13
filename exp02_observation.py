import csv
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import pogema
from pogema import BatchAStarAgent, GridConfig, pogema_v0


def make_scenarios():
    # Три карты 10 x 10 с обходными маршрутами.
    scenarios = []

    for name in ["wall", "zigzag", "dead_end"]:
        grid = [[0] * 10 for _ in range(10)]

        if name == "wall":
            for row in range(8):
                grid[row][4] = 1

            start, goal = [1, 1], [1, 8]

        elif name == "zigzag":
            for row in range(7):
                grid[row][3] = 1

            for row in range(3, 10):
                grid[row][6] = 1

            start, goal = [1, 1], [1, 8]

        else:
            for col in range(2, 7):
                grid[2][col] = grid[7][col] = 1

            for row in range(2, 8):
                grid[row][6] = 1

            start, goal = [4, 4], [4, 8]

        scenarios.append((name, grid, start, goal))

    return scenarios


def shortest_path_length(grid, start, goal):
    # Полная карта используется только для оценки результата.
    queue = deque([(tuple(start), 0)])
    visited = {tuple(start)}

    while queue:
        (row, col), distance = queue.popleft()

        if [row, col] == goal:
            return distance

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc

            if 0 <= nr < 10 and 0 <= nc < 10:
                if grid[nr][nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), distance + 1))

    raise RuntimeError("The goal is unreachable")


def run_episode(name, grid, start, goal, radius):
    config = GridConfig(
        map=[row.copy() for row in grid],
        agents_xy=[start.copy()],
        targets_xy=[goal.copy()],
        num_agents=1,
        seed=42,
        obs_radius=radius,
        observation_type="POMAPF",
        collision_system="soft",
        on_target="nothing",
        max_episode_steps=128,
    )

    env = pogema_v0(grid_config=config)

    # В каждом эпизоде создаём агента с новой памятью.
    agent = BatchAStarAgent()
    agent.reset_states()

    blocked = 0
    success = False

    try:
        observations, _ = env.reset()
        started = time.perf_counter()

        for steps in range(1, 129):
            before = tuple(
                env.unwrapped.get_agents_xy(ignore_borders=True)[0]
            )

            actions = agent.act(observations)

            requested_move = (
                tuple(config.MOVES[actions[0]]) != (0, 0)
            )

            observations, _, terminated, truncated, _ = env.step(
                actions.copy()
            )

            after = tuple(
                env.unwrapped.get_agents_xy(ignore_borders=True)[0]
            )

            blocked += int(requested_move and before == after)

            if all(terminated) or all(truncated):
                success = bool(all(terminated))
                break

        runtime = time.perf_counter() - started

    finally:
        env.close()

    optimal = shortest_path_length(grid, start, goal)

    efficiency = optimal / steps if success else 0.0

    print(
        f"{name:8} | radius={radius} | success={int(success)} | "
        f"steps={steps:3} | optimal={optimal:2} | blocked={blocked}"
    )

    return {
        "experiment": "02_observation",
        "scenario": name,
        "obs_radius": radius,
        "seed": 42,
        "num_agents": 1,
        "success": int(success),
        "steps": steps,
        "optimal_steps": optimal,
        "path_efficiency": round(efficiency, 6),
        "blocked_move_commands": blocked,
        "runtime_seconds": round(runtime, 6),
        "max_episode_steps": 128,
        "on_target": "nothing",
        "collision_system": "soft",
        "algorithm": "BatchAStarAgent",
        "pogema_version": pogema.__version__,
        "python_version": sys.version.split()[0],
    }


def main():
    print("Experiment 02: observation radius")
    print("POGEMA:", pogema.__version__)

    rows = []

    for name, grid, start, goal in make_scenarios():
        for radius in [1, 5]:
            rows.append(
                run_episode(name, grid, start, goal, radius)
            )

    timestamp = datetime.now().strftime(
        "exp02_%Y%m%d_%H%M%S_%f"
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
