"""
Genetic Algorithm for optimizing NCAA tournament bracket predictions.

Uses a 67-bit chromosome representing all game outcomes in a 68-team
single-elimination tournament. The objective function maximizes expected
bracket score using precomputed win probabilities (log5 model).

Usage:
    python genetic_algorithm.py [--teams-path PATH] [--seed SEED] [--generations N]
"""

import csv
import os
import random
import numpy as np
import mlx.core as mx
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROUND_NAMES = [
    'First Four', 'Round of 64', 'Round of 32',
    'Sweet 16', 'Elite 8', 'Final Four', 'Championship',
]


# Configuration

@dataclass
class GAConfig:
    teams_path: str = "2026/teams_2026.csv"
    population_size: int = 100000
    eval_chunk: int = 100000            # max individuals per GPU eval batch
    generations: int = 200
    crossover_rate: float = 0.5
    mutation_rate: float = None         # defaults to 1/67
    tournament_size: int = 10
    noise: float = 0.1                  # std dev of win-rate perturbation
    min_prob: float = 0.1               # clamp win rates to [min_prob, 1 - min_prob]
    seed: int = None
    draw: bool = False                  # skip GA, load bit_freq.csv and draw

    def __post_init__(self):
        if self.mutation_rate is None:
            self.mutation_rate = 1.0 / 67


# Bracket Scorer

class BracketScorer:
    """
    Builds a tournament bracket from a teams CSV, precomputes P(team wins game)
    for every game using the log5 model, then scores any 67-bit chromosome
    in O(67) time.

    Chromosome layout (67 bits):
        [0:4]   First Four play-in games
        [4:36]  Round of 64  (8 games x 4 regions)
        [36:52] Round of 32  (4 games x 4 regions)
        [52:60] Sweet 16     (2 games x 4 regions)
        [60:64] Elite 8      (1 game  x 4 regions)
        [64:66] Final Four   (2 games)
        [66]    Championship  (1 game)

    Each bit: 0 = top/left team advances, 1 = bottom/right team advances.
    """

    SEEDS = [(1, 16), (8, 9), (5, 12), (4, 13),
             (6, 11), (3, 14), (7, 10), (2, 15)]
    REGIONS = ['south', 'east', 'west', 'midwest']

    def __init__(self, teams_csv, round_points=None,
                 noise=0.0, noise_seed=None, min_prob=0.01):
        self.team_winrate = {}
        self.region_teams = {}  # (region, seed) -> [team_name, ...]
        self.ff_pairs = []
        self.min_prob = min_prob
        self._load(teams_csv)

        self.pts = round_points or [0, 1, 2, 4, 8, 16, 32]
        self.noise = noise
        self.noise_rng = np.random.default_rng(noise_seed)

        # Game arrays — each game has a left input, right input, round index,
        # and a precomputed probability distribution over actual winners.
        self.left = [None] * 67    # str (leaf team) or int (prior game index)
        self.right = [None] * 67
        self.rnd = [0] * 67
        self.dist = [None] * 67   # {team: P(team wins this game)}
        self.n = 67

        self._build()
        self._precompute()  # base (no-noise) distributions, also used by decode/print
        self._build_dense()

    # Data loading

    def _load(self, path):
        opp = {}  # region -> opp_region
        with open(path) as f:
            for row in csv.DictReader(f):
                t = row['team']
                self.team_winrate[t] = float(row['winrate'])
                self.region_teams.setdefault(
                    (row['region'], int(row['seed'])), []
                ).append(t)
                opp[row['region']] = row['opp_region']
        # Derive ff_pairs: deduplicate by sorting each pair
        seen = set()
        for reg, opp_reg in opp.items():
            pair = (reg, opp_reg)
            key = tuple(sorted(pair))
            if key not in seen:
                seen.add(key)
                self.ff_pairs.append(pair)

    def _pw(self, a, b, perturb=None):
        """P(a beats b) using the log5 formula."""
        wa, wb = self.team_winrate[a], self.team_winrate[b]
        if perturb is not None:
            wa += perturb.get(a, 0.0)
            wb += perturb.get(b, 0.0)
        wa = max(self.min_prob, min(1 - self.min_prob, wa))
        wb = max(self.min_prob, min(1 - self.min_prob, wb))
        n = wa * (1 - wb)
        return n / (n + wb * (1 - wa))

    # Bracket construction

    def _build(self):
        gi = 0
        ff = {}  # (region, seed) -> game index for First Four games

        # First Four: (region, seed) pairs with exactly 2 teams
        for reg in self.REGIONS:
            for seed in (16, 11):
                ts = self.region_teams.get((reg, seed), [])
                if len(ts) == 2:
                    self.left[gi], self.right[gi], self.rnd[gi] = ts[0], ts[1], 0
                    ff[(reg, seed)] = gi
                    gi += 1
        assert gi == 4, f"Expected 4 First Four games, found {gi}"

        def src(reg, seed):
            """Return game index (int) if First Four, else team name (str)."""
            if (reg, seed) in ff:
                return ff[(reg, seed)]
            return self.region_teams[(reg, seed)][0]

        # Round of 64 (8 games per region)
        r1 = {}
        for reg in self.REGIONS:
            r1[reg] = []
            for hi, lo in self.SEEDS:
                self.left[gi], self.right[gi], self.rnd[gi] = src(reg, hi), src(reg, lo), 1
                r1[reg].append(gi); gi += 1

        # Round of 32
        r2 = {}
        for reg in self.REGIONS:
            r2[reg] = []
            for g in range(4):
                self.left[gi], self.right[gi], self.rnd[gi] = r1[reg][2*g], r1[reg][2*g+1], 2
                r2[reg].append(gi); gi += 1

        # Sweet 16
        s16 = {}
        for reg in self.REGIONS:
            s16[reg] = []
            for g in range(2):
                self.left[gi], self.right[gi], self.rnd[gi] = r2[reg][2*g], r2[reg][2*g+1], 3
                s16[reg].append(gi); gi += 1

        # Elite 8
        e8 = {}
        for reg in self.REGIONS:
            self.left[gi], self.right[gi], self.rnd[gi] = s16[reg][0], s16[reg][1], 4
            e8[reg] = gi; gi += 1

        # Final Four
        f4 = []
        for ra, rb in self.ff_pairs:
            self.left[gi], self.right[gi], self.rnd[gi] = e8[ra], e8[rb], 5
            f4.append(gi); gi += 1

        # Championship
        self.left[gi], self.right[gi], self.rnd[gi] = f4[0], f4[1], 6
        gi += 1
        self.n = gi

    # Precomputation

    def _precompute(self):
        """
        For each game, compute {team: P(team wins this game)}.
        Processes games in order so each game's inputs are already computed.
        """
        for gi in range(self.n):
            ld = self._get_dist(self.left[gi])
            rd = self._get_dist(self.right[gi])
            d = {}
            for a, pa in ld.items():
                for b, pb in rd.items():
                    w = pa * pb
                    if w < 1e-15:
                        continue
                    p = self._pw(a, b)
                    d[a] = d.get(a, 0.0) + w * p
                    d[b] = d.get(b, 0.0) + w * (1 - p)
            self.dist[gi] = d

    def _get_dist(self, src):
        """Probability distribution for a game input (leaf or prior game)."""
        return {src: 1.0} if isinstance(src, str) else self.dist[src]

    # Dense arrays for GPU batch evaluation

    def _build_dense(self):
        """Build dense arrays for GPU-accelerated batch evaluation."""
        teams = sorted(self.team_winrate.keys())
        self.team_to_idx = {t: i for i, t in enumerate(teams)}
        self.idx_to_team = teams
        self.num_teams = len(teams)
        self.winrates_arr = mx.array(
            [self.team_winrate[t] for t in teams], dtype=mx.float32)

        self._left_src = []
        self._right_src = []
        self._is_leaf_left = []
        self._is_leaf_right = []
        self._round_pts_list = []

        for gi in range(self.n):
            L, R = self.left[gi], self.right[gi]
            self._is_leaf_left.append(isinstance(L, str))
            self._is_leaf_right.append(isinstance(R, str))
            self._left_src.append(
                self.team_to_idx[L] if isinstance(L, str) else L)
            self._right_src.append(
                self.team_to_idx[R] if isinstance(R, str) else R)
            self._round_pts_list.append(float(self.pts[self.rnd[gi]]))

        # Convert precomputed dists to dense arrays
        self._dist_dense = []
        for gi in range(self.n):
            d = np.zeros(self.num_teams)
            for team, prob in self.dist[gi].items():
                d[self.team_to_idx[team]] = prob
            self._dist_dense.append(mx.array(d, dtype=mx.float32))

    def _compute_dist_batch(self, pw):
        """Compute game distributions for a batch of perturbed pw matrices.

        pw: (batch, num_teams, num_teams).
        Returns list of 67 arrays, each (batch, num_teams).
        """
        batch_size = pw.shape[0]
        dist = [None] * self.n
        eye = mx.eye(self.num_teams)
        one_minus_pw_t = 1 - mx.swapaxes(pw, -2, -1)

        for gi in range(self.n):
            if self._is_leaf_left[gi]:
                ld = mx.broadcast_to(
                    eye[self._left_src[gi]], (batch_size, self.num_teams))
            else:
                ld = dist[self._left_src[gi]]

            if self._is_leaf_right[gi]:
                rd = mx.broadcast_to(
                    eye[self._right_src[gi]], (batch_size, self.num_teams))
            else:
                rd = dist[self._right_src[gi]]

            pr = (pw @ rd[:, :, None])[:, :, 0]
            pl = (one_minus_pw_t @ ld[:, :, None])[:, :, 0]
            dist[gi] = ld * pr + rd * pl

        return dist

    def batch_objective(self, population, eval_chunk=100_000):
        """Evaluate entire population on GPU.

        population: mx.array (pop_size, 67) of int.
        eval_chunk: max individuals per GPU batch (noisy path only).
        Returns mx.array (pop_size,) of float scores.
        """
        pop_size = population.shape[0]

        if self.noise > 0 and pop_size > eval_chunk:
            parts = []
            for i in range(0, pop_size, eval_chunk):
                chunk = self._batch_objective_core(
                    population[i:i + eval_chunk])
                mx.eval(chunk)
                parts.append(chunk)
            return mx.concatenate(parts)

        return self._batch_objective_core(population)

    def _batch_objective_core(self, population):
        """Evaluate a single chunk on GPU."""
        pop_size = population.shape[0]

        if self.noise > 0:
            perturbs = (mx.random.normal(shape=(pop_size, self.num_teams))
                        * self.noise)
            wr = mx.clip(self.winrates_arr + perturbs, self.min_prob, 1 - self.min_prob)
            wa = wr[:, :, None]
            wb = wr[:, None, :]
            num = wa * (1 - wb)
            pw = num / (num + wb * (1 - wa))
            dist = self._compute_dist_batch(pw)
            batched = True
        else:
            dist = self._dist_dense
            batched = False

        picks = [None] * self.n
        scores = mx.zeros((pop_size,))
        arange = mx.arange(pop_size)

        for gi in range(self.n):
            if self._is_leaf_left[gi]:
                lt = mx.full((pop_size,), self._left_src[gi], dtype=mx.int32)
            else:
                lt = picks[self._left_src[gi]]

            if self._is_leaf_right[gi]:
                rt = mx.full((pop_size,), self._right_src[gi], dtype=mx.int32)
            else:
                rt = picks[self._right_src[gi]]

            picked = mx.where(population[:, gi] == 0, lt, rt)
            picks[gi] = picked

            d = dist[gi]
            if not batched:
                probs = d[picked]
            else:
                flat_idx = arange * self.num_teams + picked
                probs = d.reshape(-1)[flat_idx]

            scores = scores + self._round_pts_list[gi] * probs

        return scores

    # Decode / display

    def decode(self, x):
        """Return list of dicts describing each game pick."""
        picks = [None] * self.n
        rows = []
        for gi in range(self.n):
            L, R = self.left[gi], self.right[gi]
            lt = L if isinstance(L, str) else picks[L]
            rt = R if isinstance(R, str) else picks[R]
            picks[gi] = lt if x[gi] == 0 else rt
            p = self.dist[gi].get(picks[gi], 0.0)
            rows.append(dict(
                game=gi, round=ROUND_NAMES[self.rnd[gi]],
                top=lt, bottom=rt, pick=picks[gi],
                prob=p, exp_pts=self.pts[self.rnd[gi]] * p,
            ))
        return rows

    def print_bracket(self, x):
        """Pretty-print the decoded bracket."""
        rows = self.decode(x)
        cur = None
        total = 0.0
        for r in rows:
            if r['round'] != cur:
                cur = r['round']
                print(f"\n--- {cur} ---")
            fav = '*' if r['prob'] > 0.5 else ' '
            print(f"  {fav} {r['top']:>20s} vs {r['bottom']:<20s} "
                  f"-> {r['pick']:<20s} P={r['prob']:.3f}  E={r['exp_pts']:.2f}")
            total += r['exp_pts']
        print(f"\n{'=' * 70}")
        print(f"  Total expected score: {total:.3f}")
        print(f"{'=' * 70}\n")


# Core GA

class GeneticAlgorithm:
    def __init__(self, config: GAConfig, scorer: BracketScorer):
        self.config = config
        self.scorer = scorer
        if self.config.seed is not None:
            mx.random.seed(self.config.seed)

        # State
        self.population = None
        self.fitness = None
        self.history: list[dict] = []

    def _init_population(self):
        return mx.random.randint(0, 2, shape=(self.config.population_size, 67))

    def _evaluate(self, population):
        scores = self.scorer.batch_objective(
            population, eval_chunk=self.config.eval_chunk)
        mx.eval(scores)
        return scores

    def _tournament_select_batch(self):
        """Select pop_size individuals via batched tournament selection."""
        cfg = self.config
        indices = mx.random.randint(
            0, cfg.population_size,
            shape=(cfg.population_size, cfg.tournament_size))
        flat_fits = self.fitness[indices.reshape(-1)].reshape(
            cfg.population_size, cfg.tournament_size)
        best_local = mx.argmax(flat_fits, axis=1)
        flat_idx = (mx.arange(cfg.population_size) * cfg.tournament_size
                    + best_local)
        best_idx = indices.reshape(-1)[flat_idx]
        return self.population[best_idx]

    def run(self, verbose: bool = True) -> dict:
        cfg = self.config

        # Initialize
        self.population = self._init_population()
        self.fitness = self._evaluate(self.population)

        for gen in range(1, cfg.generations + 1):
            # Selection
            parents = self._tournament_select_batch()

            # Crossover (uniform, applied per-pair)
            p1 = parents[0::2]
            p2 = parents[1::2]
            n_pairs = min(p1.shape[0], p2.shape[0])
            p1m, p2m = p1[:n_pairs], p2[:n_pairs]
            do_cx = mx.random.uniform(
                shape=(n_pairs, 1)) < cfg.crossover_rate
            bit_mask = mx.random.uniform(shape=(n_pairs, 67)) < 0.5
            c1 = mx.where(do_cx, mx.where(bit_mask, p1m, p2m), p1m)
            c2 = mx.where(do_cx, mx.where(bit_mask, p2m, p1m), p2m)
            parts = [c1, c2]
            if p1.shape[0] > n_pairs:
                parts.append(p1[n_pairs:])
            children = mx.concatenate(parts, axis=0)[:cfg.population_size]

            # Mutation
            flip = mx.random.uniform(
                shape=children.shape) < cfg.mutation_rate
            self.population = mx.where(flip, 1 - children, children)
            mx.eval(self.population)

            # Evaluate
            self.fitness = self._evaluate(self.population)

            # Logging
            bit_freq_mx = mx.mean(
                self.population.astype(mx.float32), axis=0)
            mx.eval(bit_freq_mx)
            bit_freq = np.array(bit_freq_mx)
            p = np.clip(bit_freq, 1e-10, 1 - 1e-10)
            entropy = float(np.mean(
                -p * np.log2(p) - (1 - p) * np.log2(1 - p)))
            fit_np = np.array(self.fitness)
            fit_mean = float(np.mean(fit_np))
            fit_std = float(np.std(fit_np))
            stats = {
                "generation": gen,
                "entropy": entropy,
                "fit_mean": fit_mean,
                "fit_std": fit_std,
                "bit_freq": bit_freq.copy(),
            }
            self.history.append(stats)

            if verbose:
                print(f"Gen {gen:>5d}  |  entropy={entropy:.4f}  "
                      f"mean={fit_mean:.4f}  std={fit_std:.4f}")

        bit_freq = np.array(
            mx.mean(self.population.astype(mx.float32), axis=0))

        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimization complete after {cfg.generations} generations")
            print(f"{'='*60}")

        return {
            "history": self.history,
            "bit_freq": bit_freq,
        }



# Run it

if __name__ == "__main__":
    import argparse
    from dataclasses import fields

    parser = argparse.ArgumentParser(description="Optimize NCAA bracket via GA")
    defaults = GAConfig()
    for f in fields(defaults):
        val = getattr(defaults, f.name)
        flag = f'--{f.name.replace("_", "-")}'
        if isinstance(val, bool):
            def parse_bool(v):
                if v.lower() == 'true': return True
                if v.lower() == 'false': return False
                raise argparse.ArgumentTypeError(f"expected true/false, got '{v}'")
            parser.add_argument(flag, type=parse_bool, default=val,
                                metavar='{true,false}')
        else:
            parser.add_argument(flag,
                                type=type(val) if val is not None else float,
                                default=val)
    args = parser.parse_args()

    config_kwargs = {f.name: getattr(args, f.name.replace('-', '_')) for f in fields(defaults)}
    config = GAConfig(**config_kwargs)

    print(f"Loading bracket from {config.teams_path}")
    scorer = BracketScorer(config.teams_path, noise=config.noise, min_prob=config.min_prob)
    freq_path = "bit_freq.csv"

    if config.draw:
        # Load bit frequencies from CSV and enter draw mode
        bit_freq = np.zeros(67)
        with open(freq_path) as f:
            for row in csv.DictReader(f):
                bit_freq[int(row['game'])] = float(row['p_bottom'])
        print(f"Loaded bit frequencies from {freq_path}")
    else:
        # Run GA
        ga = GeneticAlgorithm(config=config, scorer=scorer)
        result = ga.run(verbose=True)
        bit_freq = result['bit_freq']

        # Save bit frequencies to CSV
        with open(freq_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['game', 'round', 'p_bottom'])
            for gi in range(scorer.n):
                writer.writerow([gi, ROUND_NAMES[scorer.rnd[gi]],
                                 f"{bit_freq[gi]:.4f}"])
        print(f"Bit frequencies saved to {freq_path}")

        # Optional: plot convergence and heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gens = [h["generation"] for h in result["history"]]
            entropies = [h["entropy"] for h in result["history"]]

            plt.figure(figsize=(10, 5))
            plt.plot(gens, entropies, linewidth=2)
            plt.xlabel("Generation")
            plt.ylabel("Mean Bit Entropy (bits)")
            plt.title("GA Convergence")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("convergence.png", dpi=150)
            print("Convergence plot saved to convergence.png")

            freq_matrix = np.array([h["bit_freq"] for h in result["history"]])
            gen_labels = [h["generation"] for h in result["history"]]

            plt.figure(figsize=(14, 6))
            plt.imshow(freq_matrix.T, aspect='auto', cmap='RdBu_r',
                       vmin=0, vmax=1, origin='lower',
                       extent=[gen_labels[0], gen_labels[-1], 0, 67])
            plt.colorbar(label="P(bit=1)")
            plt.xlabel("Generation")
            plt.ylabel("Game index")
            plt.title("Bit Frequency Heatmap (red=1/bottom team, blue=0/top team)")
            for boundary, label in [(4, "R64"), (36, "R32"), (52, "S16"),
                                    (60, "E8"), (64, "F4"), (66, "Ch")]:
                plt.axhline(y=boundary, color='black', linewidth=0.5, linestyle='--')
                plt.text(gen_labels[0], boundary + 0.5, f" {label}",
                         fontsize=7, va='bottom')
            plt.tight_layout()
            plt.savefig("bit_freq.png", dpi=150)
            print("Bit frequency heatmap saved to bit_freq.png")
        except ImportError:
            pass

    # Interactive bracket drawing
    rng = np.random.default_rng()
    draw_num = 0
    print("\nPress Enter to draw a bracket, or Ctrl-C to quit.\n")
    try:
        while True:
            input()
            draw_num += 1
            sample = (rng.random(67) < bit_freq).astype(int)
            decoded = scorer.decode(sample)
            score = sum(r['exp_pts'] for r in decoded)
            print(f"--- Draw #{draw_num} (expected score: {score:.3f}) ---")
            scorer.print_bracket(sample)
    except (KeyboardInterrupt, EOFError):
        print("\nDone.")
