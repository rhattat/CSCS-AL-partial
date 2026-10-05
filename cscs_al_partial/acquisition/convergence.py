"""
Convergence Checker
====================
Stops the active-learning loop when the round-to-round validation Dice
change falls below a threshold for a set number of consecutive rounds, or
when the residual pool is exhausted, or a hard round cap is reached.

Note: the plateau check compares the current round to the immediately
preceding round (not to the running best), and uses the absolute value of
the change -- a small regression counts as a plateau too, not only
insufficient improvement.

Deployment without a held-out validation set
----------------------------------------------
`update()` above needs a validation Dice, which needs pre-existing
ground-truth labels -- unavailable when CSCS-AL is applied to a genuinely
unlabeled dataset from scratch. `update_proxy()` provides two label-free
alternatives, selected via `mode` at construction:

  - mode="stability": stop when the model's own hard segmentations on the
    residual pool stop changing round-to-round (prediction agreement, as a
    Dice-like overlap between round r-1 and round r). See
    `compute_prediction_agreement` in iterative_scoring.py.
  - mode="entropy": stop when the mean segmentation uncertainty (U_seg,
    already computed each round for the acquisition score) stops moving,
    i.e. further annotation is no longer changing what the model is unsure
    about.

Both are standard proxies in label-free active-learning deployments and
reuse signals CSCS-AL already computes for acquisition -- no extra
inference pass is required. Pool-exhaustion and max-rounds checks are
identical to the supervised path.
"""

from typing import List, Optional


class ConvergenceChecker:

    def __init__(self, delta_pp: float = 0.5, patience: int = 2, max_rounds: int = 9,
                 mode: str = "supervised", stability_delta_pp: float = 0.5,
                 entropy_delta_pct: float = 2.0):
        """
        Args:
            delta_pp            : (supervised mode) Dice plateau threshold, in points
            patience            : consecutive plateau rounds required to stop, all modes
            max_rounds          : hard round cap, all modes
            mode                : "supervised" (default, needs update()) |
                                   "stability" | "entropy" (need update_proxy())
            stability_delta_pp  : (mode="stability") prediction agreement is
                                   considered a plateau round when it is >=
                                   (100 - stability_delta_pp) percent
            entropy_delta_pct   : (mode="entropy") relative change in mean
                                   pool uncertainty, in percent, below which
                                   a round counts as a plateau round
        """
        self.delta_pp = delta_pp
        self.patience = patience
        self.max_rounds = max_rounds
        self.mode = mode
        self.stability_delta_pp = stability_delta_pp
        self.entropy_delta_pct = entropy_delta_pct
        self._history: List[float] = []
        self._proxy_history: List[float] = []
        self._proxy_calls = 0
        self._plateau_count = 0

    def update(self, dice_pct: float, pool_remaining: int = 999, round_budget: int = 1) -> bool:
        """Record Dice (as a percentage) and return True if the loop should stop."""
        self._history.append(dice_pct)
        n = len(self._history)

        if n > self.max_rounds:
            self._reason = f"Max rounds ({self.max_rounds})"
            return True

        if pool_remaining < round_budget:
            self._reason = f"Pool exhausted ({pool_remaining} < {round_budget})"
            return True

        if n >= 2:
            delta = abs(self._history[-1] - self._history[-2])
            if delta < self.delta_pp:
                self._plateau_count += 1
            else:
                self._plateau_count = 0
            if self._plateau_count >= self.patience:
                self._reason = f"Plateau: |Delta Dice| < {self.delta_pp}pp for {self.patience} rounds"
                return True

        return False

    def update_proxy(
        self,
        pool_remaining: int = 999,
        round_budget: int = 1,
        prediction_agreement: Optional[float] = None,
        pool_uncertainty: Optional[float] = None,
    ) -> bool:
        """
        Label-free counterpart to update(). Requires mode="stability" or
        mode="entropy" (set at construction). Pass prediction_agreement (0-100
        Dice-like scale) for "stability", or pool_uncertainty (mean U_seg
        this round) for "entropy" -- whichever matches self.mode.

        A round with no signal yet (e.g. round 0, before any task-model
        predictions exist) is a no-op for the plateau check: pass
        signal=None and only the pool-exhaustion/max-rounds checks apply.

        Returns True if the loop should stop.
        """
        if self.mode not in ("stability", "entropy"):
            raise ValueError(
                f"update_proxy() requires mode='stability' or 'entropy', got mode={self.mode!r} "
                f"(construct ConvergenceChecker(mode=...) accordingly, or call update() instead "
                f"for the default supervised/ground-truth mode)"
            )

        self._proxy_calls += 1
        if self._proxy_calls > self.max_rounds:
            self._reason = f"Max rounds ({self.max_rounds})"
            return True

        if pool_remaining < round_budget:
            self._reason = f"Pool exhausted ({pool_remaining} < {round_budget})"
            return True

        signal = prediction_agreement if self.mode == "stability" else pool_uncertainty
        if signal is None:
            return False

        self._proxy_history.append(signal)
        n = len(self._proxy_history)

        if self.mode == "stability":
            is_plateau_round = signal >= (100.0 - self.stability_delta_pp)
            reason = (f"Plateau: prediction agreement >= "
                      f"{100.0 - self.stability_delta_pp:.2f}% for {self.patience} rounds")
        else:
            if n < 2:
                return False
            prev = self._proxy_history[-2]
            rel_change = abs(signal - prev) / max(abs(prev), 1e-8) * 100.0
            is_plateau_round = rel_change < self.entropy_delta_pct
            reason = (f"Plateau: pool uncertainty change < "
                      f"{self.entropy_delta_pct}% for {self.patience} rounds")

        if is_plateau_round:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        if self._plateau_count >= self.patience:
            self._reason = reason
            return True

        return False

    @property
    def reason(self) -> str:
        return getattr(self, '_reason', 'Not converged')

    @property
    def history(self) -> List[float]:
        return self._history.copy()

    @property
    def proxy_history(self) -> List[float]:
        return self._proxy_history.copy()
