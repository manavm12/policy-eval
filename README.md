# policy-eval

policy-eval is a **simulation-based framework** for evaluating and comparing
policies in systems with human behavior under uncertainty.

It is designed to answer one question:

> *Given the same world and the same uncertainty, how do different policies compare?*

---

## What this is

policy-eval provides a **generic experiment engine** for running controlled,
stochastic simulations and comparing **policy outcome distributions** across
multiple metrics.

The framework strictly separates:

- **Domains** – define the world and how it evolves  
- **Actors** – define behavior under limited observation  
- **Policies** – define system-level rules, incentives, and constraints  
- **Metrics** – evaluate outcomes *after* a run completes  

This separation is enforced by design.

---

## What this is NOT

-  Not a predictive model of the real world  
-  Not an optimization or policy-recommendation system  
-  Not an agent framework  
-  Not tied to any specific domain, dataset, or application  

policy-eval does **not** claim that optimal policies exist.  
It assumes tradeoffs are inevitable and makes them explicit.

---

## Core ideas

- Policies are compared under **identical randomness**
- Outcomes are evaluated as **distributions**, not single numbers
- Metrics never influence behavior
- Actors never know which policy is active
- Policies never inspect actor internals

The framework is a **tool for thinking and comparison**, not decision automation.

---

## Usage (minimal)

You do **not** inherit from the framework.

You simply implement the required methods and pass your objects to the engine.

```python
from __future__ import annotations
from typing import Any

from policy_eval import RunConfig, evaluate, MetricSpec
from policy_eval.abstractions.actor import Observation
from policy_eval.abstractions.policy import PolicyContext
from policy_eval.core.types import Event, Trajectory

# --- Domain (the world) ---
class CounterDomain:
    name = "counter"

    def initial_state(self, rng: Any) -> int:
        return 0

    def actors(self, state: int):
        return []

    def policy_context(self, state: int, t: int) -> PolicyContext:
        return PolicyContext(t=t, system_view={"value": state})

    def observe(self, state: int, actor, t: int) -> Observation:
        raise RuntimeError("no actors")

    def transition(self, state: int, policy_action: int, actor_actions, rng: Any, t: int) -> int:
        return state + int(policy_action)

    def record(self, state: int, t: int) -> Event:
        return Event(t=t, payload=state)

    def finalize(self, events: list[Event], final_state: int) -> Trajectory:
        return Trajectory(events=tuple(events), final_state=final_state)

# --- Policies (rules) ---
class Inc1:
    name = "inc_1"
    def decide(self, ctx: PolicyContext) -> int:
        return 1

class Inc2:
    name = "inc_2"
    def decide(self, ctx: PolicyContext) -> int:
        return 2

# --- Metric (evaluation) ---
class FinalValue:
    name = "final_value"
    def evaluate(self, traj: Trajectory) -> float:
        return float(traj.final_state)

res = evaluate(
    domain=CounterDomain(),
    policies=[Inc1(), Inc2()],
    metrics=[FinalValue()],
    seeds=range(100),
    cfg=RunConfig(horizon=10),
    pareto_metrics=[MetricSpec(name="final_value", direction="higher_better")],
)

print(res.pareto)
print(res.summaries[("inc_1", "final_value")].mean)
