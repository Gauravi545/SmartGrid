"""
Run from the project root:   python -m tests.test_agent_logic
Plain asserts, so pytest also works.
"""
import random

from agents.ev_agent import (
    EVAgent, Status, negotiation_round,
    NO_SHOW_PENALTY, LIE_PENALTY, COMPLETION_REWARD,
)
from baseline.fcfs_baseline import FCFSStation

STATION = (0.0, 0.0)


def make(i, battery, dist=2.0, **kw):
    return EVAgent(i, battery, (dist, 0.0), 0, STATION, **kw)


# ---------------------------------------------------------------- urgency
def test_lower_battery_is_more_urgent():
    assert make(1, 0.10).urgency_score() > make(2, 0.80).urgency_score()


# ------------------------------------------------- decentralized assignment
def test_top_n_self_assign():
    agents = [make(i, b) for i, b in enumerate([0.9, 0.1, 0.5, 0.2, 0.7, 0.3])]
    winners = negotiation_round(agents, 0, free_bays=2, tick=0)
    assert {a.agent_id for a in winners} == {1, 3}      # the two lowest batteries


def test_out_of_range_agents_are_not_heard():
    near = make(1, 0.5, dist=1.0)
    far = make(2, 0.05, dist=30.0)                      # outside comm range
    far.update_state(0)
    assert far.status == Status.DRIVING and far.broadcast() is None
    winners = negotiation_round([near, far], 0, free_bays=1, tick=0)
    assert winners == [near]


# ------------------------------------------------------- reputation / no-show
def test_no_show_loses_reputation_and_frees_bay():
    a = make(1, 0.2, will_no_show=True)
    negotiation_round([a], 0, 1, tick=0)
    assert a.status == Status.ASSIGNED
    before = a.reputation
    for t in range(1, 20):
        a.move()
        if a.check_reservation(t):
            break
    assert a.status == Status.NO_SHOW
    assert abs(a.reputation - (before - NO_SHOW_PENALTY)) < 1e-9


def test_completion_reward_is_capped():
    a = make(1, 0.79, dist=0.0, reputation=0.99)
    negotiation_round([a], 0, 1, 0)
    a.begin_charging(0)
    for t in range(1, 10):
        a.charge(50, t)
    assert a.status == Status.DONE and a.reputation == 1.0


# ------------------------------------------------------------ selfish agent
def test_selfish_agent_is_punished_and_loses_priority():
    """A selfish car (real battery 90 %) claims 2 % to jump the queue.
    It wins once, gets caught at the plug, and its reputation drops."""
    honest = make(1, 0.15, dist=0.0)
    cheat = make(2, 0.90, dist=0.0, selfish=True)

    winners = negotiation_round([honest, cheat], 0, free_bays=1, tick=0)
    assert winners == [cheat], "cheating works the FIRST time (nothing to catch it yet)"
    cheat.begin_charging(0)                              # plug reveals the truth
    assert cheat.caught_lying
    assert abs(cheat.reputation - (0.5 - LIE_PENALTY)) < 1e-9

    # Next visit: same lie, but reputation is now low.
    cheat2 = make(3, 0.90, dist=0.0, selfish=True, reputation=cheat.reputation)
    winners = negotiation_round([honest, cheat2], 0, free_bays=1, tick=100)
    assert winners == [honest], "after being caught, the cheat no longer wins"


# ------------------------------------- mini side-by-side (sanity check only)
def build_traffic(seed):
    rng = random.Random(seed)
    out = []
    for i in range(14):
        ang_x = rng.uniform(6, 14)
        out.append(EVAgent(i, rng.uniform(0.05, 0.6), (ang_x, 0.0), 0, STATION,
                           power_request_kw=50))
    return out


def run_fcfs(agents, bays=2, t_max=800):
    st = FCFSStation(0, bays, p_max_kw=80)
    for t in range(t_max):
        for a in agents:
            a.move()
        st.step(t, agents)
    return st


def run_decentralized(agents, bays=2, t_max=800):
    for t in range(t_max):
        for a in agents:
            a.move()
            a.mark_arrival(t)
            a.check_reservation(t)
        busy = sum(a.status in (Status.ASSIGNED, Status.CHARGING) for a in agents)
        negotiation_round(agents, 0, bays - busy, t)
        for a in agents:
            if a.status == Status.ASSIGNED and a.at_station():
                a.begin_charging(t)
            a.charge(50, t)


def summarize(label, agents):
    low = [a.wait_time for a in agents if a.wait_time is not None and a.battery_level and a.energy_received_kwh and a.agent_id in LOW]
    allw = [a.wait_time for a in agents if a.wait_time is not None]
    print(f"{label:14s} served={len(allw):2d}  avg wait={sum(allw)/len(allw):6.1f} min  "
          f"avg wait of LOW-battery cars={sum(low)/max(1,len(low)):6.1f} min")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    base = build_traffic(7)
    LOW = {a.agent_id for a in base if a.battery_level < 0.25}
    f = build_traffic(7); run_fcfs(f)
    d = build_traffic(7); run_decentralized(d)
    print()
    summarize("FCFS", f)
    summarize("Decentralized", d)