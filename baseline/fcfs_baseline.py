"""
baseline/fcfs_baseline.py  --  Person 1: the "before" system

Models how public chargers work today:
  * whoever physically ARRIVES first gets the next free bay (a single queue)
  * urgency, battery level and reputation are ignored
  * no grid awareness: every plugged-in car draws its full requested power,
    even if the total exceeds P_max (we COUNT that as an overload event)

It reuses the same EVAgent class so both systems run on identical traffic.
The simulator moves agents (agent.move()); this class only handles the
queue and the charging.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from agents.ev_agent import EVAgent, Status


class FCFSStation:
    def __init__(self, station_id: int, n_bays: int, p_max_kw: float):
        self.station_id = station_id
        self.n_bays = n_bays
        self.p_max_kw = p_max_kw
        self.queue: List[EVAgent] = []      # waiting, in order of arrival
        self.charging: List[EVAgent] = []   # currently plugged in
        self.overload_events = 0            # ticks where demand > P_max
        self.overload_kw_total = 0.0        # cumulative excess kW

    def step(self, tick: int, agents: Sequence[EVAgent]) -> Dict:
        # 1. Detect new physical arrivals and join the back of the queue.
        newly_arrived = []
        for a in agents:
            if (a.station_id == self.station_id
                    and a.arrival_tick is None
                    and a.status in (Status.DRIVING, Status.SEEKING)
                    and a.at_station()):
                a.mark_arrival(tick)
                a.status = Status.SEEKING
                newly_arrived.append(a)
        # Same-tick arrivals: lower id first, so runs are reproducible.
        newly_arrived.sort(key=lambda a: a.agent_id)
        self.queue.extend(newly_arrived)

        # 2. Fill free bays strictly from the front of the queue.
        while len(self.charging) < self.n_bays and self.queue:
            a = self.queue.pop(0)
            a.begin_charging(tick, verify=False)   # nobody checks claims in FCFS
            self.charging.append(a)

        # 3. No grid awareness: everybody gets what they ask for.
        demand = sum(a.power_request_kw for a in self.charging)
        overloaded = demand > self.p_max_kw
        if overloaded:
            self.overload_events += 1
            self.overload_kw_total += demand - self.p_max_kw
        for a in self.charging:
            a.charge(a.power_request_kw, tick)
        self.charging = [a for a in self.charging if a.status == Status.CHARGING]

        return {
            "tick": tick,
            "queue_len": len(self.queue),
            "n_charging": len(self.charging),
            "demand_kw": demand,
            "p_max_kw": self.p_max_kw,
            "overloaded": overloaded,
        }