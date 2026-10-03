"""
agents/ev_agent.py  --  Person 1: Agent & Negotiation Logic

Each EVAgent is an independent decision-maker. It:
  1. senses its own state (battery, distance, reputation)
  2. computes its OWN urgency score
  3. broadcasts that score to nearby agents
  4. observes the broadcasts it can hear
  5. ranks itself and decides for ITSELF whether it gets a bay

Nothing in this file computes a global ranking and hands it to anyone.

Time model: 1 tick = 1 minute.  Distance unit: km.  Energy: kWh / kW.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Tunable constants (all in one place so Person 2/3 can sweep them)
# --------------------------------------------------------------------------
W1, W2, W3 = 0.5, 0.3, 0.2        # urgency weights: battery, proximity, reputation

REP_START = 0.5                   # neutral starting reputation
NO_SHOW_PENALTY = 0.20            # reserved a bay, never arrived
LIE_PENALTY = 0.35                # caught misreporting battery at the plug
COMPLETION_REWARD = 0.05          # finished a session as negotiated
REP_MIN, REP_MAX = 0.0, 1.0       # reputation is clamped to this range

COMM_RANGE_KM = 5.0               # how far a broadcast is heard
ARRIVE_RADIUS_KM = 0.05           # "physically at the station"
RESERVATION_GRACE_TICKS = 3       # extra minutes allowed beyond the ETA
TARGET_SOC = 0.80                 # fast chargers stop at 80 %
LIE_TOLERANCE = 0.10              # claimed vs. real battery mismatch allowed
SELFISH_CLAIM = 0.02              # what a selfish agent pretends its battery is


class Status(Enum):
    DRIVING = "driving"      # outside comm range, not yet contending
    SEEKING = "seeking"      # in range, broadcasting, wants a bay
    ASSIGNED = "assigned"    # holds a reservation (may still be en route)
    CHARGING = "charging"    # plugged in
    DONE = "done"            # session finished
    NO_SHOW = "no_show"      # had a reservation, never showed
    DIVERTED = "diverted"    # gave up waiting, went to another station


@dataclass(frozen=True)
class Broadcast:
    """The only thing agents ever say to each other."""
    sender_id: int
    station_id: int
    urgency: float
    eta: float                       # ticks until arrival
    position: Tuple[float, float]    # so receivers can check range


class EVAgent:
    def __init__(
        self,
        agent_id: int,
        battery_level: float,                    # 0.0 - 1.0 (true value)
        position: Tuple[float, float],
        station_id: int,
        station_position: Tuple[float, float],
        battery_capacity_kwh: float = 60.0,
        power_request_kw: float = 50.0,
        speed_km_per_tick: float = 0.8,          # ~48 km/h
        reputation: float = REP_START,
        selfish: bool = False,
        will_no_show: bool = False,
        comm_range_km: float = COMM_RANGE_KM,
        patience_ticks: int = 30,
        weights: Tuple[float, float, float] = (W1, W2, W3),
    ):
        self.agent_id = agent_id
        self.battery_level = battery_level
        self.position = position
        self.station_id = station_id
        self.station_position = station_position
        self.battery_capacity_kwh = battery_capacity_kwh
        self.power_request_kw = power_request_kw
        self.speed = speed_km_per_tick
        self.reputation = reputation
        self.selfish = selfish
        self.will_no_show = will_no_show
        self.comm_range_km = comm_range_km
        self.patience_ticks = patience_ticks
        self.w1, self.w2, self.w3 = weights

        self.status = Status.DRIVING
        self.inbox: List[Broadcast] = []

        # bookkeeping that Person 3's logger reads
        self.seek_start_tick: Optional[int] = None
        self.arrival_tick: Optional[int] = None
        self.charge_start_tick: Optional[int] = None
        self.finish_tick: Optional[int] = None
        self.reservation_deadline: Optional[int] = None
        self.energy_received_kwh = 0.0
        self.caught_lying = False

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------
    def distance_to_station(self) -> float:
        return math.dist(self.position, self.station_position)

    def eta_ticks(self) -> float:
        return self.distance_to_station() / self.speed

    def at_station(self) -> bool:
        return self.distance_to_station() <= ARRIVE_RADIUS_KM

    @property
    def claimed_battery(self) -> float:
        """What this agent TELLS others. Honest agents tell the truth;
        selfish agents understate their battery to look more urgent."""
        if self.selfish:
            return min(self.battery_level, SELFISH_CLAIM)
        return self.battery_level

    # ------------------------------------------------------------------
    # Step 1: urgency score (computed by the agent itself, every tick)
    # ------------------------------------------------------------------
    def urgency_score(self) -> float:
        battery_term = 1.0 - self.claimed_battery
        # Plan says 1/distance, which explodes to infinity at distance 0.
        # 1/(1+d) is bounded in (0, 1] and keeps the same "closer = higher" shape.
        proximity_term = 1.0 / (1.0 + self.distance_to_station())
        return (self.w1 * battery_term
                + self.w2 * proximity_term
                + self.w3 * self.reputation)

    # ------------------------------------------------------------------
    # State transitions driven by what the agent senses
    # ------------------------------------------------------------------
    def update_state(self, tick: int) -> None:
        """DRIVING -> SEEKING once the station is within communication range."""
        if (self.status == Status.DRIVING
                and self.distance_to_station() <= self.comm_range_km):
            self.status = Status.SEEKING
            self.seek_start_tick = tick

    def move(self) -> None:
        """Drive one tick straight toward the station."""
        if self.status == Status.ASSIGNED and self.will_no_show:
            return  # this driver reserved a bay then wandered off
        if self.status in (Status.CHARGING, Status.DONE,
                           Status.NO_SHOW, Status.DIVERTED):
            return
        d = self.distance_to_station()
        if d <= ARRIVE_RADIUS_KM:
            return
        step = min(self.speed, d)
        dx = (self.station_position[0] - self.position[0]) / d
        dy = (self.station_position[1] - self.position[1]) / d
        self.position = (self.position[0] + dx * step,
                         self.position[1] + dy * step)

    def mark_arrival(self, tick: int) -> None:
        if self.arrival_tick is None and self.at_station():
            self.arrival_tick = tick

    # ------------------------------------------------------------------
    # Steps 2-3: broadcast, observe, self-rank, self-assign
    # ------------------------------------------------------------------
    def broadcast(self) -> Optional[Broadcast]:
        """Only agents still contending for a bay speak."""
        if self.status != Status.SEEKING:
            return None
        return Broadcast(self.agent_id, self.station_id,
                         self.urgency_score(), self.eta_ticks(), self.position)

    def observe(self, messages: Sequence[Broadcast]) -> None:
        """Keep only what this agent could actually HEAR:
        not its own, same station, sender within MY comm range."""
        self.inbox = [
            m for m in messages
            if m.sender_id != self.agent_id
            and m.station_id == self.station_id
            and math.dist(m.position, self.position) <= self.comm_range_km
        ]

    def decide_slot(self, free_bays: int, tick: int) -> bool:
        """Rank myself among everyone I heard; take a bay if I'm in the top N.
        Returns True if I just self-assigned."""
        if self.status != Status.SEEKING or free_bays <= 0:
            return False
        mine = self.broadcast()
        contenders = self.inbox + [mine]
        # Deterministic tie-break (urgency desc, earlier ETA, lower id) so every
        # agent that hears the same messages computes the SAME ranking.
        contenders.sort(key=lambda m: (-round(m.urgency, 9), m.eta, m.sender_id))
        my_rank = next(i for i, m in enumerate(contenders)
                       if m.sender_id == self.agent_id)
        if my_rank < free_bays:
            self.status = Status.ASSIGNED
            self.reservation_deadline = (
                tick + math.ceil(self.eta_ticks()) + RESERVATION_GRACE_TICKS)
            return True
        return False

    # ------------------------------------------------------------------
    # "Wait, or query a nearby alternate station"
    # ------------------------------------------------------------------
    def should_divert(self, tick: int) -> bool:
        return (self.status == Status.SEEKING
                and self.seek_start_tick is not None
                and tick - self.seek_start_tick > self.patience_ticks)

    def pick_alternate(self, alternates: List[Dict]) -> Optional[int]:
        """alternates: [{'station_id', 'available_bays', 'distance_km'}, ...]
        Choose the closest alternate that has a free bay, else None."""
        options = [a for a in alternates if a["available_bays"] > 0]
        if not options:
            return None
        best = min(options, key=lambda a: a["distance_km"])
        self.status = Status.DIVERTED
        return best["station_id"]

    # ------------------------------------------------------------------
    # Step 5: reputation, no-show handling, charging
    # ------------------------------------------------------------------
    def _adjust_reputation(self, delta: float) -> None:
        self.reputation = max(REP_MIN, min(REP_MAX, self.reputation + delta))

    def check_reservation(self, tick: int) -> bool:
        """Call every tick. If my deadline passed and I'm not there,
        I'm a no-show: lose reputation and free the bay. Returns True if so."""
        if (self.status == Status.ASSIGNED
                and self.reservation_deadline is not None
                and tick > self.reservation_deadline
                and not self.at_station()):
            self.status = Status.NO_SHOW
            self._adjust_reputation(-NO_SHOW_PENALTY)
            return True
        return False

    def begin_charging(self, tick: int, verify: bool = True) -> bool:
        """Plug in. The plug physically reveals the true battery level, which
        is how a lie gets caught without any central authority.
        Returns False if the agent isn't allowed to plug in."""
        if self.status not in (Status.ASSIGNED, Status.SEEKING) or not self.at_station():
            return False
        if verify and abs(self.claimed_battery - self.battery_level) > LIE_TOLERANCE:
            self.caught_lying = True
            self._adjust_reputation(-LIE_PENALTY)
        self.status = Status.CHARGING
        self.charge_start_tick = tick
        return True

    def charge(self, power_kw: float, tick: int, dt_ticks: int = 1) -> float:
        """Receive power for one tick. Returns kWh delivered. Person 2 decides
        power_kw (grid-aware split); the agent just accepts what it is given."""
        if self.status != Status.CHARGING:
            return 0.0
        energy = power_kw * dt_ticks / 60.0
        room = (TARGET_SOC - self.battery_level) * self.battery_capacity_kwh
        energy = max(0.0, min(energy, room))
        self.battery_level += energy / self.battery_capacity_kwh
        self.energy_received_kwh += energy
        if self.battery_level >= TARGET_SOC - 1e-9:
            self.battery_level = TARGET_SOC
            self.status = Status.DONE
            self.finish_tick = tick
            self._adjust_reputation(+COMPLETION_REWARD)
        return energy

    # ------------------------------------------------------------------
    # Metrics helper for Person 3
    # ------------------------------------------------------------------
    @property
    def wait_time(self) -> Optional[int]:
        """Minutes between physically arriving and plugging in."""
        if self.arrival_tick is None or self.charge_start_tick is None:
            return None
        return self.charge_start_tick - self.arrival_tick


# --------------------------------------------------------------------------
# Helper: one full negotiation round (what Person 2's simulator calls per tick)
# --------------------------------------------------------------------------
def negotiation_round(agents: Sequence[EVAgent], station_id: int,
                      free_bays: int, tick: int) -> List[EVAgent]:
    """Phase 1: everyone updates and broadcasts.
    Phase 2: everyone listens, ranks itself and decides independently.
    The two phases are separate so all agents decide on the same snapshot."""
    local = [a for a in agents if a.station_id == station_id]
    for a in local:
        a.update_state(tick)
    messages = [m for m in (a.broadcast() for a in local) if m is not None]
    for a in local:
        a.observe(messages)
    return [a for a in local if a.decide_slot(free_bays, tick)]