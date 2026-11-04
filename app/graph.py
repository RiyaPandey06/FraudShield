"""Account-device-merchant graph with delayed-label risk propagation and community detection.

Labels are *never* visible instantly: ``schedule_label`` makes a label usable only once the stream
clock passes ``available_at`` (label maturity, e.g. a chargeback arriving days later). The same code
path is used for training and for live serving, so train/serve behaviour is identical.
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import networkx as nx

MERCHANT_PRIOR_RATE, MERCHANT_PRIOR_WEIGHT = 0.035, 20.0


class GraphService:
    def __init__(self, refresh_every: int = 500, seed: int = 42) -> None:
        self.graph = nx.Graph()                    # account / device / merchant nodes (analyst view)
        self.account_device_graph = nx.Graph()     # account-device edges only (community detection)
        self.refresh_every, self.seed = refresh_every, seed
        self.device_accounts: dict[str, set[str]] = defaultdict(set)
        self.account_devices: dict[str, set[str]] = defaultdict(set)
        self.account_fraud: dict[str, int] = defaultdict(int)
        self.device_fraud: dict[str, int] = defaultdict(int)
        self.merchant_fraud: dict[str, int] = defaultdict(int)
        self.merchant_seen: dict[str, int] = defaultdict(int)
        self.community_of: dict[str, int] = {}
        self.community_members: dict[int, list[str]] = {}
        self.community_fraud: dict[int, int] = defaultdict(int)
        self._pending: list[tuple[float, int, str, str, str, str, int]] = []
        self._seq = 0
        self._events = 0

    # ---- label maturity -------------------------------------------------
    def schedule_label(self, tx: dict, label: int, available_at: float) -> None:
        self._seq += 1
        heapq.heappush(self._pending, (available_at, self._seq, tx["transaction_id"], tx["account_id"],
                                       tx["device_id"], tx["merchant_id"], int(label)))

    def release(self, now: float) -> list[tuple[str, int]]:
        """Reveal every label whose maturity time has passed; returns (transaction_id, label) pairs."""
        released = []
        while self._pending and self._pending[0][0] <= now:
            _, _, tx_id, account, device, merchant, label = heapq.heappop(self._pending)
            released.append((tx_id, label))
            self.merchant_seen[merchant] += 1
            if label:
                first = self.account_fraud[account] == 0
                self.account_fraud[account] += 1
                self.device_fraud[device] += 1
                self.merchant_fraud[merchant] += 1
                cid = self.community_of.get(account)
                if first and cid is not None:
                    self.community_fraud[cid] += 1
        return released

    # ---- online context -------------------------------------------------
    def context(self, tx: dict) -> dict:
        account, device, merchant = tx["account_id"], tx["device_id"], tx["merchant_id"]
        devices = self.account_devices[account] | {device}
        neighbours: set[str] = set()
        for dev in devices:
            neighbours |= self.device_accounts[dev]
        neighbours.discard(account)
        cid = self.community_of.get(account)
        size = len(self.community_members[cid]) if cid is not None else 1
        confirmed = self.community_fraud[cid] if cid is not None else int(self.account_fraud[account] > 0)
        community_risk = confirmed / max(size, 1)
        device_shared = len(self.device_accounts[device] - {account})
        rate = (self.merchant_fraud[merchant] + MERCHANT_PRIOR_RATE * MERCHANT_PRIOR_WEIGHT) / (
            self.merchant_seen[merchant] + MERCHANT_PRIOR_WEIGHT)
        return {
            "graph_degree": len(self.account_devices[account]),
            "account_risk": self.account_fraud[account],
            "device_risk": self.device_fraud[device],
            "merchant_risk": round(rate, 6),
            "neighbor_fraud_accounts": sum(1 for n in neighbours if self.account_fraud[n] > 0),
            "community_size": size,
            "community_risk": round(community_risk, 6),
            "ring_flag": int(size >= 4 and community_risk >= 0.2),
            "mule_flag": int(device_shared >= 2 and self.device_fraud[device] >= 1),
            "linked_accounts": len(neighbours),
            "linked_account_ids": sorted(neighbours)[:12],
        }

    def observe(self, tx: dict) -> None:
        account, device, merchant = tx["account_id"], tx["device_id"], tx["merchant_id"]
        a, d, m = f"account:{account}", f"device:{device}", f"merchant:{merchant}"
        self.graph.add_edges_from([(a, d), (a, m)])
        self.account_device_graph.add_edge(a, d)
        self.device_accounts[device].add(account)
        self.account_devices[account].add(device)
        self._events += 1
        if self._events % self.refresh_every == 0:
            self.refresh_communities()

    def refresh_communities(self) -> None:
        """Louvain communities over the account-device graph, using only history seen so far."""
        communities = nx.community.louvain_communities(self.account_device_graph, seed=self.seed)
        self.community_of.clear()
        self.community_members.clear()
        self.community_fraud.clear()
        for cid, nodes in enumerate(sorted(communities, key=lambda c: sorted(c)[0])):
            members = sorted(n.removeprefix("account:") for n in nodes if n.startswith("account:"))
            if not members:
                continue
            self.community_members[cid] = members
            for member in members:
                self.community_of[member] = cid
                if self.account_fraud[member] > 0:
                    self.community_fraud[cid] += 1
