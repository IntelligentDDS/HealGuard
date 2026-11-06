from __future__ import annotations

import json
from collections import defaultdict

from loguru import logger


def summary_decision(event):
    if "decision_result" in event:
        value = {"result": event["decision_result"]}
        if event.get("selection") is not None:
            value["selection"] = event["selection"]
        return value
    if event.get("outcome") == "matched_exception":
        return {key: event[key] for key in ("outcome", "exception_type") if key in event}
    raise ValueError("unsupported_summary_decision")


def read_summary_path(runtime, diagnostics):
    """Accept only a unique ordering under the recorded sequence constraints."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if any(event.get("action") == "heal_after" and event.get("outcome") == "interrupted" for event in runtime):
        raise ValueError("summary_healing_interrupted")
    if not runtime or len({event.get("_pid") for event in runtime}) != 1:
        raise ValueError("multiple_or_missing_summary_processes")
    threads = {event["thread_id"] for event in runtime if "thread_id" in event}
    if len(threads) != 1 or None in threads:
        raise ValueError("concurrent_or_missing_summary_threads")
    if any(event.get("collector_fallback") for event in runtime):
        raise ValueError("summary_collector_failure")
    atoms, chains, fixed = [], [], {}
    def add_atom(outputs, lower, upper):
        if type(lower) is not int or type(upper) is not int or lower < 1 or upper < lower:
            raise ValueError("invalid_summary_sequence_bounds")
        index = len(atoms)
        atoms.append((lower, upper, tuple(outputs)))
        return index
    def point(point_id, decision=None):
        return (point_id, json.dumps(decision, sort_keys=True))
    for event in runtime:
        action, seq = event["action"], event["seq"]
        count = event.get("repeat_count", 1)
        if type(count) is not int or count < 1:
            raise ValueError("invalid_summary_repeat_count")
        last = event.get("last_repeat_seq", seq)
        if action in ("route_summary", "loop_summary"):
            path = event["path"]
            decisions = event.get("first_decisions", [])
            if not path or (count > 1 and decisions != event.get("last_decisions", [])):
                raise ValueError("inconsistent_summary_repeated_decisions")
            decision_index, outputs = 0, []
            for point_id in path:
                decision = None
                if decision_index < len(decisions) and decisions[decision_index]["point_id"] == point_id:
                    decision = summary_decision(decisions[decision_index])
                    decision_index += 1
                is_loop = point_id.endswith((":for_statement", ":while_statement"))
                if is_loop and decision is None:
                    nested = [item for item in runtime if item.get("action") == "loop_summary"
                              and item.get("loop_point_id") == point_id and seq <= item["seq"] <= last]
                    if action != "loop_summary" or not nested:
                        raise ValueError("nested_loop_placeholder_origin_unresolved")
                    diagnostics["epmr_nested_placeholders"] = diagnostics.get("epmr_nested_placeholders", 0) + count
                    continue
                if is_loop:
                    decision.setdefault("selection", None)
                values = [point(point_id)] if not is_loop else []
                if decision is not None:
                    values.append(point(point_id, decision))
                outputs.append(values)
            if decision_index != len(decisions):
                raise ValueError("summary_decision_alignment_unresolved")
            chain = [add_atom(values, seq, last) for _ in range(count) for values in outputs]
            fixed[chain[0]] = seq
            chains.append(chain)
        elif action == "path_point":
            if count == 1 and last != seq:
                raise ValueError("invalid_single_path_point_sequence")
            chain = [add_atom([point(event["point_id"])], seq, last) for _ in range(count)]
            fixed[chain[0]], fixed[chain[-1]] = seq, last
            chains.append(chain)
        elif action == "path_decision":
            add_atom([point(event["point_id"], summary_decision(event))], seq, seq)
        elif action == "path_decision_repeat_summary":
            first = event["first_seq"]
            originals = [item for item in runtime if item.get("action") == "path_decision"
                         and item.get("seq") == first and item.get("point_id") == event["point_id"]]
            if len(originals) != 1 or summary_decision(originals[0]) != summary_decision(event):
                raise ValueError("decision_repeat_origin_unresolved")
            chain = [add_atom([point(event["point_id"], summary_decision(event))], first + 1, last) for _ in range(count)]
            fixed[chain[-1]] = last
            chains.append(chain)
            add_atom([], seq, seq)
        elif action in ("heal_before", "heal_after", "call_stack_definition", "sink_before", "sink_after", "heal_failed"):
            add_atom([], seq, seq)
        else:
            raise ValueError(f"unsupported_summary_action:{action}")
    end = max(max(event["seq"], event.get("last_repeat_seq", event["seq"])) for event in runtime)
    diagnostics.update(epmr_summary_atoms=len(atoms), epmr_summary_last_seq=end,
                       epmr_path_source="original_runtime_summaries",
                       epmr_integrity="sequence_coverage_without_historical_digest")
    if len(atoms) != end:
        raise ValueError("summary_sequence_coverage_mismatch")
    # Tighten chain bounds before constructing the sparse assignment problem.
    bounds = [[fixed.get(index, low), fixed.get(index, high)] for index, (low, high, _) in enumerate(atoms)]
    for chain in chains:
        for left, right in zip(chain, chain[1:]):
            bounds[right][0] = max(bounds[right][0], bounds[left][0] + 1)
        for left, right in reversed(list(zip(chain, chain[1:]))):
            bounds[left][1] = min(bounds[left][1], bounds[right][1] - 1)
    if any(low > high for low, high in bounds):
        raise ValueError("summary_order_constraints_inconsistent")
    variable_count = sum(high - low + 1 for low, high in bounds)
    diagnostics["epmr_order_solver_variables"] = variable_count
    if variable_count > 200000:
        logger.warning(f"Summary ordering requires {variable_count} assignment variables; leaving EPMR missing without truncating data")
        raise ValueError("summary_order_solver_capacity_exceeded")
    variables = [(index, seq) for index, (low, high) in enumerate(bounds) for seq in range(low, high + 1)]
    atom_vars, slot_vars = defaultdict(list), defaultdict(list)
    for column, (index, seq) in enumerate(variables):
        atom_vars[index].append((column, seq))
        slot_vars[seq].append((column, 1))
    rows, cols, values, lower, upper = [], [], [], [], []
    def constrain(terms, low, high):
        row = len(lower)
        for column, value in terms:
            rows.append(row)
            cols.append(column)
            values.append(value)
        lower.append(low)
        upper.append(high)
    for index in range(len(atoms)):
        constrain([(column, 1) for column, _ in atom_vars[index]], 1, 1)
    for seq in range(1, end + 1):
        constrain(slot_vars[seq], 1, 1)
    for chain in chains:
        for left, right in zip(chain, chain[1:]):
            constrain(atom_vars[left] + [(column, -seq) for column, seq in atom_vars[right]], -np.inf, -1)
    def solve(objective):
        matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), len(variables))).tocsc()
        return milp(np.asarray(objective, dtype=float), integrality=np.ones(len(variables)),
                    bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lower, upper),
                    options={"time_limit": 10})
    candidate = solve(np.zeros(len(variables)))
    if candidate.status != 0:
        raise ValueError(f"summary_order_solver_status:{candidate.status}:{candidate.message}")
    selected = [variables[column] for column, value in enumerate(candidate.x) if value > 0.5]
    slots = {seq: atoms[index][2] for index, seq in selected}
    # A second solve looks for a different emitted sequence, not a permutation of
    # indistinguishable repetitions; neither solve consults the Golden path.
    changes = [(column, 1) for column, (index, seq) in enumerate(variables) if atoms[index][2] != slots[seq]]
    if changes:
        constrain(changes, 1, float("inf"))
        alternative = solve(np.zeros(len(variables)))
        if alternative.status == 0:
            diagnostics["epmr_order_ambiguity"] = json.dumps([
                {"seq": seq, "candidate_events": slots[seq], "alternative_events": atoms[index][2]}
                for column, (index, seq) in enumerate(variables)
                if alternative.x[column] > 0.5 and atoms[index][2] != slots[seq]], ensure_ascii=False)
            raise ValueError("summary_global_order_ambiguous")
        if alternative.status != 2:
            raise ValueError(f"summary_uniqueness_solver_status:{alternative.status}:{alternative.message}")
    return [value for seq in range(1, end + 1) for value in slots[seq]]
