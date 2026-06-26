"""Parse a pandapower IEEE test case into flat numpy arrays for use in relaxations.

Calling `load_network(case_name)` returns `(net, nd)` where `net` is the live
pandapower network object (used directly by the local solver) and `nd` is a
`NetworkData` instance containing all the constant arrays the convex relaxations
need (Y matrix, limits, costs, topology).

Bus ordering in all arrays: the 0-based row/column index into the arrays corresponds
to the position of each bus in `net.bus.index`, NOT the raw pandapower bus ID.
Use `nd.bus_ids[k]` to recover the pandapower bus ID for row k.
"""

from dataclasses import dataclass

import numpy as np
import pandapower as pp
import pandapower.networks as pn


# ── default case ──────────────────────────────────────────────────────────────
DEFAULT_CASE = "case9"


@dataclass
class NetworkData:
    """All constant network arrays needed to build the convex relaxations."""

    # ── sizing ────────────────────────────────────────────────────────────────
    n_buses: int
    n_gens: int     # total generators including the ext_grid (slack)
    n_loads: int    # rows in net.load
    baseMVA: float

    # ── bus data (shape: n_buses) ─────────────────────────────────────────────
    bus_ids: np.ndarray   # pandapower bus IDs for row-to-ID lookup
    v_min: np.ndarray     # per-unit lower bound on |V|
    v_max: np.ndarray     # per-unit upper bound on |V|
    slack_idx: int        # 0-based row index of the slack (ext_grid) bus
    v_ref: float          # reference voltage magnitude at slack bus [pu]

    # ── admittance matrix [per-unit, complex] (n_buses × n_buses) ────────────
    Y: np.ndarray         # G + jB; G = np.real(Y), B = np.imag(Y)

    # ── branch data for SOCP (n_branches) ────────────────────────────────────
    branch_from: np.ndarray  # 0-based row index of the from-bus
    branch_to: np.ndarray    # 0-based row index of the to-bus

    # ── generator data (n_gens) ───────────────────────────────────────────────
    # Generators are ordered: net.gen rows first, then ext_grid as the last entry.
    gen_bus_idx: np.ndarray  # 0-based row index of each generator's bus
    pg_min: np.ndarray       # MW
    pg_max: np.ndarray       # MW
    qg_min: np.ndarray       # MVar
    qg_max: np.ndarray       # MVar
    cost_c0: np.ndarray      # $/hr  (constant term)
    cost_c1: np.ndarray      # $/MWh (linear coefficient)
    cost_c2: np.ndarray      # $/MW²h (quadratic coefficient)

    # ── load data (n_loads) ──────────────────────────────────────────────────
    # Matches the row order of net.load exactly.
    load_bus_idx: np.ndarray  # 0-based row index of each load's bus
    pd_nominal: np.ndarray    # MW   (nominal demand — used to define the sampling range)
    qd_nominal: np.ndarray    # MVar (nominal demand)


# ── helper ─────────────────────────────────────────────────────────────────────

def _extract_ybus(net, bus_ids):
    """Return the n×n complex admittance matrix in bus-table order."""
    ext_to_int = net._pd2ppc_lookups["bus"]
    int_ids = [ext_to_int[b] for b in bus_ids]
    Y_int = net._ppc["internal"]["Ybus"].toarray()
    return Y_int[np.ix_(int_ids, int_ids)]


def _branch_topology(Y, threshold=1e-8):
    """Return (branch_from, branch_to) derived from the off-diagonal sparsity of Y.

    Using the Y matrix directly (rather than net.line) ensures transformers are
    included — they create off-diagonal Y entries that net.line does not cover.
    We enumerate undirected pairs k < m where |Y[k,m]| or |Y[m,k]| is nonzero.
    """
    import scipy.sparse as sp
    # Convert to sparse and find nonzero off-diagonal upper-triangle entries.
    # |Y| + |Y.T| is nonzero wherever either Y[k,m] or Y[m,k] is nonzero.
    Ys = sp.csr_matrix(Y)
    upper = sp.triu(np.abs(Ys) + np.abs(Ys.T), k=1)
    upper.eliminate_zeros()
    upper = upper > threshold
    fr, to = upper.nonzero()
    return fr.astype(int), to.astype(int)


def _generator_data(net, bus_id_to_idx):
    """Return generator arrays ordered as [net.gen rows ..., ext_grid].

    The ext_grid (slack bus) is always appended last so it can be distinguished
    from dispatchable generators if needed.
    """
    cost_table = net.poly_cost.copy()
    gen_cost   = cost_table[cost_table["et"] == "gen"].set_index("element")
    ext_cost   = cost_table[cost_table["et"] == "ext_grid"].set_index("element")

    def _gen_cost(idx, table):
        if idx in table.index:
            row = table.loc[idx]
            return float(row["cp0_eur"]), float(row["cp1_eur_per_mw"]), float(row["cp2_eur_per_mw2"])
        return 0.0, 0.0, 0.0

    # ── dispatchable generators (net.gen) ──────────────────────────────────
    gen_bus_idx, pg_min, pg_max, qg_min, qg_max = [], [], [], [], []
    c0_list, c1_list, c2_list = [], [], []

    for i in net.gen.index:
        row = net.gen.loc[i]
        gen_bus_idx.append(bus_id_to_idx[int(row["bus"])])
        pg_min.append(float(row.get("min_p_mw", 0.0)))
        pg_max.append(float(row.get("max_p_mw", 9999.0)))
        qg_min.append(float(row.get("min_q_mvar", -9999.0)))
        qg_max.append(float(row.get("max_q_mvar",  9999.0)))
        c0, c1, c2 = _gen_cost(i, gen_cost)
        c0_list.append(c0); c1_list.append(c1); c2_list.append(c2)

    # ── ext_grid (slack bus generator) ────────────────────────────────────
    eg = net.ext_grid.iloc[0]
    gen_bus_idx.append(bus_id_to_idx[int(eg["bus"])])
    pg_min.append(float(eg.get("min_p_mw", -9999.0)))
    pg_max.append(float(eg.get("max_p_mw",  9999.0)))
    qg_min.append(float(eg.get("min_q_mvar", -9999.0)))
    qg_max.append(float(eg.get("max_q_mvar",  9999.0)))
    eg_idx = int(net.ext_grid.index[0])
    c0, c1, c2 = _gen_cost(eg_idx, ext_cost)
    c0_list.append(c0); c1_list.append(c1); c2_list.append(c2)

    return (
        np.array(gen_bus_idx),
        np.array(pg_min),  np.array(pg_max),
        np.array(qg_min),  np.array(qg_max),
        np.array(c0_list), np.array(c1_list), np.array(c2_list),
    )


# ── public API ────────────────────────────────────────────────────────────────

def load_network(case_name=DEFAULT_CASE, v_min=None, v_max=None):
    """Load a pandapower IEEE test case and extract a NetworkData object.

    Parameters
    ----------
    case_name : str
    v_min, v_max : float or None
        If provided, override the per-bus voltage bounds in the pandapower
        case with a single uniform value.  Useful for large cases (e.g.
        case300) whose pandapower bounds are tighter than the standard
        MATPOWER defaults and render the OPF infeasible at nominal demand.

    Returns
    -------
    net : pandapower network (used by solve_local_pypower; kept for reference)
    nd  : NetworkData with all constant arrays for the relaxations
    """
    net = getattr(pn, case_name)()

    # Run a power flow to populate net._ppc (needed for Y-bus extraction).
    pp.runpp(net, numba=False, verbose=False)

    bus_ids      = net.bus.index.to_numpy()
    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}
    n_buses      = len(bus_ids)

    Y = _extract_ybus(net, bus_ids)

    # Slack bus
    slack_bus_id = int(net.ext_grid["bus"].iloc[0])
    slack_idx    = bus_id_to_idx[slack_bus_id]
    v_ref        = float(net.ext_grid["vm_pu"].iloc[0])

    branch_from, branch_to = _branch_topology(Y)

    (gen_bus_idx, pg_min, pg_max, qg_min, qg_max,
     cost_c0, cost_c1, cost_c2) = _generator_data(net, bus_id_to_idx)

    load_bus_idx = np.array([bus_id_to_idx[int(b)] for b in net.load["bus"].values])
    pd_nominal   = net.load["p_mw"].values.copy().astype(float)
    qd_nominal   = net.load["q_mvar"].values.copy().astype(float)

    v_min_arr = net.bus["min_vm_pu"].values.astype(float)
    v_max_arr = net.bus["max_vm_pu"].values.astype(float)
    if v_min is not None:
        v_min_arr = np.full(n_buses, float(v_min))
    if v_max is not None:
        v_max_arr = np.full(n_buses, float(v_max))

    nd = NetworkData(
        n_buses=n_buses,
        n_gens=len(gen_bus_idx),
        n_loads=len(pd_nominal),
        baseMVA=float(net.sn_mva),
        bus_ids=bus_ids,
        v_min=v_min_arr,
        v_max=v_max_arr,
        slack_idx=slack_idx,
        v_ref=v_ref,
        Y=Y,
        branch_from=branch_from,
        branch_to=branch_to,
        gen_bus_idx=gen_bus_idx,
        pg_min=pg_min, pg_max=pg_max,
        qg_min=qg_min, qg_max=qg_max,
        cost_c0=cost_c0, cost_c1=cost_c1, cost_c2=cost_c2,
        load_bus_idx=load_bus_idx,
        pd_nominal=pd_nominal,
        qd_nominal=qd_nominal,
    )
    return net, nd
