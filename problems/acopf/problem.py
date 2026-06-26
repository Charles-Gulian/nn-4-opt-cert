"""AC-OPF problem: local solver (pandapower / IPOPT) and three convex relaxations.

Problem
-------
Minimize total generation cost subject to AC power flow equations:

    min   sum_g  c2_g * P_g^2 + c1_g * P_g + c0_g
    s.t.  AC power balance at every bus
          |V_k|_min <= |V_k| <= |V_k|_max  (voltage magnitude limits)
          P_g_min <= P_g <= P_g_max          (generator real-power limits)
          Q_g_min <= Q_g <= Q_g_max          (generator reactive-power limits)

Parameter
---------
p = [Pd_0, ..., Pd_{K-1}, Qd_0, ..., Qd_{K-1}]   (MW, MVar for K = n_loads loads)

This vector matches the row ordering of `net.load` exactly.

Relaxations
-----------
Three relaxations are implemented, each returning a valid lower bound on the
optimal cost:

  SDP          -- Semidefinite relaxation (Lavaei-Low 2012, real 2n x 2n formulation).
                  Lifts V V^T to a 2n x 2n PSD matrix X; exact when rank(X) == 1.

  SOCP         -- Second-order cone relaxation (Jabr 2006).
                  Introduces c_e = Re(V_k conj(V_m)), s_e = Im(V_k conj(V_m)),
                  u_k = |V_k|^2 per branch e=(k,m); exact when c_e^2+s_e^2 = u_k*u_m.

  chordal_sdp  -- Sparse/chordal SDP (Madani-Ashraphijuo-Lavaei 2014).
                  Uses a tree decomposition to replace the dense 2n×2n PSD
                  constraint with one small Hermitian PSD constraint per bag.
                  Tractable for large cases (case300: treewidth≈6, ~15s/solve).

Standard interface
------------------
  solve_relaxation(p, args=None)  ->  (value, result_dict)
  solve_local(p, args=None)       ->  (value, result_dict)

Parameter sampling lives in generate_data.py.
"""

import pathlib
import shutil
import sys

import numpy as np
import cvxpy as cp
import pyomo.environ as pyo
import scipy.sparse as sp

from problems.acopf.network import NetworkData, load_network, DEFAULT_CASE


# ── local solver ──────────────────────────────────────────────────────────────

def solve_local(p, args=None):
    """Solve AC-OPF via Pyomo + IPOPT (polar form).

    IPOPT handles ill-conditioned Y-bus matrices (e.g. PEGASE cases with
    extreme transformer impedances) far better than pandapower's PIPS solver.

    Formulation (polar, per-unit):
        P_k = sum_m Vm_k * Vm_m * (G_km cos(Va_k-Va_m) + B_km sin(Va_k-Va_m))
        Q_k = sum_m Vm_k * Vm_m * (G_km sin(Va_k-Va_m) - B_km cos(Va_k-Va_m))

    Parameters
    ----------
    p : array-like, shape (2 * n_loads,)
        Concatenated [Pd (MW), Qd (MVar)] for each row of net.load.
    args : dict, optional
        'nd'        : pre-built NetworkData
        'case_name' : fallback if 'nd' is absent

    Returns
    -------
    value : float   — optimal total generation cost ($/hr), or np.nan if infeasible
    result : dict   — dispatch and voltage profile on success
    """
    args = args or {}
    nd = args.get("nd")
    if nd is None:
        _, nd = load_network(args.get("case_name", DEFAULT_CASE))

    n_loads = nd.n_loads
    Pd = np.asarray(p[:n_loads], dtype=float)
    Qd = np.asarray(p[n_loads:], dtype=float)
    S  = nd.baseMVA

    G = np.real(nd.Y)
    B = np.imag(nd.Y)
    n = nd.n_buses

    # Aggregate demand per bus (MW → per-unit)
    Pd_bus = np.zeros(n)
    Qd_bus = np.zeros(n)
    for i, k in enumerate(nd.load_bus_idx):
        Pd_bus[k] += Pd[i] / S
        Qd_bus[k] += Qd[i] / S

    # Generator-to-bus mapping: list of gen indices at each bus
    gens_at = [[] for _ in range(n)]
    for g, k in enumerate(nd.gen_bus_idx):
        gens_at[k].append(g)

    m = pyo.ConcreteModel()

    # ── variables ────────────────────────────────────────────────────────────
    m.buses = pyo.Set(initialize=range(n))
    m.gens  = pyo.Set(initialize=range(nd.n_gens))

    # ── warm start from AC power flow ─────────────────────────────────────────
    # Running pandapower's Newton-Raphson AC power flow first gives IPOPT a
    # much better starting point for Vm and Va.  Flat start (Vm=1, Va=0) with
    # a large Y-bus causes huge initial constraint violations and KKT
    # ill-conditioning; power-flow angles are typically within ±40° and Vm
    # close to 1, dramatically reducing the initial infeasibility.
    import copy
    import pandapower as pp

    Vm_init = np.ones(n)
    Va_init = np.zeros(n)
    try:
        net_pf = args.get("net")
        if net_pf is None:
            import pandapower.networks as pn
            net_pf = getattr(pn, args.get("case_name", DEFAULT_CASE))()
        net_pf = copy.deepcopy(net_pf)
        net_pf.load["p_mw"]   = Pd   # Pd/Qd are already in MW/MVar
        net_pf.load["q_mvar"] = Qd
        pp.runpp(net_pf, numba=False, verbose=False)
        if net_pf.converged:
            # Map pandapower bus IDs → 0-based nd row indices
            bid2idx = {int(b): i for i, b in enumerate(nd.bus_ids)}
            for bus_id, idx in bid2idx.items():
                row = net_pf.res_bus.loc[bus_id]
                Vm_init[idx] = float(row["vm_pu"])
                Va_init[idx] = float(np.radians(row["va_degree"]))
    except Exception:
        pass  # fall back to flat start if power flow fails

    m.Vm = pyo.Var(m.buses, bounds=lambda _, k: (nd.v_min[k], nd.v_max[k]),
                   initialize=lambda _, k: float(Vm_init[k]))
    m.Va = pyo.Var(m.buses, bounds=(-np.pi, np.pi),
                   initialize=lambda _, k: float(Va_init[k]))

    # Initialise Pg proportionally to pg_max so total generation ≈ total load.
    # When pg_min=0 (common in IEEE cases), pg_min init means zero generation
    # against full load — a huge constraint violation from the start.
    total_load_pu = Pd_bus.sum()
    pg_max_pu = nd.pg_max / S
    pg_min_pu = nd.pg_min / S
    cap_total = pg_max_pu.sum() if pg_max_pu.sum() > 0 else 1.0
    pg_init_pu = np.clip(
        pg_max_pu / cap_total * total_load_pu,
        pg_min_pu, pg_max_pu,
    )

    m.Pg = pyo.Var(m.gens,
                   bounds=lambda _, g: (pg_min_pu[g], pg_max_pu[g]),
                   initialize=lambda _, g: float(pg_init_pu[g]))
    m.Qg = pyo.Var(m.gens,
                   bounds=lambda _, g: (nd.qg_min[g] / S, nd.qg_max[g] / S),
                   initialize=lambda _, g: (nd.qg_min[g] + nd.qg_max[g]) / (2 * S))

    # Slack bus: fix angle to 0
    m.Va[nd.slack_idx].fix(0.0)
    m.Vm[nd.slack_idx].fix(nd.v_ref)

    # ── power balance ─────────────────────────────────────────────────────────
    def _p_balance(m, k):
        P_flow = sum(
            m.Vm[k] * m.Vm[j] * (G[k, j] * pyo.cos(m.Va[k] - m.Va[j])
                                  + B[k, j] * pyo.sin(m.Va[k] - m.Va[j]))
            for j in range(n) if abs(G[k, j]) + abs(B[k, j]) > 1e-12
        )
        P_gen = sum(m.Pg[g] for g in gens_at[k]) if gens_at[k] else 0.0
        return P_flow == P_gen - Pd_bus[k]

    def _q_balance(m, k):
        Q_flow = sum(
            m.Vm[k] * m.Vm[j] * (G[k, j] * pyo.sin(m.Va[k] - m.Va[j])
                                  - B[k, j] * pyo.cos(m.Va[k] - m.Va[j]))
            for j in range(n) if abs(G[k, j]) + abs(B[k, j]) > 1e-12
        )
        Q_gen = sum(m.Qg[g] for g in gens_at[k]) if gens_at[k] else 0.0
        return Q_flow == Q_gen - Qd_bus[k]

    m.p_bal = pyo.Constraint(m.buses, rule=_p_balance)
    m.q_bal = pyo.Constraint(m.buses, rule=_q_balance)

    # ── objective ─────────────────────────────────────────────────────────────
    def _cost(m):
        return sum(
            nd.cost_c0[g]
            + nd.cost_c1[g] * S    * m.Pg[g]
            + nd.cost_c2[g] * S**2 * m.Pg[g] ** 2
            for g in range(nd.n_gens)
        )
    m.obj = pyo.Objective(rule=_cost, sense=pyo.minimize)

    # ── solve ─────────────────────────────────────────────────────────────────
    # Prefer the ipopt binary that lives alongside the active Python interpreter
    # (i.e. inside the current conda env) over any system-level binary, which
    # may have broken rpath links on macOS.
    _ipopt_bin = (
        shutil.which("ipopt", path=str(pathlib.Path(sys.executable).parent))
        or "ipopt"
    )
    solver = pyo.SolverFactory("ipopt", executable=_ipopt_bin)
    solver.options["max_iter"]           = 1000
    solver.options["tol"]               = 1e-6
    solver.options["print_level"]       = 0   # silent
    solver.options["mu_strategy"]       = "adaptive"
    # Gradient-based scaling handles the ~10^4 spread in Y-bus entries.
    solver.options["nlp_scaling_method"] = "gradient-based"
    # Accept a slightly looser solution rather than failing outright.
    # For our use case (1e-3 cost accuracy) this is more than tight enough.
    solver.options["acceptable_tol"]    = 1e-4
    solver.options["acceptable_iter"]   = 5

    res = solver.solve(m, tee=False)

    ok = (res.solver.status == pyo.SolverStatus.ok and
          res.solver.termination_condition in (
              pyo.TerminationCondition.optimal,
              pyo.TerminationCondition.locallyOptimal,
              pyo.TerminationCondition.feasible,   # IPOPT "acceptable" solution
          ))

    if not ok:
        return np.nan, {"success": False}

    # Recover cost in $/hr (objective is already in $/hr since c0/c1/c2 are)
    cost = float(pyo.value(m.obj))

    # Reconstruct per-generator dispatch in MW/MVar
    pg_mw   = np.array([pyo.value(m.Pg[g]) * S for g in range(nd.n_gens)])
    qg_mvar = np.array([pyo.value(m.Qg[g]) * S for g in range(nd.n_gens)])
    vm_pu   = np.array([pyo.value(m.Vm[k])     for k in range(n)])
    va_deg  = np.array([np.degrees(pyo.value(m.Va[k])) for k in range(n)])

    return cost, {
        "success": True,
        "pg_mw":   pg_mw,
        "qg_mvar": qg_mvar,
        "vm_pu":   vm_pu,
        "va_deg":  va_deg,
    }


def solve_local_pypower(p, args=None):
    """Solve AC-OPF via pandapower's built-in PIPS solver.

    Kept as a reference / fast alternative for well-conditioned cases.
    Fails on ill-conditioned networks (e.g. PEGASE) — use solve_local instead.

    Parameters
    ----------
    p : array-like, shape (2 * n_loads,)
    args : dict, optional
        'net'       : pre-loaded pandapower network (deepcopied before use)
        'case_name' : fallback if 'net' is absent
    """
    import copy
    import pandapower as pp

    args = args or {}
    net = args.get("net")
    if net is None:
        net, _ = load_network(args.get("case_name", DEFAULT_CASE))
    net = copy.deepcopy(net)

    n_loads = len(net.load)
    net.load["p_mw"]   = p[:n_loads]
    net.load["q_mvar"] = p[n_loads:]

    if "cp2_eur_per_mw2" in net.poly_cost.columns:
        net.poly_cost["cp2_eur_per_mw2"] = net.poly_cost["cp2_eur_per_mw2"].clip(lower=1e-4)

    try:
        pp.runpp(net, numba=False, verbose=False)
    except Exception:
        pass

    try:
        pp.runopp(net, numba=False, verbose=False)
        cost = float(net.res_cost)
        success = np.isfinite(cost)
    except Exception:
        success = False
        cost = np.nan

    if success:
        return cost, {
            "success": True,
            "pg_mw":   net.res_gen["p_mw"].values.copy(),
            "qg_mvar": net.res_gen["q_mvar"].values.copy(),
            "vm_pu":   net.res_bus["vm_pu"].values.copy(),
            "va_deg":  net.res_bus["va_degree"].values.copy(),
        }
    return np.nan, {"success": False}


# ── SDP relaxation (Lavaei-Low, real 2n×2n formulation) ──────────────────────
#
# Lift v = [V_re; V_im] ∈ R^{2n} to X = v v^T ∈ R^{2n×2n}, X ⪰ 0.
#
# Key identities:
#   Re(V_k conj(V_m)) = X[k,m] + X[n+k, n+m]
#   Im(V_k conj(V_m)) = X[n+k,m] − X[k, n+m]
#   |V_k|^2           = X[k,k]   + X[n+k, n+k]
#
# Power injection at bus k (per-unit, with G = Re(Y), B = Im(Y)):
#   P_k = sum_m  G[k,m]*(X[k,m]+X[n+k,n+m])  + B[k,m]*(X[n+k,m]−X[k,n+m])
#   Q_k = sum_m  G[k,m]*(X[n+k,m]−X[k,n+m])  − B[k,m]*(X[k,m]+X[n+k,n+m])
#
def _build_sdp_problem(nd: NetworkData):
    """Build and return a re-usable cvxpy SDP for the given network.

    Returns
    -------
    prob      : cp.Problem (parameterised by Pd_param, Qd_param)
    Pd_param  : cp.Parameter, shape (n_loads,)   [MW]
    Qd_param  : cp.Parameter, shape (n_loads,)   [MVar]
    X         : cp.Variable, shape (2n, 2n) — the SDP matrix
    P_g       : cp.Variable, shape (n_gens,)     [MW]
    Q_g       : cp.Variable, shape (n_gens,)     [MVar]
    """
    n   = nd.n_buses
    nn  = 2 * n
    G   = np.real(nd.Y)
    B   = np.imag(nd.Y)
    S   = nd.baseMVA   # base power [MVA]

    # Parameters: demands in per-unit (divided by baseMVA for scaling)
    Pd_param = cp.Parameter(nd.n_loads, name="Pd", value=nd.pd_nominal.copy())
    Qd_param = cp.Parameter(nd.n_loads, name="Qd", value=nd.qd_nominal.copy())

    # SDP variable: X = vv^T ∈ R^{2n×2n}, entries are per-unit voltages squared (order 1)
    X = cp.Variable((nn, nn), symmetric=True)

    # Generator dispatch in per-unit so all decision variables are order 1.
    # P_g_pu = P_g_MW / baseMVA.  Cost is recovered by substituting back:
    #   cost = c0 + c1*(P_g_pu*S) + c2*(P_g_pu*S)^2 = c0 + c1*S*P_g_pu + c2*S^2*P_g_pu^2
    P_g = cp.Variable(nd.n_gens, name="P_g_pu")
    Q_g = cp.Variable(nd.n_gens, name="Q_g_pu")

    constraints = [X >> 0]

    # ── voltage magnitude bounds ────────────────────────────────────────────
    for k in range(n):
        v_sq = X[k, k] + X[n + k, n + k]
        constraints += [
            nd.v_min[k] ** 2 <= v_sq,
            v_sq <= nd.v_max[k] ** 2,
        ]

    # ── slack-bus voltage magnitude fixed to v_ref ─────────────────────────
    s = nd.slack_idx
    constraints.append(X[s, s] + X[n + s, n + s] == nd.v_ref ** 2)


    # ── demand and generation aggregated per bus (per-unit) ───────────────
    L_inc = sp.csr_matrix(
        (np.ones(nd.n_loads), (nd.load_bus_idx, np.arange(nd.n_loads))),
        shape=(n, nd.n_loads))
    G_inc = sp.csr_matrix(
        (np.ones(nd.n_gens), (nd.gen_bus_idx, np.arange(nd.n_gens))),
        shape=(n, nd.n_gens))

    # Use expr @ M.T instead of M @ expr: routes through CVXPY's __matmul__
    # rather than scipy's, avoiding the deprecated-* warning.
    Pd_bus = (Pd_param / S) @ L_inc.T
    Qd_bus = (Qd_param / S) @ L_inc.T
    Pg_bus = P_g @ G_inc.T
    Qg_bus = Q_g @ G_inc.T

    # ── power balance: all terms are per-unit (order 1) ──────────────────
    for k in range(n):
        Re_Wkm = X[k, :n]     + X[n + k, n:]    # Re(V_k conj(V_m))
        Im_Wkm = X[n + k, :n] - X[k, n:]         # Im(V_k conj(V_m))

        P_inj_pu = G[k, :] @ Re_Wkm + B[k, :] @ Im_Wkm
        Q_inj_pu = G[k, :] @ Im_Wkm - B[k, :] @ Re_Wkm

        constraints += [
            P_inj_pu == Pg_bus[k] - Pd_bus[k],
            Q_inj_pu == Qg_bus[k] - Qd_bus[k],
        ]

    # ── generator limits (per-unit) ───────────────────────────────────────
    constraints += [
        nd.pg_min / S <= P_g, P_g <= nd.pg_max / S,
        nd.qg_min / S <= Q_g, Q_g <= nd.qg_max / S,
    ]

    # ── objective: cost with P_g in per-unit ─────────────────────────────
    cost_expr = (
        nd.cost_c0.sum()
        + nd.cost_c1 @ (S * P_g)
        + cp.quad_form(P_g, np.diag(nd.cost_c2 * S**2))
    )
    prob = cp.Problem(cp.Minimize(cost_expr), constraints)
    return prob, Pd_param, Qd_param, X, P_g, Q_g


# ── SOCP relaxation (Jabr) ─────────────────────────────────────────────────
#
# Per-bus voltage variable:   u[k] = |V_k|^2                    (per-unit)
# Per-branch variables:       c[e] = Re(V_{from} conj(V_{to}))  (per-unit)
#                             s[e] = Im(V_{from} conj(V_{to}))  (per-unit)
#
# SOCP constraint for each branch e=(k,m):
#   c[e]^2 + s[e]^2 <= u[k] * u[m]
# equivalently (rotated SOC):
#   || [2*c[e]; 2*s[e]; u[k]-u[m]] ||_2 <= u[k] + u[m]
#
# Power injection using the full admittance matrix (bus pair (k,m) with Y[k,m]!=0):
#   Re(conj(Y[k,m]) * W[k,m]) = G[k,m]*c_km + B[k,m]*s_km
#   Im(conj(Y[k,m]) * W[k,m]) = G[k,m]*s_km − B[k,m]*c_km
#
# where c_km = c[e] if branch e goes k→m, or c[e] if k←m (c is symmetric),
#       s_km = s[e] if k→m, or −s[e] if k←m (s is antisymmetric).
#
def _build_socp_problem(nd: NetworkData):
    """Build and return a re-usable cvxpy SOCP for the given network.

    Returns
    -------
    prob      : cp.Problem
    Pd_param  : cp.Parameter, shape (n_loads,)
    Qd_param  : cp.Parameter, shape (n_loads,)
    u         : cp.Variable, shape (n_buses,)   — |V|^2 per bus [pu]
    c_var     : cp.Variable, shape (n_branches,) — Re(V_from conj(V_to)) [pu]
    s_var     : cp.Variable, shape (n_branches,) — Im(V_from conj(V_to)) [pu]
    P_g       : cp.Variable, shape (n_gens,)     [MW]
    Q_g       : cp.Variable, shape (n_gens,)     [MVar]
    """
    n   = nd.n_buses
    nb  = len(nd.branch_from)
    S   = nd.baseMVA


    Pd_param = cp.Parameter(nd.n_loads, name="Pd", value=nd.pd_nominal.copy())
    Qd_param = cp.Parameter(nd.n_loads, name="Qd", value=nd.qd_nominal.copy())

    u     = cp.Variable(n,  name="u",     nonneg=True)   # |V|^2 per bus [pu]
    c_var = cp.Variable(nb, name="c")                    # Re(V_k conj(V_m)) [pu]
    s_var = cp.Variable(nb, name="s")                    # Im(V_k conj(V_m)) [pu]
    P_g   = cp.Variable(nd.n_gens, name="P_g_pu")        # generator real power [pu]
    Q_g   = cp.Variable(nd.n_gens, name="Q_g_pu")        # generator reactive power [pu]

    constraints = []

    # ── voltage magnitude bounds ──────────────────────────────────────────
    constraints += [nd.v_min ** 2 <= u, u <= nd.v_max ** 2]
    constraints.append(u[nd.slack_idx] == nd.v_ref ** 2)

    # ── SOCP (Jabr) constraints per branch ─────────────────────────────────
    # Each branch e (from k→m): ||[2c_e, 2s_e, u_k - u_m]||_2 <= u_k + u_m
    # cp.norm does not support axis= for per-column norms, so we keep the
    # per-branch loop.  nb is O(branches) ~ O(n), not O(n²), so this is fine.
    fr = nd.branch_from
    to = nd.branch_to
    for e, (k, m) in enumerate(zip(fr, to)):
        constraints.append(
            cp.norm(cp.hstack([2 * c_var[e], 2 * s_var[e], u[k] - u[m]]))
            <= u[k] + u[m]
        )

    # ── power balance at each bus (vectorized) ────────────────────────────
    # Build sparse coefficient matrices A_Pc, A_Ps, A_Qc, A_Qs (n × nb)
    # such that:
    #   P_inj = diag(G)*u  +  A_Pc @ c_var  +  A_Ps @ s_var
    #   Q_inj = -diag(B)*u +  A_Qc @ c_var  +  A_Qs @ s_var
    #
    # For branch e connecting from-bus k to to-bus m:
    # Power flow convention (from derivation of Jabr):
    #   P_k += G[k,m]*c_e + B[k,m]*s_e   (from-bus)
    #   Q_k += G[k,m]*s_e - B[k,m]*c_e   (from-bus)
    #   P_m += G[m,k]*c_e - B[m,k]*s_e   (to-bus,  s_mk = -s_km)
    #   Q_m +=-G[m,k]*s_e - B[m,k]*c_e   (to-bus)
    Gfr = np.real(nd.Y[fr, to])   # G[k,m] for each branch  (nb,)
    Bfr = np.imag(nd.Y[fr, to])  # B[k,m] for each branch  (nb,)
    erange = np.arange(nb)

    rows_f = fr;  rows_t = to   # from-bus and to-bus row indices

    A_Pc = sp.csr_matrix(
        (np.concatenate([ Gfr,  Gfr]),
         (np.concatenate([rows_f, rows_t]), np.concatenate([erange, erange]))),
        shape=(n, nb), dtype=float)

    A_Ps = sp.csr_matrix(
        (np.concatenate([ Bfr, -Bfr]),
         (np.concatenate([rows_f, rows_t]), np.concatenate([erange, erange]))),
        shape=(n, nb), dtype=float)

    A_Qc = sp.csr_matrix(
        (np.concatenate([-Bfr, -Bfr]),
         (np.concatenate([rows_f, rows_t]), np.concatenate([erange, erange]))),
        shape=(n, nb), dtype=float)

    A_Qs = sp.csr_matrix(
        (np.concatenate([ Gfr, -Gfr]),
         (np.concatenate([rows_f, rows_t]), np.concatenate([erange, erange]))),
        shape=(n, nb), dtype=float)

    # Diagonal shunt terms
    G_diag = np.real(np.diag(nd.Y))
    B_diag = np.imag(np.diag(nd.Y))

    # Incidence matrices for loads and generators (n × n_loads / n × n_gens)
    L_inc = sp.csr_matrix(
        (np.ones(nd.n_loads), (nd.load_bus_idx, np.arange(nd.n_loads))),
        shape=(n, nd.n_loads))
    G_inc = sp.csr_matrix(
        (np.ones(nd.n_gens), (nd.gen_bus_idx, np.arange(nd.n_gens))),
        shape=(n, nd.n_gens))

    P_inj = cp.multiply(G_diag, u) + c_var @ A_Pc.T + s_var @ A_Ps.T
    Q_inj = cp.multiply(-B_diag, u) + c_var @ A_Qc.T + s_var @ A_Qs.T

    Pd_bus = (Pd_param / S) @ L_inc.T
    Qd_bus = (Qd_param / S) @ L_inc.T
    Pg_bus = P_g @ G_inc.T
    Qg_bus = Q_g @ G_inc.T

    constraints += [
        P_inj == Pg_bus - Pd_bus,
        Q_inj == Qg_bus - Qd_bus,
    ]

    # ── generator limits (per-unit) ───────────────────────────────────────
    constraints += [
        nd.pg_min / S <= P_g, P_g <= nd.pg_max / S,
        nd.qg_min / S <= Q_g, Q_g <= nd.qg_max / S,
    ]

    cost_expr = (
        nd.cost_c0.sum()
        + nd.cost_c1 @ (S * P_g)
        + cp.quad_form(P_g, np.diag(nd.cost_c2 * S**2))
    )
    prob = cp.Problem(cp.Minimize(cost_expr), constraints)
    return prob, Pd_param, Qd_param, u, c_var, s_var, P_g, Q_g


# ── chordal SDP relaxation (Madani-Ashraphijuo-Lavaei 2014) ──────────────────
#
# Uses a greedy minimum-fill elimination ordering (chordal.py) to decompose
# the network graph into a set of clique bags.  Instead of a single dense
# n×n Hermitian PSD constraint, we impose one small PSD constraint per bag.
#
# Variables per bag are tied together via shared scalars:
#   V2[k]    = |V_k|^2                      (real, one per bus)
#   VV[(k,m)] = V_k * conj(V_m), k < m      (complex, one per unique bag pair)
#
# For each bag B_i with buses [b_0,...,b_{s-1}]:
#   W_i ∈ C^{s×s} Hermitian PSD
#   W_i[r, r]  = V2[b_r]            (diagonal = voltage magnitude squared)
#   W_i[r, t]  = VV[(b_r, b_t)]     (upper triangle, b_r < b_t)
#   W_i[t, r]  = conj(W_i[r, t])    (automatic from Hermitian variable)
#
# Consistency between bags sharing a bus-pair (b, c) is enforced implicitly:
# both bags constrain their (b, c) entry to the same scalar VV[(b,c)].
#
# Power balance (complex form):
#   P_k + jQ_k = sum_m  conj(Y[k,m]) * W[k,m]
#
def _build_chordal_sdp_problem(nd: NetworkData):
    """Build and return a re-usable cvxpy chordal SDP for the given network.

    Returns
    -------
    prob      : cp.Problem
    Pd_param  : cp.Parameter, shape (n_loads,)
    Qd_param  : cp.Parameter, shape (n_loads,)
    bag_vars  : list of cp.Variable, one Hermitian PSD matrix per bag
    bags      : list of sorted lists of 0-based bus indices (one per bag)
    P_g       : cp.Variable, shape (n_gens,)
    Q_g       : cp.Variable, shape (n_gens,)
    """
    from problems.acopf.chordal import greedy_elimination, unique_pairs_in_bags

    n = nd.n_buses
    Y = nd.Y          # complex n×n admittance matrix
    S = nd.baseMVA

    # ── Step 1: tree decomposition ────────────────────────────────────────────
    bags, tw = greedy_elimination(nd.branch_from, nd.branch_to, n)

    # ── Step 2: shared scalar variables ──────────────────────────────────────
    V2 = cp.Variable(n, nonneg=True, name="V2")   # |V_k|^2, one per bus

    pairs = unique_pairs_in_bags(bags)
    # VV[(k,m)] = V_k * conj(V_m) for k < m; complex scalar
    VV = {(k, m): cp.Variable(complex=True, name=f"VV_{k}_{m}") for (k, m) in pairs}

    # Generator dispatch, per-unit
    P_g = cp.Variable(nd.n_gens, name="P_g_pu")
    Q_g = cp.Variable(nd.n_gens, name="Q_g_pu")

    Pd_param = cp.Parameter(nd.n_loads, name="Pd", value=nd.pd_nominal.copy())
    Qd_param = cp.Parameter(nd.n_loads, name="Qd", value=nd.qd_nominal.copy())

    constraints = []

    # ── Step 3: per-bag Hermitian PSD variables ───────────────────────────────
    bag_vars = []
    for bag in bags:
        s = len(bag)
        W_bag = cp.Variable((s, s), hermitian=True)
        constraints.append(W_bag >> 0)

        # Diagonal: W_bag[i,i] = V2[bag[i]]
        for i, bi in enumerate(bag):
            constraints.append(cp.real(W_bag[i, i]) == V2[bi])

        # Upper triangle: W_bag[i,j] = VV[(min,max)]
        for i, bi in enumerate(bag):
            for j in range(i + 1, len(bag)):
                bj = bag[j]
                k, m = (bi, bj) if bi < bj else (bj, bi)
                if bi < bj:
                    constraints.append(W_bag[i, j] == VV[(k, m)])
                else:
                    constraints.append(W_bag[i, j] == cp.conj(VV[(k, m)]))

        bag_vars.append(W_bag)

    # ── Step 4: helper to get W[k,m] CVXPY expression ────────────────────────
    def _W(k, m):
        if k == m:
            return V2[k]
        key = (min(k, m), max(k, m))
        vv = VV[key]
        return vv if k < m else cp.conj(vv)

    # ── Step 5: power balance (complex form) ──────────────────────────────────
    def _bus_sum(vec, idx_map):
        d = [0.0] * n
        for i, bus in enumerate(idx_map):
            d[bus] = d[bus] + vec[i]
        return d

    Pd_bus = _bus_sum(Pd_param / S, nd.load_bus_idx)
    Qd_bus = _bus_sum(Qd_param / S, nd.load_bus_idx)
    Pg_bus = _bus_sum(P_g, nd.gen_bus_idx)
    Qg_bus = _bus_sum(Q_g, nd.gen_bus_idx)

    for k in range(n):
        # S_k = sum_m conj(Y[k,m]) * W[k,m]  (complex power injection)
        S_inj = complex(np.conj(Y[k, k])) * V2[k]
        for m in range(n):
            if m == k:
                continue
            y_km = Y[k, m]
            if abs(y_km) < 1e-12:
                continue
            key = (min(k, m), max(k, m))
            if key not in VV:
                # Bus pair not covered by any bag — should not happen for a
                # valid tree decomposition, but guard defensively.
                continue
            S_inj = S_inj + complex(np.conj(y_km)) * _W(k, m)

        constraints += [
            cp.real(S_inj) == Pg_bus[k] - Pd_bus[k],
            cp.imag(S_inj) == Qg_bus[k] - Qd_bus[k],
        ]

    # ── Step 6: voltage magnitude bounds ─────────────────────────────────────
    constraints += [nd.v_min ** 2 <= V2, V2 <= nd.v_max ** 2]
    constraints.append(V2[nd.slack_idx] == nd.v_ref ** 2)

    # ── Step 7: generator limits (per-unit) ──────────────────────────────────
    constraints += [
        nd.pg_min / S <= P_g, P_g <= nd.pg_max / S,
        nd.qg_min / S <= Q_g, Q_g <= nd.qg_max / S,
    ]

    # ── Step 8: objective ─────────────────────────────────────────────────────
    cost_expr = (
        nd.cost_c0.sum()
        + nd.cost_c1 @ (S * P_g)
        + cp.quad_form(P_g, np.diag(nd.cost_c2 * S**2))
    )
    prob = cp.Problem(cp.Minimize(cost_expr), constraints)
    return prob, Pd_param, Qd_param, bag_vars, bags, P_g, Q_g


# ── dispatch ─────────────────────────────────────────────────────────────────

_BUILDERS = {
    "sdp":         _build_sdp_problem,
    "socp":        _build_socp_problem,
    "chordal_sdp": _build_chordal_sdp_problem,
}


def _build_relaxation_problem(relaxation, nd):
    return _BUILDERS[relaxation](nd)


def solve_relaxation(p, args=None):
    """Solve one of the three convex relaxations.

    Parameters
    ----------
    p    : array-like, shape (2 * n_loads,)  — [Pd (MW), Qd (MVar)]
    args : dict, optional
        'relaxation' : one of 'sdp', 'socp', 'dc_opf' / 'lindistflow'
                       (default: 'socp')
        'nd'         : pre-built NetworkData (avoids reloading the network)
        'prob_cache' : dict returned by a prior call — reuses built cvxpy problems
        'case_name'  : fallback if 'nd' is absent
        'solver'     : cvxpy solver name (default depends on relaxation)

    Returns
    -------
    value  : float  — relaxation lower bound on optimal cost ($/hr), or np.nan
    result : dict
        'exact'   : bool — True if solution is likely tight (rank-1 / feasibility)
        'relaxation' : str — which relaxation was solved
    """
    args = args or {}
    relaxation = args.get("relaxation", "socp")

    nd = args.get("nd")
    if nd is None:
        _, nd = load_network(args.get("case_name", DEFAULT_CASE))

    n_loads = nd.n_loads
    Pd = np.asarray(p[:n_loads], dtype=float)
    Qd = np.asarray(p[n_loads:], dtype=float)

    # Use cached problem objects if available to avoid re-building every call.
    cache = args.get("prob_cache")
    if cache is None or relaxation not in cache:
        built = _build_relaxation_problem(relaxation, nd)
        if cache is not None:
            cache[relaxation] = built
    else:
        built = cache[relaxation]

    # SOCP defaults to CLARABEL: a Rust-based primal-dual interior-point solver
    # that exploits sparse cone structure and uses far less memory than SCS
    # (which builds an O((vars+constraints)²) internal matrix) or MOSEK (whose
    # interior-point allocations can OOM-kill on moderate-sized cases).
    # SDP and chordal_sdp still use MOSEK — it's much faster for semidefinite cones.
    _default_solver = cp.CLARABEL if relaxation == "socp" else cp.MOSEK
    solver      = args.get("solver", _default_solver)
    solver_opts = args.get("solver_opts", {})

    def _solve_with_fallback(prob, primary_solver):
        """Try primary solver; fall back to CLARABEL if it raises SolverError.

        MOSEK sometimes raises SolverError instead of returning status='infeasible'.
        CLARABEL is memory-safe and robust for both SOCP and SDP fallback.

        Extra kwargs can be passed via args["solver_opts"].
        """
        try:
            prob.solve(solver=primary_solver, verbose=False, **solver_opts)
        except cp.error.SolverError:
            prob.solve(solver=cp.CLARABEL, verbose=False)

    # Set demand parameters and solve.
    if relaxation == "sdp":
        prob, Pd_param, Qd_param, X, P_g, Q_g = built
        Pd_param.value = Pd
        Qd_param.value = Qd
        _solve_with_fallback(prob, solver)
        value = float(prob.value) if prob.status in ("optimal", "optimal_inaccurate") else np.nan

        # Exactness: rank-1 ↔ second-largest eigenvalue negligible.
        exact = False
        if X.value is not None:
            eigvals = np.sort(np.linalg.eigvalsh(X.value))[::-1]
            exact = bool(eigvals[0] > 1e-8 and eigvals[1] < 1e-4 * eigvals[0])

    elif relaxation == "chordal_sdp":
        prob, Pd_param, Qd_param, bag_vars, bags, P_g, Q_g = built
        Pd_param.value = Pd
        Qd_param.value = Qd
        _solve_with_fallback(prob, solver)
        value = float(prob.value) if prob.status in ("optimal", "optimal_inaccurate") else np.nan

        # Exactness: every bag's second-largest eigenvalue is negligible.
        # This is equivalent to the global rank-1 condition when the relaxation
        # is exact (Rank_Check.m from the MATLAB reference implementation).
        exact = True
        for W_bag in bag_vars:
            if W_bag.value is None:
                exact = False
                break
            if W_bag.value.shape[0] > 1:
                eigs = np.sort(np.linalg.eigvalsh(W_bag.value))[::-1]
                if eigs[0] < 1e-8 or eigs[1] >= 1e-4 * eigs[0]:
                    exact = False
                    break

    else:  # socp
        prob, Pd_param, Qd_param, u, c_var, s_var, P_g, Q_g = built
        Pd_param.value = Pd
        Qd_param.value = Qd
        _solve_with_fallback(prob, solver)
        value = float(prob.value) if prob.status in ("optimal", "optimal_inaccurate") else np.nan

        # Exactness: all Jabr constraints binding ↔ c²+s² == u_k*u_m.
        exact = False
        if u.value is not None and c_var.value is not None:
            gaps = [
                u.value[k] * u.value[m] - c_var.value[e] ** 2 - s_var.value[e] ** 2
                for e, (k, m) in enumerate(zip(nd.branch_from, nd.branch_to))
            ]
            exact = bool(max(gaps) < 1e-4)

    return value, {"exact": exact, "relaxation": relaxation, "status": prob.status}
