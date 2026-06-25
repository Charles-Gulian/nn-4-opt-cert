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
Two relaxations are implemented, each returning a valid lower bound on the
optimal cost:

  SDP          -- Semidefinite relaxation (Lavaei-Low 2012, real 2n x 2n formulation).
                  Lifts V V^T to a 2n x 2n PSD matrix X; exact when rank(X) == 1.

  SOCP         -- Second-order cone relaxation (Jabr 2006).
                  Introduces c_e = Re(V_k conj(V_m)), s_e = Im(V_k conj(V_m)),
                  u_k = |V_k|^2 per branch e=(k,m); exact when c_e^2+s_e^2 = u_k*u_m.

Standard interface
------------------
  solve_relaxation(p, args=None)  ->  (value, result_dict)
  solve_local(p, args=None)       ->  (value, result_dict)

Parameter sampling lives in generate_data.py.
"""

import copy

import numpy as np
import cvxpy as cp
import pandapower as pp

from problems.acopf.network import NetworkData, load_network, DEFAULT_CASE


# ── local solver ──────────────────────────────────────────────────────────────

def solve_local(p, args=None):
    """Solve AC-OPF via pandapower's interior-point solver (IPOPT).

    Parameters
    ----------
    p : array-like, shape (2 * n_loads,)
        Concatenated [Pd (MW), Qd (MVar)] for each row of net.load.
    args : dict, optional
        'net'   : pre-loaded pandapower network (deepcopied before modification)
        'case_name' : fallback if 'net' is absent

    Returns
    -------
    value : float   — optimal total generation cost ($/hr), or np.nan if infeasible
    result : dict   — dispatch and voltage profile on success
    """
    args = args or {}
    net = args.get("net")
    if net is None:
        net, _ = load_network(args.get("case_name", DEFAULT_CASE))
    net = copy.deepcopy(net)

    n_loads = len(net.load)
    net.load["p_mw"]   = p[:n_loads]
    net.load["q_mvar"] = p[n_loads:]

    try:
        pp.runopp(net, numba=False, verbose=False)
        # net.converged tracks runpp (power flow), not OPF; use res_cost instead.
        cost = float(net.res_cost)
        success = np.isfinite(cost)
    except Exception:
        success = False
        cost = np.nan

    if success:
        result = {
            "success": True,
            "pg_mw":   net.res_gen["p_mw"].values.copy(),
            "qg_mvar": net.res_gen["q_mvar"].values.copy(),
            "vm_pu":   net.res_bus["vm_pu"].values.copy(),
            "va_deg":  net.res_bus["va_degree"].values.copy(),
        }
    else:
        cost = np.nan
        result = {"success": False}

    return cost, result


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
    def _bus_sum(vec, idx_map):
        d = [0.0] * n
        for i, k in enumerate(idx_map):
            d[k] = d[k] + vec[i]
        return d

    Pd_bus = _bus_sum(Pd_param / S, nd.load_bus_idx)   # per-unit demand per bus
    Qd_bus = _bus_sum(Qd_param / S, nd.load_bus_idx)
    Pg_bus = _bus_sum(P_g, nd.gen_bus_idx)             # P_g already per-unit
    Qg_bus = _bus_sum(Q_g, nd.gen_bus_idx)

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
    for i in range(nd.n_gens):
        constraints += [
            nd.pg_min[i] / S <= P_g[i], P_g[i] <= nd.pg_max[i] / S,
            nd.qg_min[i] / S <= Q_g[i], Q_g[i] <= nd.qg_max[i] / S,
        ]

    # ── objective: cost with P_g in per-unit ─────────────────────────────
    # c0 + c1*(P_g_pu*S) + c2*(P_g_pu*S)^2
    cost_expr = sum(
        nd.cost_c0[i]
        + nd.cost_c1[i] * S   * P_g[i]
        + nd.cost_c2[i] * S**2 * cp.square(P_g[i])
        for i in range(nd.n_gens)
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
    G   = np.real(nd.Y)
    B   = np.imag(nd.Y)
    S   = nd.baseMVA

    Pd_param = cp.Parameter(nd.n_loads, name="Pd", value=nd.pd_nominal.copy())
    Qd_param = cp.Parameter(nd.n_loads, name="Qd", value=nd.qd_nominal.copy())

    u     = cp.Variable(n,  name="u",     nonneg=True)   # |V|^2 per bus [pu]
    c_var = cp.Variable(nb, name="c")                    # Re(V_k conj(V_m)) [pu]
    s_var = cp.Variable(nb, name="s")                    # Im(V_k conj(V_m)) [pu]
    P_g   = cp.Variable(nd.n_gens, name="P_g_pu")        # generator real power [pu]
    Q_g   = cp.Variable(nd.n_gens, name="Q_g_pu")        # generator reactive power [pu]

    # Build branch lookup: (k,m) → branch index (for k = from-bus, m = to-bus)
    branch_idx = {}
    for e, (k, m) in enumerate(zip(nd.branch_from, nd.branch_to)):
        branch_idx[(k, m)] = e
        branch_idx[(m, k)] = e   # same edge, opposite orientation

    constraints = []

    # ── voltage magnitude bounds ──────────────────────────────────────────
    for k in range(n):
        constraints += [nd.v_min[k] ** 2 <= u[k], u[k] <= nd.v_max[k] ** 2]
    constraints.append(u[nd.slack_idx] == nd.v_ref ** 2)

    # ── SOCP (Jabr) constraints per branch ─────────────────────────────────
    for e, (k, m) in enumerate(zip(nd.branch_from, nd.branch_to)):
        # Rotated-SOC form: ||[2c; 2s; u_k - u_m]||_2 <= u_k + u_m
        constraints.append(
            cp.norm(cp.hstack([2 * c_var[e], 2 * s_var[e], u[k] - u[m]]))
            <= u[k] + u[m]
        )

    # ── power balance at each bus ─────────────────────────────────────────
    # We build the injection expression by iterating over all (k, m) pairs where
    # Y[k,m] ≠ 0.  For (k,m) that correspond to a branch, we use c_var/s_var.
    # For the diagonal (k=k), we use u[k] directly.
    #
    # Re(conj(Y[k,m]) * W[k,m]):
    #   diagonal (m=k):        G[k,k]*u[k]        (Im(W[k,k])=0)
    #   off-diagonal m≠k, branch k→m:  G[k,m]*c[e] + B[k,m]*s[e]
    #   off-diagonal m≠k, branch m→k:  G[k,m]*c[e] − B[k,m]*s[e]   (s_mk = −s_km)

    def _bus_sum(vec, idx_map):
        d = [0.0] * n
        for i, k in enumerate(idx_map):
            d[k] = d[k] + vec[i]
        return d

    Pd_bus = _bus_sum(Pd_param / S, nd.load_bus_idx)   # per-unit demand per bus
    Qd_bus = _bus_sum(Qd_param / S, nd.load_bus_idx)
    Pg_bus = _bus_sum(P_g, nd.gen_bus_idx)             # P_g already per-unit
    Qg_bus = _bus_sum(Q_g, nd.gen_bus_idx)

    for k in range(n):
        P_terms = [G[k, k] * u[k]]
        Q_terms = [-B[k, k] * u[k]]

        for m in range(n):
            if m == k:
                continue
            Gkm = G[k, m]
            Bkm = B[k, m]
            if abs(Gkm) + abs(Bkm) < 1e-12:
                continue

            e = branch_idx.get((k, m))
            if e is None:
                continue

            if nd.branch_from[e] == k:
                P_terms.append(Gkm * c_var[e] + Bkm * s_var[e])
                Q_terms.append(Gkm * s_var[e] - Bkm * c_var[e])
            else:
                P_terms.append(Gkm * c_var[e] - Bkm * s_var[e])
                Q_terms.append(-Gkm * s_var[e] - Bkm * c_var[e])

        constraints += [
            sum(P_terms) == Pg_bus[k] - Pd_bus[k],
            sum(Q_terms) == Qg_bus[k] - Qd_bus[k],
        ]

    # ── generator limits (per-unit) ───────────────────────────────────────
    for i in range(nd.n_gens):
        constraints += [
            nd.pg_min[i] / S <= P_g[i], P_g[i] <= nd.pg_max[i] / S,
            nd.qg_min[i] / S <= Q_g[i], Q_g[i] <= nd.qg_max[i] / S,
        ]

    cost_expr = sum(
        nd.cost_c0[i]
        + nd.cost_c1[i] * S    * P_g[i]
        + nd.cost_c2[i] * S**2 * cp.square(P_g[i])
        for i in range(nd.n_gens)
    )
    prob = cp.Problem(cp.Minimize(cost_expr), constraints)
    return prob, Pd_param, Qd_param, u, c_var, s_var, P_g, Q_g


# ── dispatch ─────────────────────────────────────────────────────────────────

_BUILDERS = {
    "sdp":  _build_sdp_problem,
    "socp": _build_socp_problem,
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

    solver      = args.get("solver", cp.MOSEK)
    solver_opts = args.get("solver_opts", {})

    def _solve_with_fallback(prob, primary_solver):
        """Try primary solver; fall back to SCS if it raises SolverError.

        MOSEK's interior-point method sometimes fails to certify infeasibility
        and raises SolverError instead of returning status='infeasible'.  SCS
        (ADMM-based) is more robust at detecting infeasibility cleanly.

        Extra kwargs (e.g. mosek_params, eps) can be passed via args["solver_opts"].
        """
        try:
            prob.solve(solver=primary_solver, verbose=False, **solver_opts)
        except cp.error.SolverError:
            prob.solve(solver=cp.SCS, verbose=False)

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
