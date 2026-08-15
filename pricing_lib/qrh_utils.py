from __future__ import annotations

import logging
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import gamma as _Gamma, gammainc, ndtr

# one dtype for the whole simulation; float32 halves memory but the rkhs gram
# matrix is ~1e11 conditioned and the iv bisection resolves to 1e-21, so f64
npf64 = np.float64

DELTA_VIX = 30.0 / 365.0
PENALTY = 1e3
BOUNDS = [(0.005, 0.45), (0.05, 3.00), (0.50, 20.0), (1e-5, 0.030)]
PARAM_NAMES = ["H", "nu", "lambda", "c"]
INIT_PARAMS = np.array([0.0964, 0.7620, 6.256, 4.557e-3])
PAPER_RANGE = {"H": (0.0233, 0.2908), "nu": (0.3676, 1.2942),
               "lambda": (5.000, 6.398), "c": (2.308e-3, 18.977e-3)}

# np.trapz renamed to np.trapezoid in numpy 2.0, removed in 2.3
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------- containers

@dataclass
class Smile:
    K: NDArray[np.float64]
    is_call: NDArray[np.bool_]
    iv: NDArray[np.float64]
    vega: NDArray[np.float64]
    F: float
    T: float


@dataclass
class SmoothCurve:
    T: NDArray[np.float64]
    Z: NDArray[np.float64]
    vars_fitted: NDArray[np.float64]
    vars_raw: NDArray[np.float64]
    fit_errs: NDArray[np.float64]
    eps: float
    cond: float
    n_tab: int = 20_000

    def __post_init__(self) -> None:
        self._tab_u = np.linspace(0.0, float(self.T[-1]), self.n_tab)
        self._tab_xi = phi_deriv(self.T[None, :], self._tab_u[:, None]) @ self.Z
        self._xi_end = float(self._tab_xi[-1])

    def __call__(self, u: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        # scalar callers exist (xi_0(0.0)), so promote before interpolating
        u = np.atleast_1d(np.asarray(u, float))
        xi = np.interp(u, self._tab_u, self._tab_xi,
                       left=float(self._tab_xi[0]), right=self._xi_end)
        return np.maximum(xi, 1e-8)


@dataclass
class ObjConfig:
    spx: Smile
    spx_step: int
    vix: list[Smile]
    vix_steps: list[int]
    vix_labels: list[str]
    xi0_fn: SmoothCurve
    n_steps: int
    delta: float
    n_paths: int = 60_000
    n_quad: int = 16
    seed: int = 20240621
    w_vix: float = 1.0
    w_fut: float = 5.0

    @property
    def tau_max(self) -> float:
        return self.n_steps * self.delta


@dataclass
class CalibratedParams:
    H: float
    nu: float
    lmd: float
    c: float
    obj: float
    kappa_l2_sq: float
    v0: float
    elapsed: float

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.H, self.nu, self.lmd, self.c])

    def bound_flags(self) -> dict[str, str]:
        out = {}
        for name, x, (lo, hi) in zip(PARAM_NAMES, self.as_array(), BOUNDS):
            span = hi - lo
            out[name] = ("at lower bound - not identified" if x <= lo + 0.01 * span
                         else "at upper bound - not identified" if x >= hi - 0.01 * span
                         else "ok")
        return out


@dataclass
class KernelParams:
    H: float
    nu: float
    lam: float

    def __post_init__(self) -> None:
        self.H, self.nu, self.lam = float(self.H), float(self.nu), float(self.lam)
        self.alpha = self.H + 0.5
        self.prefac = self.nu / _Gamma(self.alpha)
        self.nu_hat = self.nu * np.sqrt(_Gamma(2.0 * self.H)) / _Gamma(self.alpha)
        self.r = self.nu_hat ** 2 / (2.0 * self.lam) ** (2.0 * self.H)

    @property
    def admissible(self) -> bool:
        return self.r < 1.0

    @property
    def memory_days(self) -> float:
        return 365.0 / self.lam

    def __repr__(self) -> str:
        return (f"KernelParams(H={self.H:.4f}, nu={self.nu:.4f}, "
                f"lam={self.lam:.4f}, ||kappa||^2={self.r:.4f}, "
                f"memory={self.memory_days:.0f}d)")


@dataclass(frozen=True)
class RunParams:
    data_path: str
    quote_dt: str
    expiry_dt: str
    n_paths: int
    workers: int
    tag: Optional[str]

    @classmethod
    def add_args(cls, ap: ArgumentParser) -> None:
        ap.add_argument("--data-path", default="../Data/spx_vix_snapshot.csv")
        ap.add_argument("--quote-dt", default="2024-06-21")
        ap.add_argument("--expiry-dt", default="2024-07-17")
        ap.add_argument("--n-paths", type=int, default=60_000)
        ap.add_argument("--workers", type=int, default=1)
        ap.add_argument("--tag", default=None)


def init_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def black_scholes_px(F: float,
                     K: NDArray[np.float64],
                     T: float,
                     sig: NDArray[np.float64],
                     is_call: NDArray[np.bool_]) -> NDArray[np.float64]:
    sq = np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / (sig * sq) + 0.5 * sig * sq
        d2 = d1 - sig * sq
    call = F * ndtr(d1) - K * ndtr(d2)
    # put is computed from call via put call parity
    return np.where(is_call, call, call - F + K)


def black_vega(F: float,
               K: NDArray[np.float64],
               T: float,
               sig: NDArray[np.float64]) -> NDArray[np.float64]:
    sq = np.sqrt(T)
    d1 = np.log(F / K) / (sig * sq) + 0.5 * sig * sq
    return F * np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi) * sq


def implied_vols(prices: NDArray[np.float64],
                 F: float,
                 K: NDArray[np.float64],
                 T: float,
                 is_call: NDArray[np.bool_],
                 n_iter: int = 72,
                 lb: float = 1e-4,
                 ub: float = 8.0) -> NDArray[np.float64]:
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = np.where(is_call, F, K)
    valid = np.isfinite(prices) & (prices > intrinsic + 1e-12) & (prices < upper - 1e-12)

    a, b = np.full(K.shape, lb), np.full(K.shape, ub)
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        too_low = black_scholes_px(F, K, T, m, is_call) < prices
        a = np.where(too_low, m, a)
        b = np.where(too_low, b, m)
    return np.where(valid, 0.5 * (a + b), np.nan)


def mid(d: pd.DataFrame) -> pd.Series:
    return 0.5 * (d["bid_1545"] + d["ask_1545"])


def get_fwd_and_discount(quote_data: pd.DataFrame) -> tuple[float, float]:
    pivot = quote_data.pivot_table(index="strike", columns="option_type",
                                   values="mid", aggfunc="first").dropna()
    pivot = pivot[(pivot["C"] > 0) & (pivot["P"] > 0)]
    if len(pivot) < 3:
        raise ValueError("not enough two-sided strikes for put-call parity")
    strikes = pivot.index.values.astype(float)
    call_put_diff = (pivot["C"] - pivot["P"]).values
    strike_near_atm = strikes[np.argmin(np.abs(call_put_diff))]
    mask = (strikes > 0.75 * strike_near_atm) & (strikes < 1.35 * strike_near_atm)
    if mask.sum() < 3:
        mask = np.ones_like(strikes, dtype=bool)
    # C - P = D*(F - K), so slope = -D and intercept = D*F
    slope, intercept = np.polyfit(strikes[mask], call_put_diff[mask], 1)
    discount = -slope
    return float(intercept / discount), float(discount)


def var_swap_single(k: NDArray[np.float64],
                    vol: NDArray[np.float64],
                    T: float) -> tuple[float, dict]:
    sigma = vol * np.sqrt(T)
    d2 = -k / sigma - sigma / 2.0
    y = ndtr(d2)

    order = np.argsort(y)
    y_s, w_s = y[order], sigma[order] ** 2

    # pchip needs strictly increasing abscissae
    keep = np.concatenate(([True], np.diff(y_s) > 1e-14))
    y_s, w_s = y_s[keep], w_s[keep]
    if y_s.size < 3:
        return np.nan, {}

    # w(T) = integral_0^1 w_bs(y) dy, y = N(d2); tails held flat in variance
    interior = quad(PchipInterpolator(y_s, w_s), y_s[0], y_s[-1], limit=200)[0]
    left = w_s[0] * y_s[0]
    right = w_s[-1] * (1.0 - y_s[-1])
    total = interior + left + right
    return total, dict(y_min=float(y_s[0]), y_max=float(y_s[-1]),
                       tail_frac=float((left + right) / total))


def var_swap_curve(data: pd.DataFrame, quote_date: pd.Timestamp,
                   symbol: str = "^SPX", t_min: float = 3.0 / 365.0,
                   min_bid: float = 0.05, max_rel_spread: float = 0.75) -> pd.DataFrame:
    df_all = data[data.underlying_symbol == symbol].copy()
    df_all["mid"] = mid(df_all)

    rows = []
    for exp, df in df_all.groupby("expiration"):
        tau = float((pd.Timestamp(exp) - quote_date).days / 365.0)
        if tau < t_min:
            continue
        try:
            forward, discount = get_fwd_and_discount(df)
        except Exception:
            continue

        otm = pd.concat([
            df[(df.option_type == "P") & (df.strike < forward)],
            df[(df.option_type == "C") & (df.strike >= forward)],
        ]).sort_values("strike")
        otm = otm[(otm["bid_1545"] >= min_bid) & (otm["ask_1545"] > 0)]
        otm = otm[((otm["ask_1545"] - otm["bid_1545"]) / otm["mid"]) < max_rel_spread]
        otm = otm[otm["implied_volatility_1545"] > 1e-3]
        if len(otm) < 8:
            continue

        strikes = otm["strike"].values.astype(float)
        is_call = (otm["option_type"].values == "C")
        log_moneyness = np.log(strikes / forward)
        vol_mid = otm["implied_volatility_1545"].values.astype(float)

        # snapshot has one iv column (mid), so re-imply bid/ask vols from prices
        vol_bid = implied_vols(otm["bid_1545"].values / discount, forward, strikes, tau, is_call)
        vol_ask = implied_vols(otm["ask_1545"].values / discount, forward, strikes, tau, is_call)
        ok = np.isfinite(vol_bid) & np.isfinite(vol_ask)
        vol_bid = np.where(ok, vol_bid, vol_mid)
        vol_ask = np.where(ok, vol_ask, vol_mid)

        w_mid, parts = var_swap_single(log_moneyness, vol_mid, tau)
        if not np.isfinite(w_mid):
            continue
        w_bid, _ = var_swap_single(log_moneyness, vol_bid, tau)
        w_ask, _ = var_swap_single(log_moneyness, vol_ask, tau)

        rows.append(dict(expiration=exp, tau=tau, days=int(round(tau * 365)),
                         F=forward, D=discount, n_options=len(otm), w_mid=w_mid,
                         vs_mid=w_mid / tau, vs_bid=w_bid / tau,
                         vs_ask=w_ask / tau, **parts))

    out = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"no usable {symbol} expiries found")
    out["vs_vol_mid"] = np.sqrt(np.maximum(out["vs_mid"], 0))
    out["quote_width_vol"] = 0.5 * (np.sqrt(np.maximum(out["vs_ask"], 0))
                                    - np.sqrt(np.maximum(out["vs_bid"], 0)))
    return out


def phi(tau: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    m = np.minimum(x, tau)
    return 1.0 - m ** 3 / 6.0 + x * tau * (2.0 + m) / 2.0


def phi_deriv(tau: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    m = np.minimum(x, tau)
    return tau - m ** 2 / 2.0 + tau * m


def var_curve_smooth(expiries: NDArray[np.float64],
                     tot_vars: NDArray[np.float64],
                     eps: Union[float, NDArray[np.float64]] = 0.03,
                     ridge: float = 1e-10) -> SmoothCurve:
    n = expiries.size
    # eps may be one allowance for all expiries or one per expiry
    eps_vec = np.full(n, float(eps)) if np.isscalar(eps) else eps

    A = phi(expiries[None, :], expiries[:, None])
    # gram matrix is ~1e11 conditioned over 5y of expiries, so regularise before solving
    A = A + ridge * np.trace(A) / n * np.eye(n)
    cond = float(np.linalg.cond(A))

    # perturbation is in vol units: (s+e)^2 T ~ w + 2 sqrt(w) @ sqrt(T)
    scale = 2.0 * np.sqrt(tot_vars) * np.sqrt(expiries)

    def obj_and_grad(err: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        v = tot_vars + scale * err
        z = np.linalg.solve(A, v)
        return float(v @ z), 2.0 * z * scale

    res = minimize(obj_and_grad, np.zeros(n), jac=True, method="L-BFGS-B",
                   bounds=list(zip(-eps_vec, eps_vec)),
                   options=dict(maxiter=500, ftol=1e-14))

    fitted = tot_vars + scale * res.x
    return SmoothCurve(T=expiries, Z=np.linalg.solve(A, fitted),
                       vars_fitted=fitted, vars_raw=tot_vars,
                       fit_errs=np.sqrt(fitted / expiries) - np.sqrt(tot_vars / expiries),
                       eps=float(np.max(eps_vec)), cond=cond)



def kappa(tau: Union[float, NDArray[np.float64]],
          p: KernelParams) -> NDArray[np.float64]:
    # scalar callers exist (delta, DELTA_VIX), so promote before masking
    tau = np.asarray(tau, float)
    out = np.zeros_like(tau)
    m = tau > 0.0
    out[m] = p.prefac * tau[m] ** (p.alpha - 1.0) * np.exp(-p.lam * tau[m])
    return out


def K00(tau: Union[float, NDArray[np.float64]],
        p: KernelParams) -> NDArray[np.float64]:
    tau = np.asarray(tau, float)
    out = np.zeros_like(tau)
    m = tau > 0.0
    out[m] = p.r * gammainc(2.0 * p.H, 2.0 * p.lam * tau[m])
    return out


def K0_kappa(tau: Union[float, NDArray[np.float64]],
             p: KernelParams) -> NDArray[np.float64]:
    tau = np.asarray(tau, float)
    out = np.zeros_like(tau)
    m = tau > 0.0
    out[m] = p.nu * p.lam ** (-p.alpha) * gammainc(p.alpha, p.lam * tau[m])
    return out


def K0_resolvent(tau: Union[float, NDArray[np.float64]],
                 p: KernelParams,
                 n_max: int = 512,
                 tol: float = 1e-14) -> NDArray[np.float64]:
    tau = np.asarray(tau, float)
    out = np.zeros_like(tau)
    m = tau > 0.0
    if not m.any():
        return out
    # series form; trapz on tau^(2H-1) diverges and is 27x too big at H=0.02
    x = 2.0 * p.lam * tau[m]
    acc = np.zeros_like(x)
    rn = 1.0
    for n in range(1, n_max + 1):
        rn *= p.r
        if rn < tol:
            break
        term = rn * gammainc(2.0 * p.H * n, x)
        acc += term
        if term.max() < tol:
            break
    out[m] = acc
    return out


def b_star(n_steps: int, delta: float, p: KernelParams) -> NDArray[np.float64]:
    edges = K00(np.arange(0, n_steps + 2, dtype=float) * delta, p)
    out = np.zeros(n_steps + 2)
    out[1:] = np.sqrt(np.maximum(np.diff(edges), 0.0) / delta)
    return out


def a_coeff(u_vals: NDArray[np.float64],
            n_steps: int,
            delta: float,
            p: KernelParams) -> NDArray[np.float64]:
    k_edge = np.arange(0, n_steps + 1, dtype=float) * delta
    vals = K00(u_vals[:, None] - k_edge[None, :], p)
    a_sq = (vals[:, :-1] - vals[:, 1:]) / delta
    return np.sqrt(np.maximum(a_sq, 0.0)).T



def impute_y0(horizons: NDArray[np.float64],
              xi0_fn: SmoothCurve,
              p: KernelParams,
              c: float,
              n_cells: int = 240) -> tuple[NDArray[np.float64], float]:
    # y0(u)^2 = xi_0(u) - var[Y_u] - c, with var[Y_u] = int_0^u xi_0(u-v) kappa(v)^2 dv
    # integrated over the lag v = u - s, so v = 0 is now and v = u is back at time 0
    cell_fractions = np.linspace(0.0, 1.0, n_cells + 1)
    lag_edges = horizons[:, None] * cell_fractions[None, :]
    # exact kappa^2 mass per cell, not width x kappa^2(mid): kappa^2 ~ v^(2H-1) is
    # singular at v = 0 but its integral K00 is finite and known in closed form
    cell_kernel_mass = np.diff(K00(lag_edges, p), axis=1)
    lag_midpoints = 0.5 * (lag_edges[:, :-1] + lag_edges[:, 1:])
    fwd_variance_at_lag = xi0_fn((horizons[:, None] - lag_midpoints).ravel())
    y_variance = np.sum(cell_kernel_mass
                        * fwd_variance_at_lag.reshape(lag_midpoints.shape), axis=1)

    fwd_variance = xi0_fn(horizons)
    y0_squared = fwd_variance - y_variance - c
    # y0^2 < 0 means no real y0 reproduces the market curve here; flooring it inflates
    # V, so report the shortfall (scale-free, and a magnitude so the optimiser has a slope)
    y0_violation = float(np.mean(np.maximum(-y0_squared, 0.0))
                         / max(np.mean(fwd_variance), 1e-12))
    # positive root: Y is minus the weighted past return, so it rises when the market falls
    return np.sqrt(np.maximum(y0_squared, 0.0)), y0_violation


def simulate_qrh(p: KernelParams,
                 c: float,
                 y0_curve: NDArray[np.float64],
                 T: float,
                 n_steps: int,
                 n_paths: int,
                 seed: int,
                 record_steps: Sequence[int] = (),
                 xi_target: Optional[NDArray[np.float64]] = None
                 ) -> tuple[dict[int, NDArray[np.float64]], NDArray[np.float64]]:
    delta = T / n_steps
    # theta of Bourgey-Gatheral sec 3 (from the HQE scheme, Gatheral 2022 appx A)
    variance_blend = npf64(1.0 / (2.0 * p.H + 1.0))
    min_variance, one = npf64(c), npf64(1.0)
    # not the model's floor c -- just guards sqrt() against rounding to a tiny negative
    numerical_floor, drift_weight = npf64(1e-10), npf64(0.25 * delta)

    bstar_weights = b_star(n_steps, delta, p)
    # reversed copy so the per-step weights are a contiguous forward slice (blas-friendly)
    bstar_weights_rev = np.ascontiguousarray(bstar_weights[::-1], dtype=npf64)
    n_weights = bstar_weights.size

    kappa_integral, kappa_sq_integral = float(K0_kappa(delta, p)), float(K00(delta, p))
    shock_loading = npf64(kappa_integral / delta)
    # sigma_chi = sqrt(V_bar * delta) and sigma_eps = sqrt(V_bar * [K00 - K0^2/delta]);
    # the sqrt(V_bar) factor is applied per step below, these are the constant parts
    sqrt_delta = npf64(np.sqrt(delta))
    resid_vol_unit = npf64(np.sqrt(kappa_sq_integral - kappa_integral ** 2 / delta))

    rng = np.random.default_rng(seed)
    z_return = rng.standard_normal((n_steps, n_paths), dtype=npf64)
    z_kernel = rng.standard_normal((n_steps, n_paths), dtype=npf64)

    target_variance = xi_target

    return_shocks = np.zeros((n_steps, n_paths), npf64)
    log_spot = np.zeros(n_paths, npf64)
    y_prev = np.full(n_paths, y0_curve[0], npf64)
    variance_prev = y_prev ** 2 + min_variance
    spx_by_step, record_at = {}, set(int(s) for s in record_steps)

    for n in range(1, n_steps + 1):
        if n == 1:
            y_forecast = np.full(n_paths, y0_curve[1], npf64)
        else:
            y_forecast = y0_curve[n] + (bstar_weights_rev[n_weights - 1 - n: n_weights - 2] @ return_shocks[:n - 1])

        cond_variance = (variance_blend * y_forecast ** 2
                         + (one - variance_blend) * y_prev ** 2 + min_variance)
        np.maximum(cond_variance, numerical_floor, out=cond_variance)
        if target_variance is not None:
            # hqe conditional variance undershoots E[V_t]=xi_0(t) by 2-4%, independent of n_steps
            cond_variance *= npf64(min(max(
                target_variance[n] / max(float(cond_variance.mean()), 1e-12), 0.5), 2.0))
        cond_vol = np.sqrt(cond_variance, out=cond_variance)

        shock = cond_vol * (sqrt_delta * z_return[n - 1])
        y_now = y_forecast
        y_now += shock_loading * shock
        y_now += cond_vol * (resid_vol_unit * z_kernel[n - 1])
        variance_now = y_now * y_now + min_variance

        log_spot -= drift_weight * (variance_now + variance_prev) + shock
        return_shocks[n - 1] = shock
        y_prev, variance_prev = y_now, variance_now
        if n in record_at:
            spx_by_step[n] = np.exp(log_spot)

    return spx_by_step, return_shocks


def vix_at_T(chi: NDArray[np.float64],
             p: KernelParams,
             c: float,
             T: float,
             n_steps: int,
             xi0_fn: SmoothCurve,
             n_quad: int = 16) -> tuple[NDArray[np.float64], float]:
    delta = T / n_steps
    x, wq = np.polynomial.legendre.leggauss(n_quad)
    u = 0.5 * DELTA_VIX * (x + 1.0)
    wq = 0.5 * DELTA_VIX * wq

    y0_fwd, y0_violation = impute_y0(T + u, xi0_fn, p, c)
    a = a_coeff(T + u, n_steps, delta, p)
    y_fwd = y0_fwd[:, None] + a.T @ chi
    integ = (y_fwd ** 2 + c) * (1.0 + K0_resolvent(DELTA_VIX - u, p))[:, None]
    return np.sqrt(np.maximum((wq @ integ) / DELTA_VIX, 0.0)) * 100.0, y0_violation


def mc_option_prices(samples: NDArray[np.float64],
                     strikes: NDArray[np.float64],
                     is_call: NDArray[np.bool_]) -> NDArray[np.float64]:
    # sorting makes the in-the-money paths a contiguous block, so a prefix sum answers
    # every strike at once: E[(S-K)+] = (sum of the block above K - K * its size) / n_paths
    sorted_samples = np.sort(samples)
    n_paths = sorted_samples.size
    # leading zero so running_sum[j] is the sum of the first j samples, incl. j = 0
    running_sum = np.concatenate(([0.0], np.cumsum(sorted_samples)))
    n_below = np.searchsorted(sorted_samples, strikes, side="right")
    call_prices = (running_sum[n_paths] - running_sum[n_below]
                   - strikes * (n_paths - n_below)) / n_paths
    put_prices = (strikes * n_below - running_sum[n_below]) / n_paths
    return np.where(is_call, call_prices, put_prices)


def mc_implied_vols(samples: NDArray[np.float64],
                    smile: Smile) -> tuple[NDArray[np.float64], float]:
    # mean(samples) as the forward makes call - put = mean(S) - K hold exactly,
    # so calls and puts imply the same vol at a shared strike
    model_forward = float(np.mean(samples))
    prices = mc_option_prices(samples, smile.K, smile.is_call)
    model_iv = implied_vols(prices, model_forward, smile.K, smile.T, smile.is_call)
    return model_iv, model_forward


# ---------------------------------------------------------------- objective

def vega_weighted_mse(model_iv: NDArray[np.float64],
                      smile: Smile) -> Optional[float]:
    # mean squared model-vs-market iv error over one smile, weighted by vega
    priced_ok = np.isfinite(model_iv)
    if priced_ok.sum() < 4:
        return None
    vega_weights = smile.vega[priced_ok] / smile.vega[priced_ok].sum()
    iv_errors = model_iv[priced_ok] - smile.iv[priced_ok]
    return float(np.sum(vega_weights * iv_errors ** 2))


def simulate_spx_and_vix(params: NDArray[np.float64],
                         cfg: ObjConfig
                         ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]], float]:
    kernel = KernelParams(*params[:3])
    min_variance = params[3]
    time_grid = np.arange(cfg.n_steps + 1) * cfg.delta
    y0_curve, grid_violation = impute_y0(time_grid, cfg.xi0_fn, kernel, min_variance)

    spx_by_step, return_shocks = simulate_qrh(
        kernel, min_variance, y0_curve, cfg.tau_max, cfg.n_steps, cfg.n_paths,
        cfg.seed, record_steps=[cfg.spx_step], xi_target=cfg.xi0_fn(time_grid))

    # one entry for the simulation grid, one per expiry's forward grid T_i + u
    vix_samples, y0_violations = [], [grid_violation]
    for vix_step in cfg.vix_steps:
        # every expiry is read off the same paths; slicing the shocks keeps delta consistent
        samples, fwd_violation = vix_at_T(return_shocks[:vix_step], kernel, min_variance,
                                          vix_step * cfg.delta, vix_step,
                                          cfg.xi0_fn, cfg.n_quad)
        vix_samples.append(samples)
        y0_violations.append(fwd_violation)

    return spx_by_step[cfg.spx_step], vix_samples, float(np.sum(y0_violations))


def objective(params: NDArray[np.float64],
              cfg: ObjConfig,
              detail: bool = False) -> Union[float, dict]:
    kernel = KernelParams(*params[:3])
    if not kernel.admissible:
        return PENALTY * (1.0 + kernel.r)
    try:
        spx_samples, vix_samples, y0_violation = simulate_spx_and_vix(params, cfg)

        spx_model_iv, spx_model_fwd = mc_implied_vols(spx_samples, cfg.spx)
        spx_error = vega_weighted_mse(spx_model_iv, cfg.spx)
        if spx_error is None:
            return PENALTY

        vix_errors, vix_model_ivs, vix_model_fwds, future_errors = [], [], [], []
        for smile, samples in zip(cfg.vix, vix_samples):
            model_iv, model_fwd = mc_implied_vols(samples, smile)
            smile_error = vega_weighted_mse(model_iv, smile)
            if smile_error is None:
                return PENALTY
            vix_errors.append(smile_error)
            vix_model_ivs.append(model_iv)
            vix_model_fwds.append(model_fwd)
            future_errors.append((model_fwd / smile.F - 1.0) ** 2)

        vix_error = float(np.mean(vix_errors))
        future_error = float(np.mean(future_errors))
        y0_penalty = 10.0 * y0_violation
        total_error = (spx_error + cfg.w_vix * vix_error
                       + cfg.w_fut * future_error + y0_penalty)

        if detail:
            return dict(F=total_error, r_spx=spx_error, r_vix=vix_error,
                        r_fut=future_error, pen=y0_penalty,
                        iv_spx=spx_model_iv, iv_vix=vix_model_ivs,
                        fut_vix=vix_model_fwds, F_spx=spx_model_fwd,
                        S_spx=spx_samples, vix_sims=vix_samples)
        return float(total_error)
    except Exception:
        return PENALTY


def model_vix_spot(params: NDArray[np.float64], cfg: ObjConfig) -> float:
    no_shocks = np.zeros((1, 1))
    vix, _ = vix_at_T(no_shocks, KernelParams(*params[:3]), params[3],
                      1e-8, 1, cfg.xi0_fn, n_quad=48)
    return float(vix[0])


def rmse(model_iv: NDArray[np.float64], mkt_iv: NDArray[np.float64]) -> float:
    m = np.isfinite(model_iv)
    return float(np.sqrt(np.mean((model_iv[m] - mkt_iv[m]) ** 2)))
