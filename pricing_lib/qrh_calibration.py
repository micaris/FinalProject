from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from pricing_lib.qrh_utils import (
    BOUNDS, DELTA_VIX, INIT_PARAMS, PARAM_NAMES, PAPER_RANGE,
    CalibratedParams, KernelParams, ObjConfig, RunParams, Smile,
    K0_resolvent, black_vega, get_fwd_and_discount, implied_vols, init_logging,
    mid, model_vix_spot, objective, rmse, var_curve_smooth, var_swap_curve,
)

N_VIX = 4
MIN_OPTIONS = 8
N_STEPS = 1               # stage 1, differential evolution
N_PATHS = 15_000
N_STEPS_POLISH = 2        # stage 2; polish path count comes from --n-paths
MAX_ITER = 30
POP_SIZE = 10
POLISH_ITER = 50
EPS_BOUNDS = (0.002, 0.03)
MARKET_VIX = 13.20        # VIX index at 15:45 on the quote date, from the snapshot


class QRH_Calibration:

    def __init__(self, run_params: RunParams) -> None:
        self.run_params = run_params
        self.quote_date = pd.Timestamp(run_params.quote_dt)
        self.tag = run_params.tag or f"qrh_{run_params.expiry_dt.replace('-', '')}"
        self.data: pd.DataFrame = pd.DataFrame()
        self.var_swaps: Optional[pd.DataFrame] = None
        self.fwd_var_fn = None
        self.vix_expiries: list[str] = []
        self.spx_smile: Optional[Smile] = None
        self.vix_smiles: list[Smile] = []
        self.obj_config: Optional[ObjConfig] = None
        self.calibrated: Optional[CalibratedParams] = None

    def build_fwd_var_curve(self) -> None:
        vs = var_swap_curve(self.data, self.quote_date, "^SPX")
        # allowance per expiry from its own re-implied bid/ask width
        eps = np.clip(vs["quote_width_vol"].values, *EPS_BOUNDS)
        self.fwd_var_fn = var_curve_smooth(vs.tau.values, vs.w_mid.values, eps=eps)
        self.var_swaps = vs
        v0 = float(self.fwd_var_fn(0.0)[0])
        logging.info(f"fwd var curve: {len(vs)} knots, cond={self.fwd_var_fn.cond:.2e}, "
                     f"V_0={v0:.6f} ({np.sqrt(v0) * 100:.2f}% vol)")

    def build_smile(self, symbol: str, expiry: str, k_lo: float, k_hi: float,
                    min_bid: float = 0.05, max_spread: float = 0.60) -> Smile:
        tau = float((pd.Timestamp(expiry) - self.quote_date).days / 365.0)
        quotes = self.data[(self.data.underlying_symbol == symbol)
                           & (self.data.expiration == expiry)].copy()
        quotes["mid"] = mid(quotes)
        forward, discount = get_fwd_and_discount(quotes)

        otm = pd.concat([
            quotes[(quotes.option_type == "P") & (quotes.strike < forward)],
            quotes[(quotes.option_type == "C") & (quotes.strike >= forward)],
        ]).sort_values("strike")
        otm = otm[(otm["bid_1545"] >= min_bid) & (otm["ask_1545"] > 0)]
        otm = otm[((otm["ask_1545"] - otm["bid_1545"]) / otm["mid"]) < max_spread]
        otm = otm[(otm["strike"] >= k_lo * forward) & (otm["strike"] <= k_hi * forward)]

        strikes = otm["strike"].values.astype(float)
        is_call = (otm["option_type"].values == "C")
        iv = implied_vols(otm["mid"].values / discount, forward, strikes, tau, is_call)

        valid = np.isfinite(iv) & (iv > 1e-3)
        strikes, is_call, iv = strikes[valid], is_call[valid], iv[valid]

        # normalise so spx strikes are comparable to the simulated S_T/S_0
        if symbol == "^SPX":
            strikes, forward = strikes / forward, 1.0
        return Smile(K=strikes, is_call=is_call, iv=iv,
                     vega=black_vega(forward, strikes, tau, iv), F=forward, T=tau)

    def build_smiles(self) -> None:
        self.spx_smile = self.build_smile("^SPX", self.run_params.expiry_dt, 0.85, 1.10)
        # check vix expiries and retain the first N_VIX with a usable smile.
        self.vix_smiles, self.vix_expiries = [], []
        for expiry in sorted(self.data[self.data.underlying_symbol == "^VIX"].expiration.unique()):
            if pd.Timestamp(expiry) <= self.quote_date:
                continue
            # weekly vix expiries are near-empty here; the quote count selects the monthlies
            smile = self.build_smile("^VIX", expiry, 0.60, 2.50)
            if len(smile.K) >= MIN_OPTIONS:
                self.vix_smiles.append(smile)
                self.vix_expiries.append(expiry)
                if len(self.vix_expiries) == N_VIX:
                    break
        logging.info(f"vix expiries: {', '.join(self.vix_expiries)}")

    def build_objective_config(self) -> None:
        self.obj_config = self.make_config(N_STEPS_POLISH, self.run_params.n_paths)

    def make_config(self, steps_per_day: int, n_paths: int) -> ObjConfig:
        def days(e):
            return (pd.Timestamp(e) - self.quote_date).days

        spx_expiry = self.run_params.expiry_dt
        max_days = max(days(e) for e in [spx_expiry] + self.vix_expiries)
        n_steps = max_days * steps_per_day
        # grid unit is a whole-day divisor so every expiry lands exactly on a step
        return ObjConfig(
            spx=self.spx_smile, spx_step=days(spx_expiry) * steps_per_day,
            vix=self.vix_smiles,
            vix_steps=[days(e) * steps_per_day for e in self.vix_expiries],
            vix_labels=self.vix_expiries, xi0_fn=self.fwd_var_fn,
            n_steps=n_steps, delta=(max_days / 365.0) / n_steps, n_paths=n_paths)

    def calibrate(self) -> CalibratedParams:
        start = time.time()

        coarse = self.make_config(N_STEPS, N_PATHS)
        rng = np.random.default_rng(7)
        init = rng.uniform([b[0] for b in BOUNDS], [b[1] for b in BOUNDS],
                           size=(POP_SIZE * len(BOUNDS), len(BOUNDS)))
        init[0] = np.clip(INIT_PARAMS, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])

        logging.info(f"stage 1 differential evolution: {coarse.n_steps} steps, "
                     f"{coarse.n_paths:,} paths")
        de = differential_evolution(
            objective, bounds=BOUNDS, args=(coarse,), init=init,
            maxiter=MAX_ITER, tol=1e-6, mutation=(0.4, 1.0), recombination=0.75,
            seed=7, polish=False, workers=self.run_params.workers,
            updating="deferred" if self.run_params.workers != 1 else "immediate")

        logging.info(f"stage 2 polish: {self.obj_config.n_steps} steps, "
                     f"{self.obj_config.n_paths:,} paths (from F={de.fun:.6f})")
        # common random numbers make the surface deterministic, so a local method works
        polish = minimize(objective, de.x, args=(self.obj_config,),
                          method="Nelder-Mead", bounds=BOUNDS,
                          options=dict(maxiter=POLISH_ITER, xatol=1e-4, fatol=1e-10))

        x, fun = (polish.x, polish.fun) if polish.fun < de.fun else (de.x, de.fun)
        p = KernelParams(*x[:3])
        self.calibrated = CalibratedParams(
            H=float(x[0]), nu=float(x[1]), lmd=float(x[2]), c=float(x[3]),
            obj=float(fun), kappa_l2_sq=p.r, v0=float(self.fwd_var_fn(0.0)[0]),
            elapsed=time.time() - start)
        return self.calibrated

    def summary(self) -> dict:
        cal = self.calibrated
        detail = objective(cal.as_array(), self.obj_config, detail=True)
        flags = cal.bound_flags()

        lines = ["calibrated", "=" * 66]
        for name, val in zip(PARAM_NAMES, cal.as_array()):
            lo, hi = PAPER_RANGE[name]
            note = "" if flags[name] == "ok" else f"   <<< {flags[name]}"
            lines.append(f"  {name:<7s} = {val:<11.6f} (paper {lo:.4f} - {hi:.4f}){note}")
        lines += [
            f"  {'||k||^2':<7s} = {cal.kappa_l2_sq:<11.6f}",
            f"  V_0     = {cal.v0:<11.6f} -> {np.sqrt(cal.v0) * 100:.2f}% instantaneous vol (market input)",
            f"  VIX_0   = {model_vix_spot(cal.as_array(), self.obj_config):<11.2f} "
            f"vs {MARKET_VIX:.2f} market   (30d average of xi_0, not sqrt(V_0))",
            f"  F       = {cal.obj:.6e}  [spx {detail['r_spx']:.2e} | "
            f"vix {detail['r_vix']:.2e} | fut {detail['r_fut']:.2e} | pen {detail['pen']:.2e}]",
            f"  E[S_T]  = {detail['F_spx']:.6f}",
            f"  SPX {self.obj_config.spx.T * 365:>4.0f}d  rmse "
            f"{rmse(detail['iv_spx'], self.obj_config.spx.iv) * 100:5.2f}%   "
            f"({len(self.obj_config.spx.K)} opts)",
        ]
        for lab, sm, iv, fm in zip(self.vix_expiries, self.obj_config.vix,
                                   detail["iv_vix"], detail["fut_vix"]):
            lines.append(f"  VIX {sm.T * 365:>4.0f}d  rmse {rmse(iv, sm.iv) * 100:5.2f}%   "
                         f"({len(sm.K):3d} opts)  future {fm:6.3f} vs {sm.F:6.3f}  [{lab}]")
        lines += [f"  {cal.elapsed:.1f}s", "=" * 66]
        logging.info("\n".join(lines))

        for name, flag in flags.items():
            if flag != "ok":
                logging.warning(f"{name} {flag}; objective was still improving in that direction")
        return detail

    def write_outputs(self, detail: dict) -> None:
        cal = self.calibrated
        p = KernelParams(cal.H, cal.nu, cal.lmd)
        meta = dict(
            run=dict(timestamp=datetime.now().isoformat(timespec="seconds"),
                     quote_date=self.run_params.quote_dt,
                     spx_expiry=self.run_params.expiry_dt,
                     vix_expiries=self.vix_expiries,
                     n_steps=self.obj_config.n_steps, n_paths=self.obj_config.n_paths,
                     wall_clock_s=round(cal.elapsed, 1)),
            parameters=dict(H=cal.H, nu=cal.nu, **{"lambda": cal.lmd}, c=cal.c),
            bound_flags=cal.bound_flags(),
            derived=dict(kappa_l2_sq=cal.kappa_l2_sq, V_0=cal.v0,
                         sqrt_V_0_pct=float(np.sqrt(cal.v0) * 100),
                         K0_resolvent_30d=float(np.atleast_1d(K0_resolvent(DELTA_VIX, p))[0]),
                         memory_days=p.memory_days),
            objective=dict(F=cal.obj, spx=detail["r_spx"], vix=detail["r_vix"],
                           fut=detail["r_fut"], penalty=detail["pen"]),
            fit=dict(SPX=dict(rmse_iv=rmse(detail["iv_spx"], self.obj_config.spx.iv),
                              n=len(self.obj_config.spx.K)),
                     VIX={lab: dict(rmse_iv=rmse(iv, sm.iv), n=len(sm.K),
                                    future_model=float(fm), future_market=float(sm.F))
                          for lab, sm, iv, fm in zip(self.vix_expiries, self.obj_config.vix,
                                                     detail["iv_vix"], detail["fut_vix"])}))
        with open(f"{self.tag}_params.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        rows = [pd.DataFrame(dict(leg="SPX", expiry=self.run_params.expiry_dt,
                                  T=self.obj_config.spx.T, strike=self.obj_config.spx.K,
                                  moneyness=self.obj_config.spx.K / self.obj_config.spx.F,
                                  mkt_iv=self.obj_config.spx.iv, model_iv=detail["iv_spx"]))]
        rows += [pd.DataFrame(dict(leg="VIX", expiry=lab, T=sm.T, strike=sm.K,
                                   moneyness=sm.K / sm.F, mkt_iv=sm.iv, model_iv=iv))
                 for lab, sm, iv in zip(self.vix_expiries, self.obj_config.vix, detail["iv_vix"])]
        out = pd.concat(rows)
        out["err_bp"] = (out.model_iv - out.mkt_iv) * 1e4
        out.to_csv(f"{self.tag}_smiles.csv", index=False)

        self.var_swaps.to_csv(f"{self.tag}_varswaps.csv", index=False)
        logging.info(f"wrote {self.tag}_params.json, _smiles.csv, _varswaps.csv")

    def run(self) -> CalibratedParams:
        self.data = pd.read_csv(self.run_params.data_path)
        self.build_fwd_var_curve()
        self.build_smiles()
        self.build_objective_config()
        self.calibrate()
        self.write_outputs(self.summary())
        return self.calibrated


if __name__ == "__main__":
    init_logging()
    parser = ArgumentParser()
    RunParams.add_args(parser)
    qrh = QRH_Calibration(
        RunParams(**vars(parser.parse_args()))
    )
    qrh.run()

    print(qrh.calibrated)
