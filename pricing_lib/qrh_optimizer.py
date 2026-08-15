import numpy as np
import pandas as pd
import cvxpy as cp

from pricing_lib.qrh_utils import ObjConfig, get_fwd_and_discount, mid, simulate_spx_and_vix


class PortfolioOptimizerQRH:

    def __init__(self, lob_data: pd.DataFrame, qrh_calibration, init_claim, claim_strike,
                 claim_sign, wealth, w_hat, risk_param, expiry="2024-07-17",
                 no_scenarios=5000, steps_per_day=2, seed=20240621):
        """"
        Portfolio Optimizer under the calibrated Quadratic Rough Heston model.
        Scenarios come from Monte Carlo rather than a quadrature over log-VIX,
        and the tradables are SPX and VIX quotes rather than VIX alone.
        """
        self.w = wealth
        self.w_hat = w_hat
        self._lambda = risk_param
        self.lob_data = lob_data
        self.no_assets = len(self.lob_data)
        self.qrh = qrh_calibration
        self.solver = 'MOSEK'
        self.no_scenarios = no_scenarios

        # Simulation details
        self.expiry = expiry
        self.steps_per_day = steps_per_day
        self.seed = seed
        self.spx_forward = None

        # Claim details
        self.claim_side = claim_sign
        self.init_claim = init_claim
        self.claim_strike = claim_strike

        # Optimizer Variables
        self.spx_prices = None
        self.vix_prices = None
        self.terminal_weights = None
        self.P = None # Payoff matrix
        self.c = None # Claim value
        self.x = None # Risky assets
        self.x_minus = None # Buy Qty
        self.x_plus = None # Sell Qty
        self.Sx = None # Transaction Cost

        # Optimizer Solver
        self.PROBLEM = None
        self.CONSTRAINT = None
        self.OBJECTIVE = None

    def initialize_params(self):

        self.init_portfolio_data()
        self.init_optimizer_variables()
        self.define_optimizer()

    def init_portfolio_data(self):
        """
        Defines portfolio data based on simulated scenarios
        """
        self.spx_prices, self.vix_prices = self.get_terminal_price_grid()
        self.terminal_weights = self.get_terminal_weights()
        self.c = self.populate_claim()

        # Populate payoff matrix
        self.P = self.get_payoff_matrix()

    def init_optimizer_variables(self):
        """
        """
        # Decision variable
        self.x_plus = cp.Variable(self.no_assets, nonneg=True)
        self.x_minus = cp.Variable(self.no_assets, nonneg=True)
        self.x = self.x_plus - self.x_minus

        ask_px = self.lob_data.ask.values
        bid_px = self.lob_data.bid.values
        # Trading cost function
        self.Sx = cp.sum(cp.multiply(ask_px, self.x_plus) - cp.multiply(bid_px, self.x_minus))

    def define_optimizer(self):
        # Defining objectives and constraints
        W = self.c - (self.P @ self.x) - (self.w - self.Sx)
        log_weights = np.log(self.terminal_weights)
        inner_expr = (self._lambda / self.w_hat) * W + log_weights
        self.OBJECTIVE = cp.Minimize((1 / self._lambda) * cp.log_sum_exp(inner_expr))
        # Defining constraints
        self.CONSTRAINT = []
        # budget constraint from (P); needed once spx quotes are tradable, their
        # payoffs are ~30x the vix ones so longs would otherwise blow up the exponent
        self.CONSTRAINT.append( self.Sx <= self.w )
        self.CONSTRAINT.append( self.x == self.x_plus - self.x_minus)
        self.CONSTRAINT.append( self.x_minus <= self.lob_data.bsize.values )
        self.CONSTRAINT.append( self.x_plus  <=  self.lob_data.asize.values)
        # Defining the optimization problem
        self.PROBLEM = cp.Problem(self.OBJECTIVE, self.CONSTRAINT)

    def get_payoff_matrix(self):
        """
        Calculates the payoff matrix for all assets and scenarios.
        SPX assets pay off on the simulated SPX, VIX assets on the simulated VIX.
        """
        assets_df = self.lob_data[['sym', 'option_type', 'strike']]

        # Initialize P matrix, cols are assets and rows are scenarios
        P = np.zeros((self.no_scenarios, self.no_assets))

        # Populating P matrix
        for j in range(self.no_assets):
            asset = assets_df.iloc[j]
            underlying = self.spx_prices if asset['sym'] == 'SPX' else self.vix_prices
            if asset['option_type'] == 'C':
                P[:, j] = np.maximum(underlying - asset['strike'], 0)
            elif asset['option_type'] == 'P':
                P[:, j] = np.maximum(asset['strike'] - underlying, 0)
            else:
                raise ValueError("Invalid Option Type")

        return P

    def populate_claim(self):
        """
        Creates vector of claim values corresponding to all scenarios
        """
        claim_values = np.zeros(self.no_scenarios)

        # initial claim value
        if self.init_claim:
            claim_values += self.init_claim

        # Add Barrier Option Payoff
        if self.claim_strike is not None:
            indicator = (self.vix_prices > self.claim_strike).astype(int)
            claim_payoff = 1000 * indicator

            # The claim side can be +1 (buy) or -1 (sell)
            claim_values += self.claim_side * claim_payoff

        return claim_values

    def get_terminal_price_grid(self):
        """
        Simulates SPX and VIX at the option maturity off one set of QRH paths
        """
        qrh = self.qrh
        days = (pd.Timestamp(self.expiry) - qrh.quote_date).days
        n_steps = days * self.steps_per_day

        config = ObjConfig(
            spx=qrh.spx_smile, spx_step=n_steps,
            vix=qrh.vix_smiles[:1], vix_steps=[n_steps],
            vix_labels=[self.expiry], xi0_fn=qrh.fwd_var_fn,
            n_steps=n_steps, delta=(days / 365.0) / n_steps,
            n_paths=self.no_scenarios, seed=self.seed)

        spx_samples, vix_samples, _ = simulate_spx_and_vix(
            qrh.calibrated.as_array(), config)

        # simulation returns S_T/S_0 with mean 1, so scale by the quoted forward.
        # using the market forward (not spot) keeps put-call parity holding in the
        # model measure, otherwise the optimizer sees a fake synthetic-forward trade
        self.spx_forward = self.get_spx_forward()
        return self.spx_forward * spx_samples, vix_samples[0]

    def get_spx_forward(self):
        """
        Forward for the SPX expiry, fitted by process_snapshot_data on the full
        chain (parity needs both legs at a strike, so it cannot use the otm subset)
        """
        spx = self.lob_data.loc[self.lob_data.sym == 'SPX', 'forward']
        return float(spx.iloc[0]) if len(spx) else 1.0

    def get_terminal_weights(self):
        """
        Every simulated path is equally likely, so (1/lambda) log E[exp(.)] is
        estimated by a log-sum-exp over paths with weight 1/N
        """
        return np.full(self.no_scenarios, 1.0 / self.no_scenarios)

    def solve(self, isverbose=False, iswarm=False):
        # mosek is much the most reliable on the exponential cones this objective
        # generates; the others are fallbacks for when it is not licensed
        for solver in [self.solver, cp.CLARABEL, cp.SCS]:
            try:
                self.PROBLEM.solve(verbose=isverbose, solver=solver, warm_start=iswarm)
                if self.x.value is not None:
                    return
            except Exception:
                continue
        raise RuntimeError("no solver converged")

    def score(self):
        return self.PROBLEM.value

    def get_terminal_wealth(self, without_cash = False):
        if without_cash:
            return self.P @ self.x.value
        if self.x.value is not None:
            return (self.P @ self.x.value) + (self.w - self.Sx.value)
        else:
            raise print("Error: No Terminal Wealth")
            return None

    def get_vix_prices(self):
        return self.vix_prices

    def get_spx_prices(self):
        return self.spx_prices

    def get_positions(self):
        return self.x.value


def process_snapshot_data(file_path, expiry="2024-07-17", symbols=("SPX", "VIX"),
                          strike_steps=None):
    """
    Tradable set for one expiry from spx_vix_snapshot.csv.
    """
    df = pd.read_csv(file_path)
    df["mid"] = mid(df)
    df["sym"] = df.underlying_symbol.str[1:]

    frames = []
    for sym in symbols:
        chain = df[(df.sym == sym) & (df.expiration == expiry)].copy()
        chain["forward"], _ = get_fwd_and_discount(chain)

        can_buy = (chain.ask_1545 > 0) & (chain.ask_size_1545 > 0)
        can_sell = (chain.bid_1545 > 0) & (chain.bid_size_1545 > 0)
        chain = chain[can_buy | can_sell]
        chain["rel_spread"] = (chain.ask_1545 - chain.bid_1545) / chain["mid"]
        chain = chain.sort_values("rel_spread").drop_duplicates(["option_type", "strike"])
        # optional: thinning the strike grid removes near-collinear payoff columns.
        # it does not change the objective value much but makes the positions unique
        step = (strike_steps or {}).get(sym)
        if step:
            chain = chain[np.isclose(chain.strike % step, 0)]
        frames.append(chain)

    lob_data = pd.concat(frames).rename(columns={
        "bid_size_1545": "bsize", "ask_size_1545": "asize",
        "bid_1545": "bid", "ask_1545": "ask"})
    lob_data = lob_data[["expiration", "sym", "option_type", "strike",
                         "bsize", "bid", "ask", "asize", "forward"]]
    return lob_data.sort_values(["sym", "strike"]).reset_index(drop=True)
