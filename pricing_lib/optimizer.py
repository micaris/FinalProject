import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.stats import norm 
from numpy.polynomial.legendre import leggauss
from scipy.stats import multivariate_normal


class PortfolioOptimizer:

    def __init__(self, lob_data : pd.DataFrame, price_data: pd.DataFrame, init_claim, claim_strike, 
                 claim_sign, wealth, w_hat, risk_param, mu, std, no_scenarios = 100):
        """"
        Portfolio Optimizer
        """
        self.w = wealth
        self.w_hat = w_hat 
        self._lambda = risk_param
        self.lob_data = lob_data
        self.no_assets = len(self.lob_data)
        self.px_data  = price_data
        self.solver = 'MOSEK'
        self.no_scenarios  = no_scenarios
        self.mu = mu  # mean of log VIX prices
        self.std = std # Standard deviation of log VIX prices

        # Claim details
        self.claim_side = claim_sign
        self.init_claim = init_claim
        self.claim_strike = claim_strike

        # Optimizer Variables
        self.terminal_prices = None
        self.terminal_weights = None
        self.P = None # Payoff matrix
        self.c = None # Claim value
        self.x = None # Risky assets
        self.x_minus = None # Buy Qty
        self.x_plus  = None # Sell Qty
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
        Defines portfolio data based on dataset
        """
        self.terminal_prices = self.get_terminal_price_grid()
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
        self.Sx = cp.sum(cp.multiply(ask_px, self.x_plus) - cp.multiply(bid_px,self.x_minus))
        # self.cashflow = self.c - self.P @ self.x 

    def define_optimizer(self):
        # Defining objectives and constraints
        W = self.c -  (self.P @ self.x) - (self.w - self.Sx) 
        log_weights = np.log(self.terminal_weights)
        inner_expr = (self._lambda/self.w_hat)*W + log_weights
        self.OBJECTIVE = cp.Minimize((1/self._lambda)*cp.log_sum_exp(inner_expr))
        # Defining constraints
        self.CONSTRAINT = []
        self.CONSTRAINT.append( self.x == self.x_plus - self.x_minus)
        self.CONSTRAINT.append( self.x_minus <= self.lob_data.bsize.values )
        self.CONSTRAINT.append( self.x_plus  <=  self.lob_data.asize.values)
        # self.CONSTRAINT.append( self.Sx      <=  self.w)
        # Defining the optimization problem
        self.PROBLEM = cp.Problem(self.OBJECTIVE, self.CONSTRAINT)

    def get_payoff_matrix(self):
        """
        Calculates the payoff matrix for all assets and scenarios.
        """
        assets_df = self.lob_data[['option_type', 'strike', 'bid']]

        # Initialize P matrix, cols are assets and rows are scenarios
        P = np.zeros((self.no_scenarios, self.no_assets))

        # Populating P matrix
        for j in range(self.no_assets):
            asset = assets_df.iloc[j]
            vix_at_maturity = np.exp(self.terminal_prices)
            if asset['option_type'] == 'C':
                P[:, j] = np.maximum(vix_at_maturity - asset['strike'], 0)
            elif asset['option_type'] == 'P':
                P[:, j] = np.maximum(asset['strike'] - vix_at_maturity, 0)
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
            vix_prices = np.exp(self.terminal_prices)
            indicator = (vix_prices > self.claim_strike).astype(int)
            claim_payoff = 1000 * indicator
            
            # The claim side can be +1 (buy) or -1 (sell)
            claim_values += self.claim_side * claim_payoff
        
        return claim_values

    def get_terminal_price_grid(self):
        """
        Construction of log VIX price grid for all scenarios
        """
        xi_0 = self.mu - 9 * self.std
        delta_xi = (18 * self.std) / (self.no_scenarios - 1)
        xi_T = [xi_0 + i * delta_xi for i in range(self.no_scenarios)] 
        return xi_T

    def get_terminal_weights(self):
        """
        Evaluates the probability of all price scenario
        """
        delta_epsilon = (18 * self.std) / (self.no_scenarios - 1)
        p_eps_M = norm.pdf(self.terminal_prices, loc=self.mu, scale=self.std) 
        p_M = p_eps_M * delta_epsilon
        return p_M

    def solve(self, isverbose= False, iswarm = False):
        self.PROBLEM.solve(verbose=isverbose, solver=self.solver, warm_start=iswarm)
    
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
        return np.exp(self.terminal_prices)
    
    def get_positions(self):
        return self.x.value


class PortfolioOptimizerMA(PortfolioOptimizer):
    """
    Implementing multi asset portfolio optimizer
    1. Extending single asset optimizer to multi-asset case
    2. Adjusting objective function to account for multiple assets
    3. Adjusting constraints accordingly
    """

    def __init__(self, *args, **kwargs):
        # Extract grid and weights before passing remaining kwargs to parent
        self.grid = kwargs.pop('grid', None) 
        self.weights = kwargs.pop('weights', None)
        
        super().__init__(*args, **kwargs)

    def initialize_params(self):
        # Similar to single asset but ensuring dimensions match 2D grid
        pass

    def define_optimizer(self): 
        # Variables
        # x_plus: Buy quantities
        self.x_plus = cp.Variable(self.no_assets, nonneg=True)
        # x_minus: Sell quantities
        self.x_minus = cp.Variable(self.no_assets, nonneg=True)
        
        # Net position x = x_plus - x_minus
        self.x = self.x_plus - self.x_minus
        
        asks = self.lob_data['Ask'].values
        bids = self.lob_data['Bid'].values
        volumes = self.lob_data['Volume'].values
        
        # Determine multipliers for calculation
        multipliers = np.ones(self.no_assets)
        for i, ticker in enumerate(self.lob_data.Ticker):
            if ticker.startswith('UEAF'):
                multipliers[i] = 125000
            elif ticker.startswith('BGAF'):
                multipliers[i] = 62500

        # Transaction Cost S(x)
        self.Sx = cp.sum(cp.multiply(asks * multipliers, self.x_plus) - cp.multiply(bids * multipliers, self.x_minus))
        
        # Payoff matrix P
        self.P = self.get_payoff_matrix()
        
        # Claim Payoff c
        if self.c is None:
            self.populate_claim()
            
        # Term inside exp: (lambda/w_hat) * (c - Px)
        term = self.c - (self.P @ self.x)
        
        # Log-Sum-Exp trick: log( sum( exp(y_i) ) )
        log_weights = np.log(self.weights)
        inner_expr = log_weights + (self._lambda / self.w_hat) * term
        
        self.OBJECTIVE = cp.Minimize( (1/self._lambda) * cp.log_sum_exp(inner_expr) )
        
        # Constraints
        self.CONSTRAINT = []
        self.CONSTRAINT.append( self.Sx <= self.w ) # Budget constraint
        self.CONSTRAINT.append( self.x == self.x_plus - self.x_minus)
        self.CONSTRAINT.append( self.x_minus <= volumes ) # Ask volume limit
        self.CONSTRAINT.append( self.x_plus  <= volumes) # Bid volume limit
        
        self.PROBLEM = cp.Problem(self.OBJECTIVE, self.CONSTRAINT)


    def get_payoff_matrix(self):
        """
        Construct payoff matrix for 2D grid.
        Rows: Scenarios (Grid points)
        Cols: Assets (Options)
        """
        # self.grid is (N_points, 2) where col 0 is EURUSD, col 1 is GBPUSD
        # The grid contains log-prices, so we take exp to get prices
        S_X = np.exp(self.grid[:, 0]) 
        S_Y = np.exp(self.grid[:, 1]) 
        
        num_scenarios = len(S_X)
        num_assets = len(self.lob_data)
        P = np.zeros((num_scenarios, num_assets))
        
        for j in range(num_assets):
            row = self.lob_data.iloc[j]
            strike = row['Strike']
            opt_type = row['OptionType'] # 'Call' or 'Put'
            ticker = row['Ticker']
            
            # Identify underlying based on Ticker
            # UEAF... -> EURUSD
            # BGAF... -> GBPUSD
            multiplier = 1.0
            if ticker.startswith('UEAF'):
                S = S_X
                multiplier = 125000
            elif ticker.startswith('BGAF'):
                S = S_Y
                multiplier = 62500
            else:
                # Default or error handling
                continue
                
            if opt_type == 'Call':
                P[:, j] = np.maximum(S - strike, 0) * multiplier
            elif opt_type == 'Put':
                P[:, j] = np.maximum(strike - S, 0) * multiplier
                
        return P

    def populate_claim(self):
        """
        Calculate claim payoff on the grid.
        """
        # self.grid is (N_points, 2)
        S_X = np.exp(self.grid[:, 0])
        S_Y = np.exp(self.grid[:, 1])
        
        self.c = np.zeros(len(S_X))
        
        if self.init_claim is not None:
            self.c += self.init_claim

        # Add Barrier Option Payoff 
        if self.claim_strike is not None:
            # Barrier on EURUSD (S_X)
            indicator = (S_X > self.claim_strike).astype(int)
            claim_payoff = 1000 * indicator
            
            # Use claim_side if available, else default to 1
            side = getattr(self, 'claim_side', 1)
            self.c += side * claim_payoff
        
    def solve(self, isverbose=False, iswarm=False):
        self.define_optimizer()
        try:
            self.PROBLEM.solve(solver=cp.MOSEK, verbose=isverbose, warm_start=iswarm)
        except:
            self.PROBLEM.solve(verbose=isverbose)
            
        return self.PROBLEM.value


def process_fx_data(file_path):
    # Load data, ensuring Date is parsed correctly
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    # 1) Clean data: Forward fill missing values then drop any remaining leading NaNs
    # This handles gaps in exchange rate reporting
    df_cleaned = df.ffill().dropna()

    # 2) Calculate log returns for X = EURUSD and Y = GBPUSD
    # Using log(P_t / P_{t-1})
    df_cleaned['log_ret_x'] = np.log(df_cleaned['EURUSD Curncy'] / df_cleaned['EURUSD Curncy'].shift(1))
    df_cleaned['log_ret_y'] = np.log(df_cleaned['GBPUSD Curncy'] / df_cleaned['GBPUSD Curncy'].shift(1))

    # Drop the first row which will have NaN returns
    returns_df = df_cleaned[['log_ret_x', 'log_ret_y']].dropna()

    # Exclude the last day: use values up to but not including the final date
    returns_df = returns_df.iloc[:-1]

    return returns_df

def generate_quadrature_and_stats(returns_df, init_px, T,  num_points=400*400):
    # 3) Calculate mean vector and covariance matrix
    X_0, Y_0 = init_px
    mean_ret = returns_df.mean().values
    cov_mtx_ret = returns_df.cov().values
    
    # Scaled Mean and Covariance by T
    mu = np.log([X_0, Y_0]) + T*mean_ret
    cov_mtx = T * cov_mtx_ret
    sigma = np.sqrt(np.diag(cov_mtx))
    
    # 4) Create 2D Gauss-Legendre quadrature grid
    # leggauss returns points in [-1, 1] and weights
    # We transform these to [mu - 9*sigma, mu + 9*sigma]
    n = int(np.sqrt(num_points)) 
    points, weights = leggauss(n)
    
    # Transformation parameters for X and Y dimensions
    # Range [a, b] = [mu - 9*sigma, mu + 9*sigma]
    a = mu - 9 * sigma
    b = mu + 9 * sigma
    
    # Standard transformation: x_transformed = 0.5*(b-a)*x_standard + 0.5*(a+b)
    grid_x = 0.5 * (b[0] - a[0]) * points + 0.5 * (a[0] + b[0])
    grid_y = 0.5 * (b[1] - a[1]) * points + 0.5 * (a[1] + b[1])
    
    # Transform weights: w_transformed = w_standard * 0.5 * (b-a)
    weights_x = weights * 0.5 * (b[0] - a[0])
    weights_y = weights * 0.5 * (b[1] - a[1])
    
    # Create 2D Meshgrid
    X, Y = np.meshgrid(grid_x, grid_y)
    W_X, W_Y = np.meshgrid(weights_x, weights_y)
    
    quad_points = np.vstack([X.ravel(), Y.ravel()]).T
    quad_weights = (W_X * W_Y).ravel()
    
    return mu, cov_mtx, sigma, quad_points, quad_weights

def calculate_joint_probability(return_vector, mu, cov_matrix):
    """
    5) Calculate the probability density of observing a vector of log returns.
    """
    # Initialize the multivariate normal distribution
    rv = multivariate_normal(mean=mu, cov=cov_matrix)
    
    # Calculate the PDF at the given vector
    prob_density = rv.pdf(return_vector)
    
    return prob_density

def process_fx_options_data(file_path):
    options_data = pd.read_csv(file_path)
    
    # Filter out rows with zero volume
    if 'Volume' in options_data.columns:
        options_data = options_data[options_data['Volume'] > 0].copy()
    
    # Adjust strike prices and premiums for specific tickers
    # BGAF6 tickers are quoted in cents, need to convert to dollars
    mask = options_data.Ticker.str.startswith('BGAF6')
    for col in ["Strike", "Bid", "Ask", "Last"]:
        if col in options_data.columns:
            options_data.loc[mask, col] = options_data.loc[mask, col] / 100
            
    return options_data


