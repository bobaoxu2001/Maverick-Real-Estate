"""
Monte Carlo Simulation for CRE Investment Analysis
====================================================
Evaluates how investment outcomes may vary under different
macroeconomic scenarios through stochastic simulation.

Applications:
  - Property value trajectory under rate/macro uncertainty
  - Portfolio return distribution analysis
  - Stress testing (recession, rate spike, etc.)
  - Confidence intervals for investment underwriting
  - Value-at-Risk (VaR) estimation

Author: Allen Xu
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config import MODEL_PARAMS


class MonteCarloSimulator:
    """Monte Carlo simulation for CRE investment outcome analysis.

    Models property value paths under stochastic processes,
    incorporating both systematic (macro) and idiosyncratic
    (property-specific) risk factors.
    """

    def __init__(self, params: dict = None):
        self.params = params or MODEL_PARAMS["simulation"]
        self.results = {}

    def simulate_property_value(
        self,
        initial_value: float,
        annual_drift: float = 0.03,
        annual_volatility: float = 0.12,
        time_horizon_years: int = None,
        n_simulations: int = None,
        macro_scenarios: dict = None,
    ) -> dict:
        """Simulate future property value paths using Geometric Brownian Motion.

        The GBM model:
          dV = μ·V·dt + σ·V·dW

        Where:
          V = property value
          μ = drift (expected annual appreciation)
          σ = volatility (annual standard deviation)
          W = Wiener process (random walk)

        Args:
            initial_value: Current property value ($)
            annual_drift: Expected annual appreciation rate
            annual_volatility: Annual return volatility
            time_horizon_years: Investment horizon
            n_simulations: Number of Monte Carlo paths
            macro_scenarios: Dict of macro scenarios affecting drift/vol

        Returns:
            Dict with simulation paths, statistics, and risk metrics
        """
        n_sims = n_simulations or self.params["n_simulations"]
        years = time_horizon_years or self.params["time_horizon_years"]
        n_steps = years * 12  # Monthly steps
        dt = 1 / 12

        rng = np.random.default_rng(42)

        # Generate random paths
        # Use Geometric Brownian Motion
        Z = rng.standard_normal((n_sims, n_steps))
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = initial_value

        for t in range(n_steps):
            # Apply macro scenario adjustments if provided
            drift_adj = annual_drift
            vol_adj = annual_volatility

            if macro_scenarios:
                # Time-varying drift based on macro scenarios
                year = t // 12
                if "rate_hike" in macro_scenarios and year < len(macro_scenarios.get("rate_impact", [])):
                    drift_adj += macro_scenarios["rate_impact"][year]
                if "recession" in macro_scenarios and year < len(macro_scenarios.get("recession_impact", [])):
                    drift_adj += macro_scenarios["recession_impact"][year]
                    vol_adj *= macro_scenarios.get("vol_multiplier", 1.5)

            paths[:, t + 1] = paths[:, t] * np.exp(
                (drift_adj - 0.5 * vol_adj**2) * dt
                + vol_adj * np.sqrt(dt) * Z[:, t]
            )

        # Compute statistics at each time step
        time_axis = np.arange(n_steps + 1) / 12  # In years

        percentiles = {}
        for ci in self.params["confidence_intervals"]:
            percentiles[f"p{int(ci*100)}"] = np.percentile(paths, ci * 100, axis=0)

        # Terminal value statistics
        terminal_values = paths[:, -1]
        terminal_returns = (terminal_values - initial_value) / initial_value

        # Value at Risk (VaR)
        var_95 = initial_value - np.percentile(terminal_values, 5)
        var_99 = initial_value - np.percentile(terminal_values, 1)
        cvar_95 = initial_value - terminal_values[terminal_values <= np.percentile(terminal_values, 5)].mean()

        # Probability of loss
        prob_loss = (terminal_values < initial_value).mean()
        prob_gain_20pct = (terminal_returns > 0.20).mean()

        self.results["property_simulation"] = {
            "initial_value": initial_value,
            "paths": paths,
            "time_axis": time_axis,
            "percentiles": percentiles,
            "terminal_stats": {
                "mean": terminal_values.mean(),
                "median": np.median(terminal_values),
                "std": terminal_values.std(),
                "min": terminal_values.min(),
                "max": terminal_values.max(),
                "mean_return": terminal_returns.mean(),
                "median_return": np.median(terminal_returns),
            },
            "risk_metrics": {
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "prob_loss": prob_loss,
                "prob_gain_20pct": prob_gain_20pct,
                "sharpe_ratio": terminal_returns.mean() / terminal_returns.std() if terminal_returns.std() > 0 else 0,
            },
            "parameters": {
                "drift": annual_drift,
                "volatility": annual_volatility,
                "n_simulations": n_sims,
                "horizon_years": years,
            },
        }

        logger.info(
            f"Simulation complete: {n_sims} paths, {years}yr horizon\n"
            f"  Expected value: ${terminal_values.mean():,.0f} "
            f"(return: {terminal_returns.mean():.1%})\n"
            f"  VaR-95: ${var_95:,.0f}, P(loss): {prob_loss:.1%}"
        )

        return self.results["property_simulation"]

    def scenario_analysis(
        self,
        initial_value: float,
        base_drift: float = 0.03,
        base_vol: float = 0.12,
    ) -> dict:
        """Run simulation under multiple macroeconomic scenarios.

        Scenarios:
          1. Base Case: Current trajectory continues
          2. Bull Case: Economic expansion, falling rates
          3. Bear Case: Recession, rising rates
          4. Stress Test: Severe downturn (GFC-like)
          5. Recovery: Post-downturn rebound
        """
        scenarios = {
            "Base Case": {
                "drift": base_drift,
                "volatility": base_vol,
                "description": "Current macro trends continue",
            },
            "Bull Case": {
                "drift": base_drift + 0.02,
                "volatility": base_vol * 0.8,
                "description": "Rate cuts, credit expansion, strong demand",
            },
            "Bear Case": {
                "drift": base_drift - 0.03,
                "volatility": base_vol * 1.3,
                "description": "Mild recession, rising cap rates",
            },
            "Stress Test (GFC-like)": {
                "drift": -0.08,
                "volatility": base_vol * 2.0,
                "description": "Severe downturn, credit freeze, forced liquidations",
            },
            "Recovery": {
                "drift": base_drift + 0.04,
                "volatility": base_vol * 1.1,
                "description": "Post-downturn recovery, opportunistic entry",
            },
        }

        scenario_results = {}
        for name, config in scenarios.items():
            logger.info(f"Running scenario: {name}")
            result = self.simulate_property_value(
                initial_value=initial_value,
                annual_drift=config["drift"],
                annual_volatility=config["volatility"],
                n_simulations=self.params["n_simulations"],
            )
            scenario_results[name] = {
                "description": config["description"],
                "parameters": config,
                "expected_value": result["terminal_stats"]["mean"],
                "expected_return": result["terminal_stats"]["mean_return"],
                "var_95": result["risk_metrics"]["var_95"],
                "prob_loss": result["risk_metrics"]["prob_loss"],
                "percentiles": {
                    k: v[-1] for k, v in result["percentiles"].items()
                },
            }

        self.results["scenario_analysis"] = scenario_results

        # Summary table
        summary = pd.DataFrame(scenario_results).T
        logger.info(f"\nScenario Analysis Summary:\n{summary[['expected_return', 'prob_loss', 'var_95']]}")

        return scenario_results

    def portfolio_simulation(
        self,
        property_values: list[float],
        property_drifts: list[float],
        property_vols: list[float],
        correlation_matrix: np.ndarray = None,
    ) -> dict:
        """Simulate a portfolio of multiple properties with correlation.

        Models joint behavior of multiple property investments,
        accounting for diversification benefits (or concentration risk).
        """
        n_properties = len(property_values)
        n_sims = self.params["n_simulations"]
        years = self.params["time_horizon_years"]
        n_steps = years * 12
        dt = 1 / 12

        rng = np.random.default_rng(42)

        # Default: moderate positive correlation between properties
        if correlation_matrix is None:
            correlation_matrix = np.full((n_properties, n_properties), 0.4)
            np.fill_diagonal(correlation_matrix, 1.0)

        # Cholesky decomposition for correlated random draws
        L = np.linalg.cholesky(correlation_matrix)

        # Initialize paths
        all_paths = np.zeros((n_properties, n_sims, n_steps + 1))
        for i in range(n_properties):
            all_paths[i, :, 0] = property_values[i]

        # Simulate correlated paths
        for t in range(n_steps):
            Z_indep = rng.standard_normal((n_properties, n_sims))
            Z_corr = L @ Z_indep  # Correlated draws

            for i in range(n_properties):
                all_paths[i, :, t + 1] = all_paths[i, :, t] * np.exp(
                    (property_drifts[i] - 0.5 * property_vols[i]**2) * dt
                    + property_vols[i] * np.sqrt(dt) * Z_corr[i, :]
                )

        # Portfolio value = sum of individual properties
        portfolio_paths = all_paths.sum(axis=0)
        initial_portfolio = sum(property_values)
        terminal_portfolio = portfolio_paths[:, -1]
        portfolio_returns = (terminal_portfolio - initial_portfolio) / initial_portfolio

        self.results["portfolio_simulation"] = {
            "n_properties": n_properties,
            "initial_portfolio_value": initial_portfolio,
            "property_paths": all_paths,
            "portfolio_paths": portfolio_paths,
            "terminal_stats": {
                "mean": terminal_portfolio.mean(),
                "median": np.median(terminal_portfolio),
                "std": terminal_portfolio.std(),
                "mean_return": portfolio_returns.mean(),
            },
            "risk_metrics": {
                "var_95": initial_portfolio - np.percentile(terminal_portfolio, 5),
                "prob_loss": (terminal_portfolio < initial_portfolio).mean(),
                "diversification_benefit": 0.0,
            },
        }
        total_std = sum(all_paths[i, :, -1].std() for i in range(n_properties))
        if total_std > 0:
            self.results["portfolio_simulation"]["risk_metrics"]["diversification_benefit"] = (
                total_std - terminal_portfolio.std()
            ) / total_std

        logger.info(
            f"Portfolio simulation: {n_properties} properties, "
            f"${initial_portfolio:,.0f} total\n"
            f"  Expected return: {portfolio_returns.mean():.1%}, "
            f"Diversification benefit: "
            f"{self.results['portfolio_simulation']['risk_metrics']['diversification_benefit']:.1%}"
        )

        return self.results["portfolio_simulation"]
