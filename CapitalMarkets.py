"""
Module for managing capital markets and their returns.
"""
import complab.math.TimeSeriesGenerator as tsg
import pandas as pd


class CapitalMarkets:
    """
    Class for managing capital markets and their returns.
    """
    def __init__(self, rng, params):
        """Initialize the capital markets with random number generator and parameters"""
        self.rng = rng
        self.needs_setup = False
        self.params = params

    def step(self, holdings: pd.DataFrame, mask):
        """Update all market values for the current time step"""
        if self.is_random:
            self.us_stocks.step()
            self.international_stocks.step()
            self.us_bonds.step()
            self.international_bonds.step()
            self.cash.step()
            self.inflation.step()

    def get_real_return_for_holdings(self, mix: dict):
        """Calculate real return for given asset mix accounting for inflation"""
        nominal_return = (
            mix["international_stock"] * self.international_stocks.get_cur_growth_rate()
            + mix["us_stock"] * self.us_stocks.get_cur_growth_rate()
            + mix["international_bond"] * self.international_bonds.get_cur_growth_rate()
            + mix["us_bond"] * self.us_bonds.get_cur_growth_rate()
            + mix["cash"] * self.cash.get_cur_growth_rate()
        )
        real_return = nominal_return - self.inflation.get_cur_growth_rate()
        return real_return

    def get_real_values_for_holdings(self, mix: pd.DataFrame):
        """Calculate real values for given asset mix accounting for inflation"""
        nominal_value = (
            self.international_stocks.get_next_value(mix["international_stock"])
            + self.us_stocks.get_next_value(mix["us_stock"])
            + self.international_bonds.get_next_value(mix["international_bond"])
            + self.us_bonds.get_next_value(mix["us_bond"])
            + self.cash.get_next_value(mix["cash"])
        )
        real_value = nominal_value * (1 - self.inflation.get_cur_growth_rate())
        return real_value

    def update_conditions(self, params):
        """Update market conditions with new parameters"""
        self.params.update(params)
        self._initialize_series()

    def _initialize_series(self):
        """Initialize market time series based on current parameters"""
        self.is_random = self.params["stochastic_model"]

        if self.is_random:
            self.us_stocks = tsg.GaussianWalkGrowthSeries(self.rng, self.params["us_stock_return"], self.params["us_stock_sd"])
            self.international_stocks = tsg.GaussianWalkGrowthSeries(self.rng, self.params["international_stock_return"], self.params["international_stock_sd"])
            self.us_bonds = tsg.GaussianWalkGrowthSeries(self.rng, self.params["us_bond_return"], self.params["us_bond_sd"])
            self.international_bonds = tsg.GaussianWalkGrowthSeries(self.rng, self.params["international_bond_return"], self.params["international_bond_sd"])
            self.cash = tsg.GaussianWalkGrowthSeries(self.rng, self.params["cash_return"], self.params["cash_sd"])
        else:
            self.us_stocks = tsg.ConstantGrowthSeries(self.rng, self.params["us_stock_return"])
            self.international_stocks = tsg.ConstantGrowthSeries(self.rng, self.params["international_stock_return"])
            self.us_bonds = tsg.ConstantGrowthSeries(self.rng, self.params["us_bond_return"])
            self.international_bonds = tsg.ConstantGrowthSeries(self.rng, self.params["international_bond_return"])
            self.cash = tsg.ConstantGrowthSeries(self.rng, self.params["cash_return"])

        self.inflation_rate = self.params["inflation_rate"]
        self.inflation = tsg.ConstantGrowthSeries(self.rng, self.inflation_rate)

    def setup(self, holdings: pd.DataFrame, mask):
        """Initialize market conditions and time series generators"""
        self._initialize_series()
