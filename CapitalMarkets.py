import complab.math.TimeSeriesGenerator as tsg 
import pandas as pd


class CapitalMarkets():
    def __init__(self, rng, params):
        self.rng = rng
        self.needs_setup = False
        self.params = params

    def step(self, holdings: pd.DataFrame, mask):
        if self.is_random:
            self.international_stock.step()
            self.us_stock.step()
            self.international_bond.step()
            self.us_bond.step()
            self.cash.step()
            self.inflation.step()
        else:
            pass

    def get_real_return_for_market_holdings(self, mix: dict):
        nominal_return =  mix["international_stock"]* self.international_stock.get_cur_growth_rate() + \
            mix["us_stock"] * self.us_stock.get_cur_growth_rate() + \
               mix["international_bond"] * self.international_bond.get_cur_growth_rate() + \
               mix["us_bond"] * self.us_bond.get_cur_growth_rate() + \
               mix["cash"] * self.cash.get_cur_growth_rate() 
        real_return = nominal_return - self.inflation.get_cur_growth_rate()

    def get_real_values_for_market_holdings(self, mix: pd.DataFrame):
        nominal_value = self.international_stock.get_next_value(mix["international_stock"]) + \
            self.us_stock.get_next_value(mix["us_stock"]) + \
             self.international_bond.get_next_value(mix["international_bond"]) + \
             self.us_bond.get_next_value(mix["us_bond"]) + \
             self.cash.get_next_value(mix["cash"]) 
        real_value = nominal_value * (1 - self.inflation.get_cur_growth_rate())

    def update_conditions(self, params):
        self.params.update(params)
        self.setup(None, None)

    def setup(self, holdings: pd.DataFrame, mask):
        
        self.is_random = self.params["stochastic_model"]

        if self.is_random:
            self.international_stock = tsg.GaussianWalkGrowthSeries(self.rng, self.params["international_stock_return"], self.params["international_stock_sd"])
            self.us_stock = tsg.GaussianWalkGrowthSeries(self.rng, self.params["us_stock_return"], self.params["us_stock_sd"])
            self.international_bond = tsg.GaussianWalkGrowthSeries(self.rng, self.params["international_bond_return"], self.params["international_bond_sd"])
            self.us_bond = tsg.GaussianWalkGrowthSeries(self.rng, self.params["us_bond_return"], self.params["us_bond_sd"])
            self.cash = tsg.GaussianWalkGrowthSeries(self.rng, self.params["cash_return"], self.params["cash_sd"])

        else:
            self.international_stock = tsg.ConstantGrowthSeries(self.rng, self.params["international_stock_return"])
            self.us_stock = tsg.ConstantGrowthSeries(self.rng, self.params["us_stock_return"])
            self.international_bond = tsg.ConstantGrowthSeries(self.rng, self.params["international_bond_return"])
            self.us_bond = tsg.ConstantGrowthSeries(self.rng, self.params["us_bond_return"])
            self.cash = tsg.ConstantGrowthSeries(self.rng, self.params["cash_return"])

        self.inflation_rate = self.params["inflation_rate"]
        self.inflation = tsg.ConstantGrowthSeries(self.rng, self.inflation_rate)
 

