"""
Module for calculating asset values and managing asset classes.
"""
from finsim.assets_and_debts.Holdings import Holdings
import finsim.assets_and_debts.AssetClass as ac
from finsim.SharedEnumsAndConstants import AssetType
from finsim.assets_and_debts.CapitalMarkets import CapitalMarkets


class AssetClassCalculator:
    """
    Class for calculating the value and accumulated interest for each holding.
    """

    def __init__(self, model, params, holdings: Holdings):
        self.model = model
        self.params = params
        self.holdings = holdings
        self._market_conditions = params.get('capital_markets_group', {}).copy()

        self.asset_classes = {}

        self.setup_capital_markets()
        self.setup_asset_classes()
        self.setup_holdings()

    def update_parameters(self, new_params):
        """Updates the parameters for the asset calculator"""
        try:
            print(f"[DEBUG] Updating parameters: {new_params}")
            if 'capital_markets_group' in new_params:
                self._market_conditions.update(new_params['capital_markets_group'])
                self.capital_markets.update_conditions(self._market_conditions)

            for asset_type, asset_class in self.asset_classes.items():
                class_key = f"ac_{asset_type.value.lower()}_group"
                if class_key in new_params:
                    if hasattr(asset_class, 'update_parameters'):
                        asset_class.update_parameters(new_params[class_key])

            return True
        except Exception as e:
            print(f"[ERROR] Error updating AssetClassCalculator parameters: {str(e)}")
            return False

    def setup_capital_markets(self):
        capital_markets_params = self.params['capital_markets_group']
        self.capital_markets = CapitalMarkets(self.model.random, capital_markets_params)

    def setup_asset_classes(self):
        """
        Creates an object for each AssetClass subclass and adds it to the asset_classes dictionary.
        """
        rng = self.model.random
        self.asset_classes[AssetType.HOUSE] = ac.AssetClass_House(self.params["ac_house_group"], rng)
        self.asset_classes[AssetType.OTHER_REAL_ESTATE] = ac.AssetClass_OtherRealEstate(
            self.params["ac_otherrealestate_group"], rng)
        self.asset_classes[AssetType.BUSINESS] = ac.AssetClass_Business(self.params["ac_business_group"], rng)
        self.asset_classes[AssetType.BROKERAGE_STOCK] = ac.AssetClass_BrokerageStocks(
            self.params["ac_brokerage_group"], rng)
        self.asset_classes[AssetType.CHECKING_SAVINGS] = ac.AssetClass_CheckingAndSavings(
            self.params["ac_checking_group"], rng)
        self.asset_classes[AssetType.VEHICLE] = ac.AssetClass_Vehicle(self.params["ac_vehicle_group"], rng)
        self.asset_classes[AssetType.OTHER_ASSETS] = ac.AssetClass_OtherAssets(
            self.params["ac_otherasset_group"], rng)
        self.asset_classes[AssetType.OTHER_DEBTS] = ac.AssetClass_AllOtherDebts(
            self.params["ac_otherdebt_group"], rng)
        self.asset_classes[AssetType.DB_RETIREMENT_PLAN] = ac.AssetClass_DB_RetirementPlan(
            self.params["ac_db_group"], rng)
        self.asset_classes[AssetType.DC_RETIREMENT_PLAN] = ac.AssetClass_DC_RetirementPlan(
            self.params["ac_dc_group"], rng)
        self.asset_classes[AssetType.GOVT_STUDENT_LOANS] = ac.AssetClass_DC_RetirementPlan(
            self.params["ac_govt_student_loans_group"], rng)
        self.asset_classes[AssetType.EDUCATION_SAVINGS] = ac.AssetClass_DC_RetirementPlan(
            self.params["ac_education_savings_group"], rng)
        self.asset_classes[AssetType.TRADITIONAL_IRA] = ac.AssetClass_DC_RetirementPlan(
            self.params["ac_trad_ira_group"], rng)
        self.asset_classes[AssetType.ROTH_IRA] = ac.AssetClass_DC_RetirementPlan(
            self.params["ac_roth_ira_group"], rng)
        self.asset_classes[AssetType.CREDIT_CARD] = ac.AssetClass_CreditCard(
            self.params["ac_creditcard_group"], rng)
        self.asset_classes[AssetType.MORTGAGE] = ac.AssetClass_Mortgage(
            self.params["ac_mortgage_group"], rng)

    def setup_holdings(self):
        for asset_type, asset_class in self.asset_classes.items():
            if (asset_class.needs_setup):
                holdings_mask = self.holdings.data['asset_type'] == asset_type
                if (sum(holdings_mask) > 0):
                    asset_class.setup(self.holdings, holdings_mask)

    def _update_market_returns(self):
        """Updates returns based on current market conditions"""
        try:
            market_performance = self._market_conditions.get("market_performance", "Normal")
            if market_performance == "Bull":
                self.params['capital_markets_group'].update({
                    "us_stock_return": 15.0,
                    "international_stock_return": 12.0,
                    "us_stock_sd": 12.0,
                    "international_stock_sd": 14.0
                })
            elif market_performance == "Bear":
                self.params['capital_markets_group'].update({
                    "us_stock_return": -10.0,
                    "international_stock_return": -8.0,
                    "us_stock_sd": 25.0,
                    "international_stock_sd": 28.0
                })
            else:  # Normal
                self.params['capital_markets_group'].update({
                    "us_stock_return": 7.0,
                    "international_stock_return": 6.0,
                    "us_stock_sd": 15.0,
                    "international_stock_sd": 17.0
                })

            self.capital_markets = CapitalMarkets(self.model.random, self.params['capital_markets_group'])

        except Exception as e:
            print(f"[ERROR] Error updating market returns: {str(e)}")

    def step(self):
        """
        Calculates the value and accumulated interest for each holding.
        """
        try:
            self._update_market_returns()

            for asset_type, asset_class in self.asset_classes.items():
                holdings_mask = self.holdings.data['asset_type'] == asset_type
                if (sum(holdings_mask) > 0):
                    asset_class.step(self.holdings, holdings_mask)

        except Exception as e:
            print(f"[ERROR] Error in AssetClassCalculator execution: {str(e)}")
