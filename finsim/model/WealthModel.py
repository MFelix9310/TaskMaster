from complab.model_base.ResetableModel import ResetableModel
import complab.param_space.ParamSpacePoint as psp
from finsim.agents.Employers import Employers
from finsim.agents.Government import Government
from finsim.agents.Households import Households
from finsim.agents.People import People
from finsim.assets_and_debts.Holdings import Holdings
from finsim.geography.USSpace import USSpace
from finsim.model.LifecycleModel import LifecycleModel
from finsim.data_io.FinSimCollector import FinSimCollector
from finsim.ui.GeoVizCollector import GeoVizCollector
from finsim.model.Module_Builder import Module_Builder
from finsim.personal_finance.WageIncomeCalculator import WageIncomeCalculator
from finsim.personal_finance.ExpenseDecider import ExpenseDecider
from finsim.personal_finance.TaxCalculator import TaxCalculator
from finsim.personal_finance.SocialSecurityCalculator import SocialSecurityCalculator
from finsim.assets_and_debts.AssetClassCalculator import AssetClassCalculator
from finsim.personal_finance.Savings_Module import Savings_ExternalFunction_Module
from finsim.SharedEnumsAndConstants import EmploymentStatus, CONST_SS_CONTRIB_MAX
import pandas as pd
import numpy as np

class WealthModel(LifecycleModel):
    def __init__(self, **kwargs):
        try:
            super().__init__(**kwargs)
            self.holdings = None
            self.government = None
            self.employers = None
            self.wage_income_calculator = None
            self.asset_class_calculator = None
            self.tax_calculator = None
            self.social_security_calculator = None
            self.expense_decider = None
            self.savings_module = None
            self.dashboard_mode = kwargs.get('dashboard_mode', False)

            if hasattr(self, 'all_params'):
                self._load_people_and_institutions()
                self._load_calculators()
                self._load_deciders()
                self._infer_missing_values()
        except Exception as e:
            self.log.error(f"Error initializing WealthModel: {str(e)}")
            raise

    def _load_people_and_institutions(self):
        try:
            self.people = People(self, self.all_params["person_group"])
            self.holdings = Holdings(self, self.all_params["holdings_group"])
            self.households = Households(self, self.people, self.holdings, self.all_params["household_group"])
            self.government = Government(self, self.all_params["government_group"])
            self.employers = Employers(self, self.all_params["employers_group"])
        except Exception as e:
            self.log.error(f"Error loading institutions: {str(e)}")
            raise

    def _load_calculators(self):
        try:
            super()._load_calculators()
            self.wage_income_calculator = WageIncomeCalculator(self, self.all_params["wage_income_group"], 
                                                             self.people, self.households)
            self.asset_class_calculator = AssetClassCalculator(self, self.all_params["asset_class_group"],
                                                             self.holdings)
            self.tax_calculator = TaxCalculator(self, self.all_params["tax_group"], 
                                              self.people, self.households)
            self.social_security_calculator = SocialSecurityCalculator(self, self.all_params["social_security_group"],
                                                                     self.people, self.households)
        except Exception as e:
            self.log.error(f"Error loading calculators: {str(e)}")
            raise

    def _load_deciders(self):
        try:
            super()._load_deciders()
            self.expense_decider = ExpenseDecider(self, self.all_params["expense_group"], 
                                                self.people, self.households, self.holdings)
            self.savings_module = Savings_ExternalFunction_Module(self.random, self.people, 
                                                                self.households, self.all_params["savings_group"])
        except Exception as e:
            self.log.error(f"Error loading deciders: {str(e)}")
            raise

    def _infer_missing_values(self):
        try:
            super()._infer_missing_values()
            
            if not hasattr(self, 'wage_income_calculator') or self.wage_income_calculator is None:
                return
                
            wage_module = getattr(self.wage_income_calculator, 'wage_income', None)
            if wage_module is None:
                return

            employed = self.people.data[self.people.data.employment_status == EmploymentStatus.EMPLOYED.value]
            missing_income = employed[employed.cur_income.isnull()]
            
            if not missing_income.empty and hasattr(wage_module, 'handle_newly_employed'):
                try:
                    wage_module.handle_newly_employed(missing_income)
                except Exception as e:
                    self.log.error(f"Error handling newly employed: {str(e)}")

            unemployed = self.people.data[self.people.data.employment_status == EmploymentStatus.UNEMPLOYED.value]
            missing_contributions = unemployed[unemployed.total_social_security_contributions.isnull() | 
                                            (unemployed.total_social_security_contributions == 0)]
            
            if not missing_contributions.empty and hasattr(wage_module, '_calc_income'):
                try:
                    inferred_income = pd.DataFrame(wage_module._calc_income(missing_contributions.copy()),
                                                 columns=["cur_income"])
                    inferred_income['clipped_income'] = np.clip(inferred_income.cur_income.astype(float).fillna(0), 
                                                              0, CONST_SS_CONTRIB_MAX).astype(int)
                    inferred_income = inferred_income.merge(
                        self.people.data.loc[inferred_income.index, 'years_of_social_security_contributions'],
                        left_index=True, right_index=True)
                    inferred_income['total_social_security_contributions'] = (
                        inferred_income.clipped_income * 
                        inferred_income.years_of_social_security_contributions.fillna(0)
                    ).astype(int)
                    self.people.data.loc[
                        inferred_income.index, 'total_social_security_contributions'] = inferred_income.total_social_security_contributions
                except Exception as e:
                    self.log.error(f"Error inferring social security contributions: {str(e)}")
        except Exception as e:
            self.log.error(f"Error in _infer_missing_values: {str(e)}")

    def reset_for_new_run(self):
        try:
            super().reset_for_new_run()
            
            if all(hasattr(self, attr) for attr in ['people', 'households', 'holdings']):
                self.module_builder = Module_Builder(self, self.people, self.households, self.holdings)
                
                if self.government is None and hasattr(self, 'all_params'):
                    self.government = Government(self, self.all_params.get("government_group", {}))
                
                if self.government and hasattr(self.government, 'do_implement_policies'):
                    try:
                        self.government.do_implement_policies(self)
                    except Exception as e:
                        self.log.error(f"Error implementing government policies: {str(e)}")
        except Exception as e:
            self.log.error(f"Error in reset_for_new_run: {str(e)}")

    def setup_collector(self):
        if not hasattr(self, 'all_params') or not self.all_params:
            self.log.error("Cannot setup collector: all_params not initialized")
            return

        try:
            collector_params = self.all_params.get("collector_group", {})
            if self.dashboard_mode:
                self.datacollector = GeoVizCollector(collector_params, self.time_converter)
            else:
                self.datacollector = FinSimCollector(collector_params, self.time_converter)
        except Exception as e:
            self.log.error(f"Error setting up collector: {str(e)}")
            raise

    def step(self):
        try:
            self.log.info(f"Starting Model Step {self.curtick}")
            
            # Core model step
            if self.schedule:
                self.schedule.step()
            
            # Life cycle updates
            if hasattr(self, 'life_cycle_decider'):
                self.life_cycle_decider.step_births()
                self.life_cycle_decider.step_deaths()
            
            # Economic updates
            if self.wage_income_calculator:
                self.wage_income_calculator.step_job_change()
                self.wage_income_calculator.step_wage_income()
            
            if self.social_security_calculator:
                self.social_security_calculator.step()
            
            if self.asset_class_calculator:
                self.asset_class_calculator.step()
            
            # Household updates
            if hasattr(self, 'households'):
                self.households.summarize_income(self.people)
                self.households.summarize_assets(self.holdings)
                self.households.create_head_of_household_flag()
            
            if self.tax_calculator:
                self.tax_calculator.step()
            
            if self.life_cycle_decider:
                self.life_cycle_decider.step()
            
            if self.savings_module:
                self.savings_module.step_savings_rate()
            
            if self.life_cycle_decider:
                self.life_cycle_decider.step_age()
            
            if self.government:
                self.government.do_step()
            
            # Data collection and state updates
            if hasattr(self, 'datacollector'):
                self.datacollector.collect(self)
            
            self.curtick += 1
            if self.curtick > self.max_steps:
                self.running = False
                
        except Exception as e:
            self.log.error(f"Error in step: {str(e)}")
            self.running = False
            raise
