
import numpy as np
import pandas as pd

from complab.data_io.TabularDataReader import create_tabular_data_reader
from finsim.SharedEnumsAndConstants import EducationLevel, EmploymentStatus
import finsim.agents.People as ppl
import finsim.agents.Households as hld
import finsim.personal_finance.Wage_Module as wim
import finsim.personal_finance.JobTransition_Module as jtm
import finsim.personal_finance.Disability_Module as dm


class WageIncomeCalculator:
    def __init__(self, model, run_params, people: ppl.People, households: hld.Households):
        self.model = model
        self.params = run_params
        self.people = people
        self.households = households

        self._setup_wage_income_module()
        self._setup_job_transition_module()
        self._setup_disability_module()

    def _setup_wage_income_module(self):
        wage_income_module_type = self.params["wage_income_module_type"]
        wage_income_params = self.params["wage_group"]

        match wage_income_module_type:
            case "Constant":
                self.wage_income = wim.Wage_Constant_Module(self.model.random, self.people, self.households, wage_income_params)
            case "Rate_Table":
                self.wage_income = wim.Wage_RateTable_Module(self.model.random, self.people, self.households, wage_income_params)
            case "External_Function":
                self.wage_income = wim.Wage_ExternalFunction_Module(self.model.random, self.people, self.households, wage_income_params)
            case "Internal_Function":
                self.wage_income = wim.Wage_InternalFunction_Module(self.model.random, self.people, self.households, wage_income_params)

    def _setup_job_transition_module(self):
        job_transition_module_type = self.params["job_transition_module_type"]
        job_transition_params = self.params["job_transition_group"]

        match job_transition_module_type:
            case "Constant":
                self.job_transition = jtm.JobTransition_Constant_Module(self.model.random, self.people, self.households, job_transition_params)
            case "Rate_Table":
                self.job_transition = jtm.JobTransition_RateTable_Module(self.model.random, self.people, self.households, job_transition_params)
            case "External_Function":
                self.job_transition = jtm.JobTransition_ExternalFunction_Module(self.model.random, self.people, self.households, job_transition_params)
            case "Internal_Function":
                self.job_transition = jtm.JobTransition_InternalFunction_Module(self.model.random, self.people, self.households, job_transition_params)

    def _setup_disability_module(self):
        disability_module_type = self.params["disability_module_type"]
        disability_params = self.params["disability_group"]

        match disability_module_type:
            case "Constant":
                self.disability = dm.Disability_Constant_Module(self.model.random, self.people, self.households, disability_params)
            case "Rate_Table":
                self.disability = dm.Disability_RateTable_Module(self.model.random, self.people, self.households, disability_params)
            case "External_Function":
                self.disability = dm.Disability_ExternalFunction_Module(self.model.random, self.people, self.households, disability_params)
            case "Internal_Function":
                self.disability = dm.Disability_InternalFunction_Module(self.model.random, self.people, self.households, disability_params)

    def step_job_change(self) -> None:

        self.job_transition.step_job_change()

        newly_employed = self.disability.step_disability()
        if (newly_employed is not None):
            self.wage_income.handle_newly_employed(newly_employed)

        '''
        ## Calc Income for New Jobs (unemployed->job and job->job)
        # Newly employed is a local placeholder -- that value should never stay beyond this function
        newly_employed = self.people.data[self.people.data['employment_status'] == "newly_employed"]
        if len(newly_employed) > 0:
            self.people.data.loc[newly_employed.index, 'cur_income'] = newly_employed.apply(self.wage_income.calculate_new_job_income, axis=1)

        # Reset job status for newly employed to "employed"
        self.people.data.loc[newly_employed.index, 'employment_status'] = EmploymentStatus.EMPLOYED.value

        # TODO -- do we need to set INCOME for unemployed? Or leave it be?
        # Set income for all unemployed to zero
        self.people.data.loc[self.people.data['employment_status'] == EmploymentStatus.UNEMPLOYED.value, 'cur_income'] = 0.0
        '''


    def update_parameters(self, params):
        self.params.update(params)
        if "wage_group" in params:
            self._setup_wage_income_module()
        if "job_transition_group" in params:
            self._setup_job_transition_module()
        if "disability_group" in params:
            self._setup_disability_module()

    def step_wage_income(self) -> None:
        self.wage_income.step_wage_income()
        
        """
        # Filter the DataFrame to include only currently employed people
        employed = self.people.data[self.people.data['employment_status'] == EmploymentStatus.EMPLOYED.value]
    
        # Determine income for each employed person based on prior income and potential raises
        if len(employed)>0:
            self.people.data.loc[employed.index, 'cur_income'] = employed.apply(self.calculate_income_after_raise, axis=1)

    def calculate_income_after_raise(self, person) -> float:
        # Calculate new income based on prior income and potential raises
        prior_income = person['cur_income']

        # TODO this should likely be a PROBABILITY plus a RATE
        raise_rate = self.wage_data.loc[(person['educ_level'], person['gender'], person['adm1'], person['age']), 'raise_rate']
        new_income = prior_income * (1 + raise_rate)
        return new_income

    def calculate_new_job_income(self, person) -> float:
        # Calculate new job income based on education, gender, state, and age
        if person['employment_status'] == 'newly employed':
            new_job_income = self.wage_data.loc[(person['educ_level'], person['gender'], person['adm1'], person['age']), 'new_job_income']
            return new_job_income
        return 0.0

        """
