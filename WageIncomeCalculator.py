"""
Module for calculating wage income and managing job transitions.
"""
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
    """
    Class for calculating wage income and managing job transitions.
    """
    def __init__(self, model, run_params, people: ppl.People, households: hld.Households):
        """Initialize the wage income calculator with model and parameters"""
        self.model = model
        self.params = run_params
        self.people = people
        self.households = households

        self._setup_wage_income_module()
        self._setup_job_transition_module()
        self._setup_disability_module()

    def _setup_wage_income_module(self):
        """Initialize wage income module based on configuration type"""
        wage_module_type = self.params["wage_income_module_type"]
        wage_params = self.params["wage_group"]

        match wage_module_type:
            case "Constant":
                self.wage_income = wim.Wage_Constant_Module(self.model.random, self.people, self.households, wage_params)
            case "Rate_Table":
                self.wage_income = wim.Wage_RateTable_Module(self.model.random, self.people, self.households, wage_params)
            case "External_Function":
                self.wage_income = wim.Wage_ExternalFunction_Module(self.model.random, self.people, self.households, wage_params)
            case "Internal_Function":
                self.wage_income = wim.Wage_InternalFunction_Module(self.model.random, self.people, self.households, wage_params)

    def _setup_job_transition_module(self):
        """Initialize job transition module based on configuration type"""
        transition_module_type = self.params["job_transition_module_type"]
        transition_params = self.params["job_transition_group"]

        match transition_module_type:
            case "Constant":
                self.job_transition = jtm.JobTransition_Constant_Module(self.model.random, self.people, self.households, transition_params)
            case "Rate_Table":
                self.job_transition = jtm.JobTransition_RateTable_Module(self.model.random, self.people, self.households, transition_params)
            case "External_Function":
                self.job_transition = jtm.JobTransition_ExternalFunction_Module(self.model.random, self.people, self.households, transition_params)
            case "Internal_Function":
                self.job_transition = jtm.JobTransition_InternalFunction_Module(self.model.random, self.people, self.households, transition_params)

    def _setup_disability_module(self):
        """Initialize disability module based on configuration type"""
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
        """Process job transitions and disability status changes"""
        self.job_transition.step_job_change()

        newly_employed = self.disability.step_disability()
        if (newly_employed is not None):
            self.wage_income.handle_newly_employed(newly_employed)

    def update_parameters(self, params):
        """Update calculator parameters and reinitialize affected modules"""
        self.params.update(params)
        if "wage_group" in params:
            self._setup_wage_income_module()
        if "job_transition_group" in params:
            self._setup_job_transition_module()
        if "disability_group" in params:
            self._setup_disability_module()

    def step_wage_income(self) -> None:
        """Process wage income calculations for the current time step"""
        self.wage_income.step_wage_income()
