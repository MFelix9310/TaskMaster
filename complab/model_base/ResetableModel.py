from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any
import datetime as dt
import complab.param_space.ParamSpacePoint as psp
from complab.math.ComplabRandom import ComplabRandom
from complab.math.TimeConverter import TickDuration, TimeConverter
from mesa.model import Model
from mesa.time import RandomActivationByType  # Add explicit import at top level

class ResetableModel(Model):
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        obj = object.__new__(cls)
        obj._seed = kwargs.get("seed")
        obj.random = None
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.origArgs = args
        self.origKWargs = kwargs

        super().__init__(seed=42)
        try:
            self.schedule = RandomActivationByType(self)  # Initialize schedule in __init__
        except Exception as e:
            self.log = logging.getLogger(type(self).__name__)
            self.log.error(f"Error initializing schedule: {str(e)}")
            raise
        self.running = True
        self.curtick = 0

        if (args is None or len(args) == 0) and (kwargs is None or len(kwargs) == 0):
            return

        if "params_in_yaml_file" in kwargs:
            self.all_params = psp.ParamSpacePoint.from_yaml_file(kwargs["params_in_yaml_file"])
        elif "all_params_nested" in kwargs:
            self.all_params = kwargs["all_params_nested"]
        else:
            unflattener = psp.UnflattenDict(kwargs)
            if unflattener.is_flattened():
                self.all_params = unflattener.unflatten()
            else:
                self.all_params = kwargs

        self.model_params = self.all_params[psp.getGroupKey("model")]
        self.rng_params = self.all_params[psp.getGroupKey("rng")]
        self._master_seed = self.rng_params["seed"]
        self.random = None

        self.log = logging.getLogger(type(self).__name__)
        self.setup_time_converter()

        if "preloaded_elements" in self.all_params:
            self._assign_reusuable_elements(self.all_params["preloaded_elements"])
        else:
            self._build_reusuable_elements()

        self.reset_for_new_run()

    def _assign_reusuable_elements(self, preloaded_elements):
        for element_name in preloaded_elements:
            setattr(self, element_name, preloaded_elements[element_name])

    def _build_reusuable_elements(self):
        pass

    def run_model(self) -> None:
        while self.running:
            self.step()
        self.process_end_of_run()

    def process_end_of_run(self) -> None:
        pass

    def reset_for_new_run(self):
        if self.random is None:
            self.random = ComplabRandom(self._master_seed)
        else:
            self.random.seed(self._master_seed)
        
        try:
            from mesa.time import RandomActivationByType
            self.schedule = RandomActivationByType(self)
        except ImportError:
            self.log.warning("mesa.time not available, using basic scheduling")
            self.schedule = None
            
        self.running = True
        self.curtick = 0

    def setup_time_converter(self):
        try:
            tick_duration = self.model_params.get("tick_duration", TickDuration.YEAR)
            
            start_date = dt.date.today()
            if "tick_zero_date" in self.model_params:
                tick_zero_date = self.model_params["tick_zero_date"]
                if isinstance(tick_zero_date, dt.date):
                    start_date = tick_zero_date
                elif isinstance(tick_zero_date, dt.datetime):
                    start_date = tick_zero_date.date()
                elif isinstance(tick_zero_date, str):
                    try:
                        start_date = dt.datetime.strptime(tick_zero_date, "%Y-%m-%d").date()
                    except ValueError:
                        self.log.error(f"Invalid date format in tick_zero_date: {tick_zero_date}")
                elif isinstance(tick_zero_date, int):
                    start_date = dt.date(tick_zero_date, 1, 1)
                else:
                    self.log.error(f"Unsupported tick_zero_date type: {type(tick_zero_date)}")
            
            self.time_converter = TimeConverter(tick_duration, start_date)
        except Exception as e:
            self.log.error(f"Error setting up time converter: {str(e)}")
            raise

    def step(self) -> None:
        if hasattr(self, 'schedule') and self.schedule:
            self.schedule.step()
        if hasattr(self, 'time_converter'):
            self.time_converter.step()
