# Run command: $ solara run SingleFile_AntDesign.py
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

# Core imports for UI components
import solara
import ipyantd.components as antd
import ipyantd.icons.components as icons
import reacton.ipywidgets as w
import reacton.core
import ipyreact
import plotly.graph_objects as go

# Data processing and visualization imports
import pandas as pd
import random
from solara import Path
import asyncio
import numpy as np
from colour import Color

# Mesa simulation framework imports
import mesa_geo as mg
from mesa.visualization.solara_viz import _check_model_params, split_model_params, UserInputs
from mesa.visualization.components import make_plot_component
from mesa.visualization.utils import force_update, update_counter
from mesa.visualization.components.matplotlib_components import PlotMatplotlib
from mesa_geo.visualization import make_geospace_leaflet

# Custom module imports for financial simulation
import complab.param_space.ParamSpacePoint as psp
from complab.controller.Controller import Controller
from finsim.ui.GeoVizRegion import GeoVizRegion
from finsim.model.WealthModel import WealthModel
from finsim.ui.static_pages import welcome_page, technical_page
from finsim.ui.comparison_static import comparison_page
from finsim.ui.components.graphics import WrappedComponentsView, SidebarSF, VisualUtils, GraphicsUtils
from finsim.ui.UIController import UIController
from finsim.ui.UIModel import UIModel, UIModel_FinSim
from finsim.ui.figure_config import optionsR
from finsim.ui.components.util import SF_Reactive

if TYPE_CHECKING:
    from mesa.model import Model

import pandas as pd
import warnings

# Suppress FutureWarning messages for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)

# File path configuration
param_file_path = "C:/Dev/sF/FinSim/input_data/params/live_params.yaml"

# Model initialization and setup
app_language = "English"
model_class = WealthModel
ui_model = UIModel_FinSim(param_file_path, model_class)
theModel = ui_model.sim_model
params = ui_model.current_params
ui_controller = UIController(ui_model)

# Global theme configuration
solara.lab.theme.themes.light.primary = "#181414"
solara.lab.theme.themes.light.secondary = "#000000"

# Color definitions for visualization
WEALTH_COLORS = [
    "rgba(109, 255, 240, 1)",
    "rgba(109, 255, 240, 0.8)",
    "rgba(16, 131, 119, 0.8)",
    "rgba(109, 255, 240, 1)",
    "rgba(6, 54, 49, 0.4)"
]

RACIAL_COLORS = [
    "rgba(255, 185, 9, 0.8)",
    "rgba(233, 103, 43, 1)"
]

# Initialize global data series
wealth_gap_series = []
wealth_gap_colors = []
net_wealth_series = []
net_wealth_colors = []
debt_series = []

# Function to reset all data series
def reset_series():
    global wealth_gap_series, wealth_gap_colors, net_wealth_series, net_wealth_colors, debt_series
    wealth_gap_series = []
    wealth_gap_colors = []
    net_wealth_series = []
    net_wealth_colors = []
    debt_series = []

# Reactive state initialization for simulation settings
rs_time = SF_Reactive(ui_model.current_params, "model_group", "max_steps", ui_controller)
rs_state = solara.reactive("All of U.S.")

# Market scenario reactive states
rs_interest_rate = SF_Reactive(ui_model.current_params, "market_scenario_group", "interest_rates", ui_controller)
rs_unemployment = SF_Reactive(ui_model.current_params, "market_scenario_group", "unemployment", ui_controller)
rs_market_performance = SF_Reactive(ui_model.current_params, "market_scenario_group", "market_performace", ui_controller)

# Outcomes of Interest reactive states
rs_oi_wealth_by_group = SF_Reactive(ui_model.current_params, "outcomes_of_interest_group", "wealth_by_group",
                                    ui_controller)
rs_oi_percent_rr = SF_Reactive(ui_model.current_params, "outcomes_of_interest_group", "percent_retirement_ready",
                               ui_controller)
rs_oi_debt_by_asset = SF_Reactive(ui_model.current_params, "outcomes_of_interest_group", "debt_level_by_asset_type",
                                  ui_controller)
rs_oi_fw_income = SF_Reactive(ui_model.current_params, "outcomes_of_interest_group", "financial_wellness_by_income",
                              ui_controller)
rs_oi_ctv = SF_Reactive(ui_model.current_params, "outcomes_of_interest_group", "customer_lifetime_value", ui_controller)

# Map configuration and visualization states
rs_custom_var = solara.reactive("retirement_readiness")
rs_custom_groupby = solara.reactive("gender")
rs_custom_title = solara.reactive("% retirement ready by gender")
rs_map_view_lat = 39.8283
rs_map_view_long = -98.5795
rs_zoom_level = 4
rs_map_outcome_var = solara.reactive("Debt Levels")

# Available states for selection in the UI
list_of_states = [
    dict(label="All of U.S.", value="usa"),
    dict(label="California", value="california"),
    dict(label="Illinois", value="illinois"),
    dict(label="Florida", value="florida"),
    dict(label="Maryland", value="maryland"),
    dict(label="Missouri", value="missouri"),
]

# Color gradients for different visualization types
color_gradient_purple = VisualUtils.interpolate_colors(color1=Color("#5B1158"), color2=Color("#A38BA0"), steps=5)
color_gradient_orange = VisualUtils.interpolate_colors(color1=Color("#D96F3D"), color2=Color("#C9A594"), steps=5)
color_gradient_green = VisualUtils.interpolate_colors(color1=Color("#397979"), color2=Color("#8ABAB3"), steps=5)


# Event handlers for UI interactions
def on_change_duration(value):
    rs_time.set(value)


def on_market_change(value):
    rs_market_performance.set(value)
    force_update()


def on_interest_change(value):
    rs_interest_rate.set(value)
    force_update()


def on_unemployment_change(value):
    rs_unemployment.set(value)
    force_update()


# Main agent portrayal function for geographic visualization
def agent_portrayal(agent):
    if isinstance(agent, GeoVizRegion):
        if agent.adm_level == 1:
            # Handle different map visualization cases based on selected outcome variable
            match rs_map_outcome_var.value:
                case "Financial Wellness":
                    current_values, _ = theModel.datacollector.get_data_needed_for_dashboard(
                        "state_level_financial_wellness")
                    theMin = current_values["median_financial_wellness"].dropna().min()
                    theMax = current_values["median_financial_wellness"].dropna().max()
                    tempSeries = current_values.loc[current_values.adm1 == agent.region_id, "median_financial_wellness"]

                    if (tempSeries is not None) & (len(tempSeries) > 0):
                        thisState = tempSeries.iloc[0]
                        thisStateLevel = np.round((thisState - theMin) / (theMax - theMin)) * 4
                        thisStateLevel = 0 if thisStateLevel is None or np.isnan(thisStateLevel) else int(
                            thisStateLevel)
                    else:
                        thisStateLevel = 0

                    return {
                        "stroke": False,
                        "fillColor": color_gradient_purple[thisStateLevel].hex,
                        "fillOpacity": .4,
                    }

                case "Retirement Readiness":
                    current_values, _ = theModel.datacollector.get_data_needed_for_dashboard(
                        "state_level_retirement_readiness_percent")
                    theMin = current_values["median_percent_readiness"].dropna().min()
                    theMax = current_values["median_percent_readiness"].dropna().max()
                    tempSeries = current_values.loc[current_values.adm1 == agent.region_id, "median_percent_readiness"]

                    if (tempSeries is not None) & (len(tempSeries) > 0):
                        thisState = tempSeries.iloc[0]
                        thisStateLevel = np.round((thisState - theMin) / (theMax - theMin)) * 4
                        thisStateLevel = 0 if thisStateLevel is None or np.isnan(thisStateLevel) else int(
                            thisStateLevel)
                    else:
                        thisStateLevel = 0

                    return {
                        "stroke": False,
                        "fillColor": color_gradient_orange[thisStateLevel].hex,
                        "fillOpacity": .4,
                    }

                case "Debt Levels":
                    current_values, _ = theModel.datacollector.get_data_needed_for_dashboard("state_level_median_debt")
                    current_values['log_debt'] = np.log(current_values["median_debt"] + 1).fillna(0)
                    theMin = current_values["log_debt"].dropna().min()
                    theMax = current_values["log_debt"].dropna().max()
                    tempSeries = current_values.loc[current_values.adm1 == agent.region_id, "log_debt"]

                    if (tempSeries is not None) & (len(tempSeries) > 0):
                        thisState = tempSeries.iloc[0]
                        thisStateLevel = np.round((thisState - theMin) / (theMax - theMin)) * 4
                        thisStateLevel = 0 if thisStateLevel is None or np.isnan(thisStateLevel) else int(
                            thisStateLevel)
                    else:
                        thisStateLevel = 0

                    return {
                        "stroke": False,
                        "fillColor": color_gradient_green[thisStateLevel].hex,
                        "fillOpacity": .4,
                    }

                case _:
                    return {
                        "color": "grey",
                        "fillOpacity": .4,
                    }


# Simulation settings component
@solara.component
def Simulation_Settings(rs_time, rs_state, list_of_states):
    # Create factory for consistent layout of settings
    factory = VisualUtils.SettingRowFactory(layout_type="vertical")

    with ipyreact.Widget.element(_type="div"):
        # Duration control
        factory.create(
            icon="ClockCircleOutlined",
            label="Duration in Years",
            control=lambda: w.BoundedIntText(
                value=rs_time.value,
                on_value=on_change_duration,
                min=1,
                max=30,
                step=1,
                layout={"width": "100%"}
            )
        )

        # Location selection
        factory.create(
            icon="EnvironmentOutlined",
            label="Location",
            control=lambda: antd.Select(
                value=rs_state.value,
                on_value=rs_state.set,
                props=dict(options=list_of_states, className="simulation-input")
            )
        )


# Market settings component for controlling economic scenarios
@solara.component
def Market_Settings(rs_market_performance, rs_interest_rate, rs_unemployment):
    factory = VisualUtils.SettingRowFactory(layout_type="vertical")
    with ipyreact.Widget.element(_type="div", props=dict(className="pd-ilsm market-box top-nm")):
        # Market performance controls
        factory.create(
            icon="StockOutlined",
            label="Market Performance",
            control=lambda: solara.ToggleButtonsSingle(
                value=rs_market_performance.value,
                on_value=on_market_change,
                values=["Bear", "Normal", "Bull"],
                style={"margin-block": "10px", "width": "100%"}
            )
        )

        # Interest rate slider
        factory.create(
            icon="PercentageOutlined",
            label="Interest Rates",
            control=lambda: antd.Slider(
                value=rs_interest_rate.value,
                on_value=on_interest_change,
                props=dict(min=0, max=2, marks={0: "Down", 1: "Neutral", 2: "Up"}, step=1, tooltip=dict(open=False),
                           className="interest-slider", style={"width": "100%"})
            )
        )

        # Unemployment rate slider
        factory.create(
            icon="UserSwitchOutlined",
            label="Unemployment",
            control=lambda: antd.Slider(
                value=rs_unemployment.value,
                on_value=on_unemployment_change,
                props=dict(min=0, max=2, marks={0: "Low", 1: "Normal", 2: "High"}, step=1, tooltip=dict(open=False),
                           className="unemployment-slider", style={"width": "100%"})
            )
        )

# Component for managing outcome settings and metrics
@solara.component
def Outcome_Settings(rs_custom_var, rs_custom_groupby, rs_custom_title):
    # Create factory for consistent row layout
    row_factory = VisualUtils.SettingRowFactory(layout_type="vertical")

    with ipyreact.Widget.element(_type="div"):
        # Current metrics section
        with antd.Row(props=dict(style={"padding": "10px"})):
            with antd.Col(props=dict(span=24, style={"boxSizing": "border-box"})):
                solara.Preformatted(f"Current", className="txt-menu")

        # Wealth gap metric
        row_factory.create(
            icon="FunctionOutlined",
            label="Wealth Gap by group",
            control=lambda: antd.Button(
                children=["Wealth Gap by group"],
                props=dict(type="primary", icon=icons.Icon("FunctionOutlined"),
                          className="btn-outcomes btn-accent")
            )
        )

        # Retirement readiness metric
        row_factory.create(
            icon="FunctionOutlined",
            label="% of Retirement Ready",
            control=lambda: antd.Button(
                children=["% of Retirement Ready"],
                props=dict(type="primary", color="#F4F1EC",
                          icon=icons.Icon("FunctionOutlined"),
                          className="btn-outcomes btn-accent")
            )
        )

        # Common metrics section
        with antd.Row(props=dict(style={"padding": "10px"})):
            with antd.Col(props=dict(span=24, style={"boxSizing": "border-box"})):
                solara.Preformatted(f"Common", className="txt-menu")

        # Debt level metric
        row_factory.create(
            icon="FunctionOutlined",
            label="Debt level by asset type",
            control=lambda: antd.Button(
                children=["Debt level by asset type"],
                props=dict(color="#F4F1EC", icon=icons.Icon("FunctionOutlined"),
                          className="btn-outcomes btn-regular")
            )
        )

        # Asset level metric
        row_factory.create(
            icon="FunctionOutlined",
            label="Asset Level",
            control=lambda: antd.Button(
                children=["Asset Level"],
                props=dict(color="#F4F1EC", icon=icons.Icon("FunctionOutlined"),
                          className="btn-outcomes btn-regular")
            )
        )

        # Custom metrics section
        with antd.Row():
            with antd.Col(props=dict(className="flex-row flex-sb top-nm gp-rsm",
                                    style={"boxSizing": "border-box", "width": "100%"})):
                solara.Preformatted(f"Custom")
                icons.Icon("PlusOutlined")

        # Custom metrics accordion
        with antd.Row(props=dict(className="retirement-outer-outer-box top-nm")):
            with antd.Col(props=dict(className="gp-rsm outer-retirement-box retirement-clr",
                                    style={"boxSizing": "border-box", "width": "100%"})):
                accordion = w.Accordion(
                    children=[Custom_Outcome_Details(rs_custom_var, rs_custom_groupby)],
                    titles=(rs_custom_title.value,)
                )

# Custom outcome details component for detailed analysis
@solara.component
def Custom_Outcome_Details(rs_custom_var, rs_custom_groupby):
    # Define available variables for analysis
    list_of_custom_vars_normal = [
        dict(label="Retirement Readiness", value="retirement_readiness"),
        dict(label="Finnacial Wellness", value="financial_wellness"),
        dict(label="Net Worth", value="net_worth"),
    ]

    # Define grouping options
    list_of_custom_groupbys = [
        dict(label="Gender", value="gender"),
        dict(label="Race", value="race"),
        dict(label="Income", value="income_band"),
    ]

    # Layout for custom outcome analysis
    with ipyreact.Widget.element(_type="div", props=dict(className="flex-col gp-csm wd-100 retirement-clr")):
        with antd.Row(props=dict(className="pd-lft-big pd-top-nm retirement-clr")):
            with antd.Col(props=dict(className="retirement-box wd-100 pd-lft-nm retirement-clr",
                                     style={"boxSizing": "border-box"})):
                solara.Preformatted(f"Measure")
                antd.Select(value=rs_custom_var.value,
                            on_value=rs_custom_var.set,
                            props=dict(options=list_of_custom_vars_normal,
                                       className="retirement_select wd-100 top-nm"))

        with antd.Row(props=dict(className="pd-lft-big retirement-clr")):
            with antd.Col(props=dict(className="flex-row gp-csm retirement-box wd-100 pd-lft-nm retirement-clr",
                                     style={"boxSizing": "border-box"})):
                solara.Preformatted(f"Show by")
                antd.Select(value=rs_custom_groupby.value,
                            on_value=rs_custom_groupby.set,
                            props=dict(options=list_of_custom_groupbys,
                                       className="retirement_select wd-100"))


# Create dynamic bar chart visualization
def create_dynamic_bar_chart(data_series, colors, width=None, height=None, yaxis_range=None):
    fig = go.Figure()

    # Add bar trace to figure
    fig.add_trace(
        go.Bar(
            x=[str(i + 1) for i in range(len(data_series))],
            y=data_series,
            marker=dict(color=colors),
            width=1.0
        )
    )

    # Calculate dynamic Y-axis range
    if not yaxis_range:
        max_val = max(data_series) if data_series else 1
        min_val = min(data_series) if data_series else 0
        range_margin = (max_val - min_val) * 0.1
        yaxis_range = [max(min_val - range_margin, 0), max_val + range_margin]

    # Update layout for consistent styling
    fig.update_layout(
        autosize=True,
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=yaxis_range),
        template="plotly_white",
        showlegend=False,
        dragmode=False,
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        bargap=0,
        barmode="stack"
    )
    return fig


# Create density chart visualization
def create_density_chart(data_series, width=None, height=None):
    fig = go.Figure()

    # Add area trace to figure
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(data_series)),
            y=data_series,
            fill='tozeroy',
            mode='lines',
            line=dict(color='rgba(255, 185, 9, 0.8)')
        )
    )

    # Update layout for consistent styling
    fig.update_layout(
        autosize=True,
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        template="plotly_white",
        showlegend=False,
        dragmode=False,
        xaxis_fixedrange=True,
        yaxis_fixedrange=True
    )
    return fig


# Main function for displaying summary data in bar format
def Summary_Data_Bar():
    global wealth_gap_series, wealth_gap_colors, net_wealth_series, net_wealth_colors, debt_series

    try:
        # Get current tick and summary data
        tick = update_counter.get()
        summary_data = ui_controller.get_summary_data()

        if not summary_data:
            print("[WARNING] No summary data available")
            return

        # Extract and validate current values from summary data
        current_wealth_gap_value = summary_data.get("wealth_gap")
        current_net_wealth_value = summary_data.get("net_wealth")
        current_debt_value = summary_data.get("debt")
        current_retirement_ready_percent = summary_data.get("retirement_ready_percent")

        # Validate all required values are present and valid
        if any(v is None or (isinstance(v, (int, float)) and np.isnan(v)) 
               for v in [current_wealth_gap_value, current_net_wealth_value, 
                        current_debt_value, current_retirement_ready_percent]):
            print("[WARNING] Invalid or missing values in summary data")
            return

        # Set default values if needed
        current_wealth_gap_value = float(current_wealth_gap_value)
        current_net_wealth_value = float(current_net_wealth_value)
        current_debt_value = float(current_debt_value)
        current_retirement_ready_percent = float(current_retirement_ready_percent)

        # Update series with new values if they have changed
        try:
            if not wealth_gap_series or current_wealth_gap_value != wealth_gap_series[-1]:
                wealth_gap_series.append(current_wealth_gap_value)
                last_wealth_gap_color = wealth_gap_colors[-1] if wealth_gap_colors else None
                new_wealth_gap_color = VisualUtils.get_non_repeating_color(last_wealth_gap_color, RACIAL_COLORS)
                wealth_gap_colors.append(new_wealth_gap_color)

            if not net_wealth_series or current_net_wealth_value != net_wealth_series[-1]:
                net_wealth_series.append(current_net_wealth_value)
                last_net_wealth_color = net_wealth_colors[-1] if net_wealth_colors else None
                new_net_wealth_color = VisualUtils.get_non_repeating_color(last_net_wealth_color, WEALTH_COLORS)
                net_wealth_colors.append(new_net_wealth_color)

            if not debt_series or current_debt_value != debt_series[-1]:
                debt_series.append(current_debt_value)
        except Exception as e:
            print(f"[WARNING] Error updating data series: {str(e)}")
            return

        # Create visualization figures
        debt_fig = create_density_chart(
            debt_series,
            height=80
        )

        wealth_gap_fig = create_dynamic_bar_chart(
            wealth_gap_series,
            wealth_gap_colors,
            height=80
        )

        net_wealth_fig = create_dynamic_bar_chart(
            net_wealth_series,
            net_wealth_colors,
            height=80
        )

        # Create retirement readiness bar chart
        retirement_ready_fig = go.Figure()
        retirement_ready_fig.add_trace(
            go.Bar(
                y=["Retirement Readiness"],
                x=[current_retirement_ready_percent],
                orientation='h',
                marker=dict(color='#9CA371'),
                name="Ready"
            )
        )
        retirement_ready_fig.add_trace(
            go.Bar(
                y=["Retirement Readiness"],
                x=[100 - current_retirement_ready_percent],
                orientation='h',
                marker=dict(color='#C8CF6F'),
                name="Not Ready"
            )
        )

        # Configure retirement readiness chart layout
        retirement_ready_fig.update_layout(
            autosize=True,
            height=100,
            margin=dict(l=0, r=0, t=38, b=0),
            barmode='stack',
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 100]),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            template="plotly_white",
            showlegend=False,
            dragmode=False,
            xaxis_fixedrange=True,
            yaxis_fixedrange=True
        )

        # Render summary data cards
        with antd.Row(props=dict(className="flex-row flex-sb summary-data-bar")):
            with antd.Col(props=dict(className="summary-col", style={"boxSizing": "border-box"})):
                InformationCard_SmallChart(
                    title="Median Debt",
                    cur_value='${:,.0f}'.format(current_debt_value),
                    percent_change=None,
                    fig=debt_fig
                )

            with antd.Col(props=dict(className="summary-col", style={"boxSizing": "border-box"})):
                InformationCard_SmallChart(
                    title="% Retirement Readiness",
                    cur_value='{:,.2f}%'.format(current_retirement_ready_percent),
                    percent_change=None,
                    fig=retirement_ready_fig
                )

            with antd.Col(props=dict(className="summary-col", style={"boxSizing": "border-box"})):
                InformationCard_SmallChart(
                    title="Racial Wealth Gap (W-B)",
                    cur_value='${:,.0f}'.format(current_wealth_gap_value),
                    percent_change=None,
                    fig=wealth_gap_fig
                )

            with antd.Col(props=dict(className="summary-col", style={"boxSizing": "border-box"})):
                InformationCard_SmallChart(
                    title="Median Net Worth",
                    cur_value='${:,.0f}'.format(current_net_wealth_value),
                    percent_change=None,
                    fig=net_wealth_fig
                )

    except Exception as e:
        print(f"[ERROR] Error in Summary_Data_Bar: {str(e)}")


# Grouped bar chart component for asset analysis
@solara.component
def GroupedBarChart():
    # Get wealth data by asset and race
    data = theModel.datacollector.collect_wealth_by_asset_and_race(theModel)
    df = pd.DataFrame(data)

    # Configure chart parameters
    x_col = 'asset_type'
    y_col = 'median_value'
    group_col = 'race'
    filter_by = ['White', 'Black', 'Hispanic', 'Asian']
    title = 'Median Asset Value, by Group'

    # Create and display grouped bar chart
    fig = GraphicsUtils.GroupedGraphic(df, x_col, y_col, group_col, title, filter_by=filter_by)
    solara.FigurePlotly(fig)


# Component for displaying informative cards with financial data
@solara.component
def InformationCard_SmallChart(title: str, cur_value: str, percent_change: float | None, fig):
    # Setup percentage change display if available
    if percent_change is not None:
        percentage_suffix = f"{percent_change}%"
        ico_name = "trending-up.png" if percent_change >= 0 else "trending-down.png"
        ico_path = f"/static/assets/{ico_name}"

    # Create card layout with title and value
    with solara.Card(classes=["summary-card wd-100"]):
        solara.Markdown(
            f"{title}",
            style={"fontSize": "16px",
                   "color": "rgb(119, 119, 119)",
                   "fontFamily": "var(--ff-main, 'Montserrat')",
                   "fontWeight": "700",
                   "position": "absolute",
                   "top": "10px",
                   "left": "15px",
                   "zIndex": "100",
                   },
        )

        # Display main value
        solara.Markdown(
            f"{cur_value}",
            style={"fontSize": "36px",
                   "fontWeight": "500",
                   "fontFamily": "var(--ff-secondary, 'Montserrat')",
                   "color": "rgba(0,0,0,0.8)",
                   "position": "absolute",
                   "top": "25px",
                   "left": "15px",
                   "zIndex": "100",
                   },
        )

        # Show percentage change if available
        if percent_change is not None:
            with solara.Column(style={"display": "flex",
                                      "flexDirection": "row",
                                      "justifyContent": "space-between",
                                      "alignItems": "center",
                                      "backgroundColor": "rgba(0,0,0,1)",
                                      "color": "rgba(255,255,255,1)",
                                      "width": "75px",
                                      "height": "24px",
                                      "borderRadius": "25px",
                                      "padding": "0px 7px",
                                      "position": "absolute",
                                      "bottom": "20px",
                                      "right": "20px",
                                      "zIndex": "100",
                                      }):
                solara.Image(f"{ico_path}", width="20px")
                solara.Markdown(
                    f"<p style='text-align:right'>{percentage_suffix}</p>",
                    style={"fontSize": "12px",
                           "fontWeight": "500",
                           "color": "var(--bg-light)",
                           "fontFamily": "var(--ff-secondary, Montserrat)",
                           "padding-right": "0px",
                           "padding-bottom": "5px",
                           },
                )

        # Display chart if provided
        if (fig is not None):
            with solara.Column(classes=["summary-graph wd-100"]):
                solara.FigurePlotly(fig)


# Component for larger informative cards with more detailed data
@solara.component
def InformationCard_BigChart(title: str, value: str, option: str, prefix=None, suffix: str = "",
                             set_option=None, percent_change: float | None = None):
    # Create main card layout
    with solara.Card(
            style={
                "height": "auto",
                "width": "100%",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
                "alignItems": "center",
                "textAlign": "center",
                "backgroundColor": "var(--bg-light)",
                "borderRadius": "8px",
                "boxShadow": "0px 4px 12px rgba(0, 0, 0, 0.1)",
                "padding": "16px",
            }
    ):
        # Add option selector if provided
        if set_option:
            with solara.Row(style={"justifyContent": "flex-end", "width": "100%", "marginBottom": "8px"}):
                solara.ToggleButtonsSingle(
                    value=option,
                    on_value=set_option,
                    values=["Student Loan", "Credit Card"]
                )

        # Display title
        solara.Markdown(
            f"{title}",
            style={
                "fontSize": "18px",
                "color": "rgba(119,119,119,1)",
                "fontFamily": "var(--ff-main, 'Montserrat')",
                "fontweight": "700",
                "marginBottom": "4px",
                "lineHeight": "1.2",
                "marginLeft": "15px",
            },
        )

        # Display value with optional prefix
        with solara.Row(style={"alignItems": "center", "marginBottom": "2px"}):
            if prefix:
                with solara.Column(style={"fontSize": "32px"}):
                    prefix()
            solara.Markdown(
                f"{value}",
                style={
                    "fontSize": "45px",
                    "color": "#222",
                    "lineHeight": "1.2",
                    "fontFamily": "var(--ff-accent, 'Montserrat')",
                    "fontweight": "500",
                    "backgroundColor": "var(--bg-light)",
                    "marginLeft": "15px",
                },
            )

        # Display percentage change if provided
        if percent_change is not None:
            percentage_suffix = f"{percent_change}%"
            ico_name = "trending-up.png" if percent_change >= 0 else "trending-down.png"
            ico_path = f"/static/assets/{ico_name}"

            if percent_change:
                with solara.Column(style={"display": "flex",
                                          "flexDirection": "row",
                                          "justifyContent": "space-between",
                                          "alignItems": "center",
                                          "backgroundColor": "var(--bg-dark, rgba(0,0,0,1))",
                                          "color": "rgba(255,255,255,1)",
                                          "width": "90px",
                                          "height": "34px",
                                          "borderRadius": "10px",
                                          "padding": "0px 7px 0px 2px",
                                          "position": "absolute",
                                          "top": "20px",
                                          "right": "20px",
                                          "zIndex": "100",
                                          }):
                    solara.Image(f"{ico_path}", width="30px"),
                    solara.Markdown(
                        f"<p style='text-align:right'>{percentage_suffix}</p>",
                        style={
                            "color": "var(--bg-light)",
                            "font-family": "var(--ff-accent, 'Montserrat')",
                            "font-size": "var(--fs-accent, 'Montserrat')",
                            "font-style": "normal",
                            "font-weight": "500",
                            "line-height": "16px",
                            "padding": "5px 0px 0px 0px",
                            "text-align": "right",
                        }
                    )

        # Display chart
        solara.FigureEcharts(option=optionsR[option])


# Component for displaying informative cards in a grouped layout
@solara.component
def InformativeCards(option_debt, option_retirement, set_option_debt):
    # Layout for retirement and net worth cards
    with antd.Row(props=dict(gutter=[20, 20], style={"flexWrap": "nowrap", "overflowX": "auto", "height": "100%"})):
        with antd.Col(props=dict(span=12, style={"boxSizing": "border-box"})):
            InformationCard_BigChart(
                title="Retirement Ready",
                value="15%",
                option=option_retirement,
                percent_change=16.3,  # type: ignore
            )
        with antd.Col(props=dict(span=12, style={"boxSizing": "border-box"})):
            InformationCard_BigChart(
                title="Net Worth (Median)",
                value="$192,200",
                option=option_debt,
            )

    # Layout for student loan debt card
    with antd.Row(props=dict(gutter=[20, 20], style={"flexWrap": "nowrap", "overflowX": "auto"})):
        with antd.Col(props=dict(span=24, style={"boxSizing": "border-box"})):
            InformationCard_BigChart(
                title="Average Student Loan Debt",
                value="$42,183",
                option=option_debt,
                set_option=set_option_debt,
            )


# Floating playbar component for simulation control
@solara.component
def FloatingPlayBar(
        model: solara.Reactive[Model],
        *,
        model_parameters: dict | solara.Reactive[dict] = None,
        play_interval: int = 100,
):
    # Initialize reactive states
    playing = solara.use_reactive(False)
    running = solara.use_reactive(True)

    # Async function to handle simulation steps
    async def step():
        try:
            while playing.value and running.value:
                try:
                    do_step()
                    # Add small delay to prevent overwhelming the websocket
                    await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"[ERROR] Step execution error: {str(e)}")
                    playing.value = False
                    break
        except Exception as e:
            print(f"[ERROR] Simulation loop error: {str(e)}")
            playing.value = False

    # Set up task for handling simulation steps
    solara.lab.use_task(
        step, dependencies=[playing.value, running.value], prefer_threaded=False
    )

    # Function to advance simulation by one step
    def do_step():
        try:
            model.value.step()
            running.value = model.value.running
            force_update()
        except Exception as e:
            print(f"[ERROR] Model step error: {str(e)}")
            raise

    # Function to reset simulation to initial state
    def do_reset():
        try:
            global theModel
            playing.value = False
            running.value = True

            reset_series()  # Reset all data series
            ui_controller.do_reset()
            theModel = ui_model.sim_model
            model.value = theModel
            force_update()
            print("[DEBUG] Model reset completed successfully")
        except Exception as e:
            print(f"[ERROR] Error during reset: {str(e)}")

    # Function to toggle play/pause state
    def do_play_pause():
        playing.value = not playing.value

    # Render control buttons
    with solara.Row(justify="left"):
        solara.Button(
            label="Reset",
            color="var(--bg-accent)",
            on_click=do_reset
        )
        solara.Button(
            label="PLAY" if not playing.value else "PAUSE",
            color="var(--bg-accent)",
            on_click=do_play_pause,
            disabled=not running.value,
        )


# Component for welcome page
@solara.component
def Welcome():
    # Set up page metadata
    with solara.Head():
        solara.Title("FinSim")
        solara.Meta(name="description", property="og:description", content="FinSim Descript")
        solara.Meta(property="og:title", content="FinSim Title")
        solara.Meta(property="og:image", content="https://solara.dev/static/assets/images/logo.svg")
        solara.Meta(property="og:type", content="website")

    # Render welcome content
    with solara.Column(classes=["welcome-page"]):
        solara.Markdown(
            "# Welcome\n\nWelcome to our Solara app! Use the navigation menu to explore the different pages."
        )


# Component for displaying simulation visualization
@solara.component
def SimDisplay(model_reactive, model_params, components):
    reactive_model_parameters = solara.use_reactive({})

    # Display model components and control bar
    WrappedComponentsView(components, model_reactive.value)
    FloatingPlayBar(
        model_reactive,
        model_parameters=model_params,
        play_interval=1,
    )


# Component for model visualization
@solara.component
def ModelViz():
    rmodel = solara.Reactive(theModel)

    # Create geospace map component
    map_component = mg.visualization.make_geospace_leaflet(
        agent_portrayal,
        zoom=rs_zoom_level,
        view=(rs_map_view_lat, rs_map_view_long)
    )

    # Display simulation with map component
    SimDisplay(rmodel, params, [map_component])

# Main Futures component for the application
@solara.component
def Futures():
    # Initialize state variables
    sidebar_open, set_sidebar_open = solara.use_state(True)
    selected_components = solara.use_reactive([
        "Summary_Data_Bar",
        "ModelViz",
        "GroupedBarChart",
        "InformativeCards"
    ])
    option_debt, set_option_debt = solara.use_state("Student Loan")
    option_retirement = "retirement_data"
    option_new, set_option_new = solara.use_state("Student Loan")

    # Define available components for selection
    component_options = [
        {"label": "Summary Data Bar", "value": "Summary_Data_Bar"},
        {"label": "Model Visualization", "value": "ModelViz"},
        {"label": "Grouped Bar Chart", "value": "GroupedBarChart"},
        {"label": "Informative Cards", "value": "InformativeCards"},
    ]

    # Handle component selection changes
    def handle_select_change(value):
        selected_components.set(value)

    # Main page layout
    with solara.Head():
        solara.Title("FinSim")

    with solara.AppBarTitle():
        solara.Image("/static/assets/logo.png", width="200px", classes=["logo-img"])

    # Main container layout
    with solara.Row(classes=["futures-container"]):
        # Sidebar toggle button
        solara.Button(
            "|||",
            on_click=lambda: set_sidebar_open(not sidebar_open),
            style=f"""
                position: absolute;
                top: 45px;
                left: {'270px' if sidebar_open else '10px'};
                z-index: 1000;
                padding: 5px 10px;
                font-size: 16px;
                background-color: var(--color-primary);
                color: black;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2);
                transition: left 0.3s ease;
            """
        )

        # Sidebar component
        SidebarSF(
            children=[
                # Sidebar header
                lambda: solara.Text("FinSim", style={
                    "overflow": "hidden",
                    "color": "var(--Colors-Base-Purple-9, #22075E)",
                    "text-overflow": "ellipsis",
                    "font-family": "var(--ff-main, Montserrat)",
                    "background-color": "var(--bg-light)",
                    "font-size": "var(--Components-Menu-Global-fontSize, 14px)",
                    "font-style": "normal",
                    "font-weight": "700",
                    "line-height": "40px",
                    "letter-spacing": "-0.28px",
                }),
                # Sidebar content
                lambda: ipyreact.Widget.element(
                    _type="div",
                    children=[
                        w.Accordion(
                            children=[
                                Simulation_Settings(rs_time, rs_state, list_of_states),
                                Market_Settings(rs_market_performance, rs_interest_rate, rs_unemployment),
                                Outcome_Settings(rs_custom_var, rs_custom_groupby, rs_custom_title),
                                antd.Select(
                                    value=selected_components.value,
                                    on_value=handle_select_change,
                                    props=dict(
                                        options=component_options,
                                        mode="multiple",
                                        allowClear=True,
                                        style={"width": "100%"},
                                        placeholder="Please select components to display",
                                    ),
                                ),
                            ],
                            titles=('Simulation', 'Market Scenarios', 'Outcomes of Interest', 'Component Selection'),
                            titleClassName="futures-sidebar-col",
                            style={
                                "backgroundColor": "var(--bg-light)",
                                "background": "var(--bg-light)",
                            },
                        )
                    ],
                ),
            ],
            sidebar_open=sidebar_open,
            set_sidebar_open=set_sidebar_open,
        )

        # Main content area
        with solara.Column(classes=["main-future"]):
            # Display simulation timeframe
            with ipyreact.Widget.element(_type="div", props={"class": "results-title"}):
                solara.Text("Simulation from 2024 to " + str(2024 + int(rs_time.value)) + "")

            # Render selected components
            if "Summary_Data_Bar" in selected_components.value:
                Summary_Data_Bar()

            if "ModelViz" in selected_components.value:
                with solara.Card(style={"backgroundColor": "white"}):
                    with solara.Row(classes=["flex-sb wd-100"], style={"align-items": "start"}):
                        with solara.Column(style={"flex": "3"}):
                            ModelViz()

            if "GroupedBarChart" in selected_components.value:
                with solara.Card(style={"backgroundColor": "white"}):
                    with solara.Row(classes=["flex-sb wd-100"], style={"align-items": "start"}):
                        with solara.Column(style={"flex": "3"}):
                            GroupedBarChart()

            if "InformativeCards" in selected_components.value:
                with solara.Card(style={"backgroundColor": "white"}):
                    with solara.Row(classes=["flex-sb wd-100"], style={"align-items": "start"}):
                        with solara.Column(style={"flex": "3"}):
                            InformativeCards(
                                option_debt=option_debt,
                                option_retirement=option_retirement,
                                set_option_debt=set_option_debt
                            )

# Application route configuration
routes = [
    solara.Route(path="/", component=welcome_page),
    solara.Route(path="About", component=technical_page),
    solara.Route(path="Futures", component=Futures),
    solara.Route(path="Compare", component=comparison_page),
]
