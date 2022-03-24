from pandas.core.common import SettingWithCopyError
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import copy
import pandas as pd
import argparse
import os
import time
import random
import numpy as np
from IPython import embed as IP
from matplotlib import pyplot as plt, cm as cmx, colors
import sqlite3
import sys
import matplotlib
matplotlib.use('Agg')


pd.options.mode.chained_assignment = None


class OutputPlotGenerator:

    def __init__(self, path_to_db, plot_type):
        self.db_path = os.path.abspath(path_to_db)

    def extractFromDatabase(self, type):
        '''
        Based on the type of the plot being generated, extract data from the corresponding table from database
        '''

        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        if (type == 'capacity'):

            # Load in the capacity data
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            cur.execute(
                "SELECT scenario, sector, t_periods, tech, regions, capacity FROM Output_CapacityByPeriodAndTech")
            self.capacity_output = cur.fetchall()
            self.capacity_output = [list(elem) for elem in self.capacity_output]
            self.capacity_output = pd.DataFrame(self.capacity_output, columns=[
                                                'scenario', 'sector', 't_periods', 'tech', 'region', 'value'])
            self.capacity_output = self.capacity_output.astype(
                {'scenario': 'category',
                 'sector': 'category',
                 't_periods': 'int',
                 'tech': 'category',
                 'region': 'category',
                 'value': 'float64'})

            # Load in the annual energy data
            cur.execute("SELECT scenario, sector, t_periods, input_comm, tech, regions, output_comm, SUM(vflow_out) FROM Output_VFlow_Out GROUP BY scenario, sector, regions, t_periods, input_comm, tech, output_comm")
            self.energy_output = cur.fetchall()
            self.energy_output = [list(elem) for elem in self.energy_output]
            self.energy_output = pd.DataFrame(self.energy_output, columns=[
                                              'scenario', 'sector', 't_periods', 'input_comm', 'tech', 'region', 'output_comm', 'value'])
            self.energy_output = self.energy_output.astype(
                {'scenario': 'category',
                 'sector': 'category',
                 't_periods': 'int',
                 'input_comm': 'category',
                 'tech': 'category',
                 'region': 'category',
                 'output_comm': 'category',
                 'value': 'float64'})

        elif (type == 'flow-out'):
            cur.execute("SELECT scenario, sector, regions, t_periods, t_season, t_day, input_comm, tech, output_comm, SUM(vflow_out) FROM Output_VFlow_Out GROUP BY scenario, sector, regions, t_periods, t_season, t_day, input_comm, tech, output_comm")
            self.flow_output = cur.fetchall()
            self.flow_output = [list(elem) for elem in self.flow_output]
            self.flow_output = pd.DataFrame(self.flow_output, columns=[
                                            'scenario', 'sector', 'region', 't_periods', 'date', 'hour', 'input_comm', 'tech', 'output_comm', 'vflow_out'])
            self.flow_output = self.flow_output.astype(
                {'scenario': 'category',
                 'sector': 'category',
                 'region': 'category',
                 't_periods': 'str',
                 'date': 'str',
                 'hour': 'str',
                 'input_comm': 'category',
                 'tech': 'category',
                 'output_comm': 'category',
                 'vflow_out': 'float64'})

            self.flow_output['timestamp'] = self.flow_output['t_periods'] + '-' + \
                self.flow_output['date'] + '-' + self.flow_output['hour'].str[1:] + ':00'
            self.flow_output['timestamp'] = pd.to_datetime(self.flow_output['timestamp'])
            self.flow_output['t_periods'] = pd.to_numeric(
                self.flow_output['t_periods'])  # convert from string to int
            self.flow_output.sort_values(by=['timestamp'], inplace=True)

            cur.execute("SELECT scenario, sector, regions, t_periods, t_season, t_day, input_comm, tech, output_comm, SUM(curtailment) FROM Output_Curtailment GROUP BY scenario, sector, regions, t_periods, t_season, t_day, input_comm, tech, output_comm")
            self.curtailment = cur.fetchall()
            self.curtailment = [list(elem) for elem in self.curtailment]
            self.curtailment = pd.DataFrame(self.curtailment, columns=[
                                            'scenario', 'sector', 'region', 't_periods', 'date', 'hour', 'input_comm', 'tech', 'output_comm', 'curtailment'])
            self.curtailment = self.curtailment.astype(
                {'scenario': 'category',
                 'sector': 'category',
                 'region': 'category',
                 't_periods': 'str',
                 'date': 'str',
                 'hour': 'str',
                 'input_comm': 'category',
                 'tech': 'category',
                 'output_comm': 'category',
                 'curtailment': 'float64'})

            self.curtailment['timestamp'] = self.curtailment['t_periods'] + '-' + \
                self.curtailment['date'] + '-' + self.curtailment['hour'].str[1:] + ':00'
            self.curtailment['timestamp'] = pd.to_datetime(self.curtailment['timestamp'])
            self.curtailment.sort_values(by=['timestamp'], inplace=True)
            self.curtailment['t_periods'] = pd.to_numeric(
                self.curtailment['t_periods'])  # convert from string to int

            self.flow_output['curtailment'] = 0.0
            _temp = self.curtailment[self.curtailment.curtailment > 0]
            for index, row in _temp.iterrows():
                idx = self.flow_output.loc[
                    (self.flow_output.scenario == row.scenario) &
                    (self.flow_output.sector == row.sector) &
                    (self.flow_output.region == row.region) &
                    (self.flow_output.timestamp == row.timestamp) &
                    (self.flow_output.input_comm == row.input_comm) &
                    (self.flow_output.tech == row.tech) &
                    (self.flow_output.output_comm == row.output_comm)
                ]

                if idx.empty:
                    continue

                self.flow_output.at[idx.index.values[0], 'curtailment'] = row.curtailment

            self.flow_output['value'] = self.flow_output['vflow_out'] + \
                self.flow_output['curtailment']
            self.ndays = len(self.flow_output.date.unique())

        elif (type == 'emissions'):
            cur.execute(
                "SELECT scenario, sector, t_periods, emissions_comm, regions, emissions FROM Output_Emissions")
            self.emissions_output = cur.fetchall()
            self.emissions_output = [list(elem) for elem in self.emissions_output]
            self.emissions_output = pd.DataFrame(self.emissions_output, columns=[
                                                 'scenario', 'sector', 't_periods', 'emissions_comm', 'region', 'value'])
            self.emissions_output = self.emissions_output.astype(
                {'scenario': 'category',
                 'sector': 'category',
                 't_periods': 'int',
                 'emissions_comm': 'category',
                 'region': 'category',
                 'value': 'float64'})

        elif (type == 'flow-in'):
            cur.execute("SELECT scenario, sector, regions, t_periods, t_season, t_day, input_comm, tech, output_comm, SUM(vflow_in) FROM Output_VFlow_In GROUP BY scenario, sector, regions, t_periods, t_season, t_day, input_comm, tech, output_comm")
            self.flow_input = cur.fetchall()
            self.flow_input = [list(elem) for elem in self.flow_input]
            self.flow_input = pd.DataFrame(self.flow_input, columns=[
                                           'scenario', 'sector', 'region', 't_periods', 'date', 'hour', 'input_comm', 'tech', 'output_comm', 'value'])
            self.flow_input = self.flow_input.astype(
                {'scenario': 'category',
                 'sector': 'category',
                 'region': 'category',
                 't_periods': 'str',
                 'date': 'str',
                 'hour': 'str',
                 'input_comm': 'category',
                 'tech': 'category',
                 'output_comm': 'category',
                 'value': 'float64'})
            self.flow_input['timestamp'] = self.flow_input['t_periods'] + '-' + \
                self.flow_input['date'] + '-' + self.flow_input['hour'].str[1:] + ':00'
            self.flow_input['timestamp'] = pd.to_datetime(self.flow_input['timestamp'])
            self.flow_input.sort_values(by=['timestamp'], inplace=True)
            self.flow_input['t_periods'] = pd.to_numeric(
                self.flow_input['t_periods'])  # convert from string to int
        elif (type == 'demand-total'):
            cur.execute("SELECT regions, periods, demand_comm, demand FROM Demand")
            self.annual_demand = cur.fetchall()
            self.annual_demand = [list(elem) for elem in self.annual_demand]
            self.annual_demand = pd.DataFrame(self.annual_demand, columns=[
                                              'region', 't_periods', 'demand_comm', 'value'])
            self.annual_demand = self.annual_demand.astype(
                {'region': 'category',
                 't_periods': 'int',
                 'demand_comm': 'category',
                 'value': 'float64'})

        elif (type == 'seg-frac'):
            cur.execute("SELECT season_name, time_of_day_name, segfrac FROM SegFrac")
            self.segfrac = cur.fetchall()
            self.segfrac = [list(elem) for elem in self.segfrac]
            self.segfrac = pd.DataFrame(self.segfrac, columns=['date', 'hour', 'value'])
            self.segfrac = self.segfrac.astype(
                {'date': 'str',
                 'hour': 'str',
                 'value': 'float64'})

        elif (type == 'demand-specific-distribution'):
            cur.execute(
                "SELECT regions, season_name, time_of_day_name, demand_name, dds FROM DemandSpecificDistribution")
            self.dsd = cur.fetchall()
            self.dsd = [list(elem) for elem in self.dsd]
            self.dsd = pd.DataFrame(
                self.dsd, columns=['region', 'date', 'hour', 'demand_comm', 'value'])
            self.dsd = self.dsd.astype(
                {'region': 'category',
                 'date': 'str',
                 'hour': 'str',
                 'value': 'float64'})
        elif (type == 'curtailment'):
            cur.execute("SELECT scenario, sector, regions, t_periods, t_season, t_day, input_comm, tech, output_comm, SUM(curtailment) FROM Output_Curtailment GROUP BY scenario, sector, regions, t_periods, t_season, t_day, input_comm, tech, output_comm")
            self.curtailment = cur.fetchall()
            self.curtailment = [list(elem) for elem in self.curtailment]
            self.curtailment = pd.DataFrame(self.curtailment, columns=[
                                            'scenario', 'sector', 'region', 't_periods', 'date', 'hour', 'input_comm', 'tech', 'output_comm', 'value'])
            self.curtailment = self.curtailment.astype(
                {'scenario': 'category',
                 'sector': 'category',
                 'region': 'category',
                 't_periods': 'str',
                 'date': 'str',
                 'hour': 'str',
                 'input_comm': 'category',
                 'tech': 'category',
                 'output_comm': 'category',
                 'value': 'float64'})

            self.curtailment['timestamp'] = self.curtailment['t_periods'] + '-' + \
                self.curtailment['date'] + '-' + self.curtailment['hour'].str[1:] + ':00'
            self.curtailment['timestamp'] = pd.to_datetime(self.curtailment['timestamp'])
            self.curtailment.sort_values(by=['timestamp'], inplace=True)
            self.curtailment['t_periods'] = pd.to_numeric(
                self.curtailment['t_periods'])  # convert from string to int

        cur.execute(
            "SELECT tech, tech_category, sector, subsector, tech_desc, cap_units, color, plot_order FROM technologies")
        self.tech_info = cur.fetchall()
        self.tech_info = [[str(word) for word in tuple] for tuple in self.tech_info]
        self.tech_info = pd.DataFrame(self.tech_info, columns=[
                                      'tech', 'tech_category', 'sector', 'subsector', 'description', 'units', 'color', 'plot_order'])
        self.tech_info = self.tech_info.astype({'tech': 'string', 'tech_category': 'string',
                                                'description': 'string', 'units': 'string', 'color': 'string', 'plot_order': 'string'})
        self.tech_info.plot_order = pd.to_numeric(self.tech_info.plot_order, errors='coerce')
        # We want to convert the plot_order to numeric type. We first must deal with empty strings and the like

        cur.execute("SELECT sector, description, color, plot_order FROM sector_labels")
        self.sector_info = cur.fetchall()
        self.sector_info = [[str(word) for word in tuple] for tuple in self.sector_info]
        self.sector_info = pd.DataFrame(self.sector_info, columns=[
                                        'sector',  'description', 'color', 'plot_order'])
        self.sector_info = self.sector_info.astype(
            {'sector': 'string', 'description': 'string', 'color': 'string', 'plot_order': 'string'})
        self.sector_info.plot_order = pd.to_numeric(self.sector_info.plot_order, errors='coerce')
        # We want to convert the plot_order to numeric type. We first must deal with empty strings and the like

        cur.execute("SELECT subsector, description, color, plot_order FROM subsector_labels")
        self.subsector_info = cur.fetchall()
        self.subsector_info = [[str(word) for word in tuple] for tuple in self.subsector_info]
        self.subsector_info = pd.DataFrame(self.subsector_info, columns=[
                                           'subsector',  'description', 'color', 'plot_order'])
        self.subsector_info = self.subsector_info.astype(
            {'subsector': 'string', 'description': 'string', 'color': 'string', 'plot_order': 'string'})
        self.subsector_info.plot_order = pd.to_numeric(
            self.subsector_info.plot_order, errors='coerce')
        # We want to convert the plot_order to numeric type. We first must deal with empty strings and the like

        cur.execute("SELECT regions, plot_order, region_note FROM regions")
        self.region_info = cur.fetchall()
        self.region_info = [[str(word) for word in tuple] for tuple in self.region_info]
        self.region_info = pd.DataFrame(self.region_info, columns=['region', 'plot_order', 'note'])
        self.region_info = self.region_info.astype(
            {'region': 'string', 'plot_order': 'string', 'note': 'string'})
        self.region_info.plot_order = pd.to_numeric(
            self.region_info.plot_order, errors='coerce')
        # We want to convert the plot_order to numeric type. We first must deal with empty strings and the like

        cur.execute("SELECT comm_name, comm_desc, units FROM commodities")
        self.commodity_info = cur.fetchall()
        self.commodity_info = [[str(word) for word in tuple] for tuple in self.commodity_info]
        self.commodity_info = pd.DataFrame(self.commodity_info, columns=[
                                           'name', 'description', 'units'])
        self.commodity_info = self.commodity_info.astype(
            {'name': 'string', 'description': 'string', 'units': 'string'})
        con.close()

    def processData(self, inputData, datatype='capacity', super_categories=False):
        '''
        Processes data to make it ready for plotting purposes
        '''

        if datatype == 'capacity':
            outputData = copy.copy(inputData)
            outputData['tech_order'] = [self.tech_info[self.tech_info.tech ==
                                                       _].plot_order.values[0] for _ in outputData['tech']]
            outputData['region_order'] = [self.region_info[self.region_info.region ==
                                                           _.split('-')[0]].plot_order.values[0] for _ in outputData['region']]

            tech_categories = list(self.tech_info.tech_category.unique())
            if '' in tech_categories:
                tech_categories.remove('')

            categories_to_tech = {}
            for tc in tech_categories:
                categories_to_tech[tc] = list(
                    self.tech_info[self.tech_info.tech_category == tc].tech.unique())

            def get_subsector(t):
                return self.tech_info[self.tech_info.tech == t].subsector.values[0]
            outputData['subsector'] = outputData['tech'].apply(get_subsector)

            if super_categories:
                for sc in outputData.scenario.unique():
                    for y in outputData.t_periods.unique():
                        for r in outputData.region.unique():
                            for s in outputData.sector.unique():
                                _d = outputData[(outputData.t_periods == y)
                                                & (outputData.region == r)
                                                & (outputData.sector == s)
                                                & (outputData.scenario == sc)]
                                for cat in tech_categories:
                                    __d = _d[_d.tech.isin(categories_to_tech[cat])]
                                    indices = __d.index
                                    if len(indices) == 0:
                                        continue
                                    elif len(indices) == 1:
                                        for tt in categories_to_tech[cat]:
                                            if tt in outputData.tech.unique():
                                                break
                                        outputData.loc[indices[0], 'tech'] = tt
                                        continue
                                    else:
                                        outputData.loc[indices[0], 'value'] = __d.value.values.sum()

                                        for tt in categories_to_tech[cat]:
                                            if tt in outputData.tech.unique():
                                                break

                                        outputData.loc[indices[0], 'tech'] = tt
                                        outputData = outputData.drop(index=indices[1:])

            # Make sure the data is presented in the proper plot_order
            outputData = outputData.sort_values(by=['t_periods', 'tech_order', 'region_order'])
            outputData = outputData.reset_index(drop=True)

        elif datatype == 'emissions':
            outputData = copy.copy(inputData)
            outputData['region_order'] = [self.region_info[self.region_info.region ==
                                                           _.split('-')[0]].plot_order.values[0] for _ in outputData['region']]
            outputData['sector_order'] = [self.sector_info[self.sector_info.sector ==
                                                           _].plot_order.values[0] for _ in outputData['sector']]
            # For each year/region combination, make sure each technology and emissions_comm appear. If not, set to 0.
            emission_commodity_list = outputData.emissions_comm.unique()
            for sc in self.emissions_output.scenario.unique():
                for y in outputData.t_periods.unique():
                    for r in outputData.region.unique():
                        for s in outputData.sector.unique():
                            _d = outputData[(outputData.t_periods == y) & (outputData.region == r)]
                            for e in emission_commodity_list:
                                if e not in _d.emissions_comm.unique():
                                    outputData = outputData.append({'scenario': sc,
                                                                    'sector': s,
                                                                    't_periods': y,
                                                                    'emissions_comm': e,
                                                                    'region': r,
                                                                    'value': 0.0,
                                                                    'region_order': self.region_info[self.region_info.region == r.split('-')[0]].plot_order.values[0],
                                                                    'sector_order': self.sector_info[self.sector_info.sector == s].plot_order.values[0]},
                                                                   ignore_index=True)
            # Make sure the data is presented in the proper plot_order
            outputData = outputData.sort_values(by=['sector_order', 't_periods'])
            outputData = outputData.reset_index(drop=True)

        elif datatype == 'energy-flow':
            outputData = copy.copy(self.flow_output)
            outputData['tech_order'] = ''
            for t in outputData.tech.unique():
                outputData.loc[outputData.tech == t,
                               'tech_order'] = self.tech_info[self.tech_info.tech == t].plot_order.values[0]
            # Some entries of tech_order may be empty. Give these a value of 999
            outputData['tech_order'] = outputData['tech_order'].replace(r'^\s*$', 999, regex=True)
            outputData['tech_order'] = outputData['tech_order'].astype('float64')
            outputData['region_order'] = ''
            for r in outputData.region.unique():
                outputData.loc[outputData.region == r, 'region_order'] = self.region_info[self.region_info.region == r.split(
                    '-')[0]].plot_order.values[0]
            outputData['region_order'] = outputData['region_order'].replace(
                r'^\s*$', 999, regex=True)
            outputData['region_order'] = outputData['region_order'].astype('int64')
            tech_categories = list(self.tech_info.tech_category.unique())
            if '' in tech_categories:
                tech_categories.remove('')
            categories_to_tech = {}
            for tc in tech_categories:
                categories_to_tech[tc] = list(
                    self.tech_info[self.tech_info.tech_category == tc].tech.unique())

            def get_subsector(t):
                return self.tech_info[self.tech_info.tech == t].subsector.values[0]
            outputData['subsector'] = outputData['tech'].apply(get_subsector)
            outputData = outputData.sort_values(by=['tech_order', 'region_order'])
            outputData = outputData.sort_values(by=['timestamp', 'tech_order', 'region_order'])
            outputData = outputData.reset_index(drop=True)

        elif datatype == 'demands':
            df_cols = ['region', 't_periods', 'date', 'hour', 'demand_comm', 'value']
            outputData = pd.DataFrame(columns=df_cols)
            for y in self.annual_demand.t_periods.unique():
                for dc in self.dsd.demand_comm.unique():
                    for r in self.dsd[self.dsd.demand_comm == dc].region.unique():

                        _df = copy.copy(self.dsd[(self.dsd.demand_comm == dc) &
                                                 (self.dsd.region == r)])
                        demand = _df.value * \
                            self.annual_demand[
                                (self.annual_demand.region == r) &
                                (self.annual_demand.t_periods == y) &
                                (self.annual_demand.demand_comm == dc)].value.values[0]
                        _addition = {
                            'region': _df.region.values,
                            't_periods': [y]*len(_df.region.values),
                            'date': _df.date,
                            'hour': _df.hour,
                            'demand_comm': _df.demand_comm,
                            'value': demand
                        }
                        outputData = outputData.append(pd.DataFrame(_addition), ignore_index=True)

            outputData['timestamp'] = outputData['t_periods'].astype(
                str) + '-' + outputData['date'] + '-' + outputData['hour'].str[1:] + ':00'
            outputData['timestamp'] = pd.to_datetime(outputData['timestamp'])

        elif datatype == 'sectors':
            self.sectorMaps = {}  # will be a dictionary with sectors mapped to its subsectors
            for sector in self.sector_info.sector.unique():
                _techs = self.tech_info[self.tech_info.sector == sector]
                subsecs = _techs.subsector.unique()
                self.sectorMaps[sector] = list(subsecs)

            return

        return outputData

    def separate_imports_and_exports(self, df):
        '''
        Separates the transmission activity to imports and exports
        '''

        regions_included = [x for x in df.region.unique() if not '-' in x]
        regions_excluded = [x for x in self.region_info.region.unique(
        ) if not '-' in x and x not in regions_included]
        trans_regions = [x for x in df.region.unique() if '-' in x]

        import_trans_regions = []
        export_trans_regions = []
        for r in trans_regions:
            r1, r2 = r.split('-')
            if (r1 in regions_excluded) and (r2 in regions_included) and (r not in import_trans_regions):
                import_trans_regions.append(r)
            if (r1 in regions_included) and (r2 in regions_excluded) and (r not in export_trans_regions):
                export_trans_regions.append(r)

        df.loc[df.region.isin(export_trans_regions), 'value'] *= -1

        return df

    def generatePlotForCapacity(self, super_categories):
        '''
        Generates Plot for Capacity
        '''
        self.extractFromDatabase(type='capacity')

        df_en = self.processData(self.energy_output, 'capacity', super_categories)
        df_cap = self.processData(self.capacity_output, 'capacity', super_categories)
        self.processData(None, 'sectors', super_categories)

        self.makeCapacityDashboard(df_cap, df_en, super_categories)

        return

    def generatePlotForEnergyFlow(self, super_categories):
        '''
        Generates Plot for Energy Flow
        '''
        self.extractFromDatabase(type='flow-out')
        self.extractFromDatabase(type='demand-total')
        self.extractFromDatabase(type='seg-frac')
        self.extractFromDatabase(type='demand-specific-distribution')

        dff = self.processData('', 'energy-flow')
        dfd = self.processData('', 'demands')
        self.processData(None, 'sectors', super_categories)
        self.makeEnergyFlowDashboard(dff, dfd, super_categories)

        return

    def generatePlotForEmissions(self):
        '''
        Generates Plot for Emissions
        '''
        self.extractFromDatabase(type='emissions')

        df_em = self.processData(self.emissions_output,  'emissions')

        self.makeEmissionsDashboard(df_em)

        return

    def makeCapacityDashboard(self, df_cap, df_en, super_categories):
        '''
        Creates the structure of the capacity dashboard
        '''
        regions = list(df_cap.region.unique().__array__())
        sectors = df_cap.sector.unique()
        techs = df_cap.tech.unique()
        years = df_cap.t_periods.unique()
        scenarios = df_cap.scenario.unique()
        subsectors = self.tech_info.subsector.unique()
        demand_comms = df_en.output_comm.unique()

        comm_options = {}
        for s in sectors:
            comm_options[s] = {}
            for ss in df_en[df_en.sector == s].subsector.unique():
                comm_options[s][ss] = list(df_en[(df_en.sector == s) & (
                    df_en.subsector == ss)].output_comm.unique())

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

        app.layout = html.Div([
            html.Div([

                html.Div([
                    html.Label('Region(s):'),
                    dcc.Dropdown(
                        id='region-selector',
                        options=[{'label': i, 'value': i} for i in regions],
                        value=[r for r in regions if '-' not in r],
                        multi=True
                    ),
                    html.Label('Scenario:'),
                    dcc.Dropdown(
                        id='scenario-selector',
                        options=[{'label': i, 'value': i} for i in scenarios],
                        value=scenarios[0]
                    ),
                    dcc.RadioItems(
                        id='capacity-or-energy',
                        options=[{'label': i, 'value': i} for i in ['Capacity', 'Activity']],
                        value='Capacity',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                    style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Sector:'),
                    dcc.Dropdown(
                        id='sector-selector',
                        options=[{'label': i.capitalize(), 'value': i} for i in sectors],
                        value=sectors[0]
                    ),
                    html.Label('Subsector(s):'),
                    dcc.Dropdown(
                        id='subsector-selector', multi=True),

                    html.Label('Commodities'),
                    dcc.Dropdown(id='comm-selector', multi=True)

                ],
                    style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ],
                style={'padding': 60}
            ),

            dcc.Graph(id='indicator-graphic')
        ])

        @app.callback(
            Output('subsector-selector', 'options'),
            Input('sector-selector', 'value'))
        def set_subsector_options(selected_sector):
            _options = []
            for opt in self.sectorMaps[selected_sector]:
                if opt not in _options:
                    _options.append(opt)
            return [{'label': i, 'value': i} for i in _options]

        @app.callback(
            Output('subsector-selector', 'value'),
            Input('subsector-selector', 'options'))
        def set_subsector_value(ss_options):
            return [ss_options[0]['value']]

        @app.callback(
            Output('comm-selector', 'options'),
            Input('sector-selector', 'value'),
            Input('subsector-selector', 'value'))
        def set_commodity_options(sector, selected_subsectors):
            _options = []
            for ss in selected_subsectors:
                if ss in comm_options[sector]:
                    for opt in comm_options[sector][ss]:
                        if opt not in _options:
                            _options.append(opt)
            return [{'label': i, 'value': i} for i in _options]

        @app.callback(
            Output('comm-selector', 'value'),
            Input('comm-selector', 'options'))
        def set_commodity_value(comm_options):
            return [comm_options[0]['value']]

        @app.callback(
            Output('indicator-graphic', 'figure'),
            Input('sector-selector', 'value'),
            Input('subsector-selector', 'value'),
            Input('region-selector', 'value'),
            Input('scenario-selector', 'value'),
            Input('comm-selector', 'value'),
            Input('capacity-or-energy', 'value'))
        def update_graph(sector, subsector, region, scenario, comms, cap_or_en):
            if cap_or_en == 'Capacity':
                dff = df_cap[(df_cap['sector'] == sector) &
                             (df_cap['subsector'].isin(subsector)) &
                             (df_cap['region'].isin(region)) &
                             (df_cap['scenario'] == scenario)]
                years = df_cap.t_periods.unique()
                units = set()
                techs = dff.tech.unique()
                for t in techs:
                    units.add(self.tech_info[self.tech_info.tech == t].units.values[0])
                if len(units) > 1:
                    print('Warning: Attempting to plot technologies with different capacity units: ', units)

                unit = list(units)[0]

                yaxis_label = 'Capacity ' + '[' + str(unit) + ']'
                if len(region) == len(regions) and len(region) != 1:
                    title = sector.capitalize() + ' Sector Capacities for All Regions'
                else:
                    title = sector.capitalize() + ' Sector Capacities for '
                    if len(region) == 1:
                        title += region[0]
                    elif len(region) == 2:
                        title += region[0] + ' and ' + region[1]
                    else:
                        for r in region[:-1]:
                            title += r
                            title += ', '
                        title += 'and ' + region[-1]

            elif cap_or_en == 'Activity':

                dff = df_en[(df_en['sector'] == sector) &
                            (df_en['subsector'].isin(subsector)) &
                            (df_en['region'].isin(region)) &
                            (df_en['output_comm'].isin(comms)) &
                            (df_en['scenario'] == scenario)]

                years = df_en.t_periods.unique()

                dff = self.separate_imports_and_exports(dff)
                units = set()
                for c in comms:
                    units.add(
                        self.commodity_info[self.commodity_info.name == comms[0]].units.values[0])

                if len(units) > 1:
                    print('Warning: Attempting to plot technologies with different activity units:', units)

                unit = list(units)[0]

                yaxis_label = 'Activity ' + '[' + str(unit) + ']'
                if len(region) == len(regions) and len(region) != 1:
                    title = sector.capitalize() + ' Sector Activity for All Regions'
                else:
                    title = sector.capitalize() + ' Sector Activity for '
                    if len(region) == 1:
                        title += region[0]
                    elif len(region) == 2:
                        title += region[0] + ' and ' + region[1]
                    else:
                        for r in region[:-1]:
                            title += r
                            title += ', '
                        title += 'and ' + region[-1]

            fig = self.stacked_bar_plot(dff, years, yaxis_label, title,
                                        unit, 'tech', self.tech_info, super_categories)

            fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                              hovermode='x unified',
                              xaxis=dict(
                tickmode='array',
                tickvals=years,
                ticktext=years
            ))

            return fig
        app.run_server(debug=True)

    def makeEnergyFlowDashboard(self, dff, dfd, super_categories):
        '''
        Creates the structure of the energy flow dashboard
        '''
        scenarios = dff.scenario.unique()
        regions = dff.region.unique()
        years = dff.t_periods.unique()
        years.sort()
        techs = dff.tech.unique()
        dates = dff.date.unique()
        hours = dff.hour.unique()
        tstamps = dff.timestamp.unique()
        sectors = dff.sector.unique()
        subsectors = self.tech_info.subsector.unique()

        tx_regions = [r for r in regions if '-' in r]
        regions = [r for r in regions if '-' not in r]
        comm_options = {}

        for s in sectors:
            comm_options[s] = {}
            for ss in dff[dff.sector == s].subsector.unique():
                comm_options[s][ss] = list(
                    dff[(dff.sector == s) & (dff.subsector == ss)].output_comm.unique())

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app.layout = html.Div([
            html.Div([
                html.Div([
                    html.Label('Region(s):'),
                    dcc.Dropdown(
                        id='region-selector',
                        options=[{'label': i, 'value': i} for i in regions],
                        value=regions,
                        multi=True
                    ),
                    html.Label('Scenario:'),
                    dcc.Dropdown(
                        id='scenario-selector',
                        options=[{'label': i, 'value': i} for i in scenarios],
                        value=scenarios[0]
                    ),
                    html.Label('Year:'),
                    dcc.Dropdown(
                        id='year-selector',
                        options=[{'label': i, 'value': i} for i in years],
                        value=years[0]
                    ),
                ],
                    style={'width': '40%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('Sector:'),
                    dcc.Dropdown(
                        id='sector-selector',
                        options=[{'label': i.capitalize(), 'value': i} for i in sectors],
                        value=sectors[0]
                    ),
                    html.Label('Subsector(s):'),
                    dcc.Dropdown(
                        id='subsector-selector', multi=True),

                    html.Label('Commodities'),
                    dcc.Dropdown(id='comm-selector', multi=True)
                ],
                    style={'width': '40%', 'display': 'inline-block'})
            ],
                style={'padding': 60}
            ),

            dcc.Graph(id='indicator-graphic')
        ])

        @app.callback(
            Output('subsector-selector', 'options'),
            Input('sector-selector', 'value'))
        def set_subsector_options(selected_sector):
            _options = []
            for opt in self.sectorMaps[selected_sector]:
                if opt not in _options:
                    _options.append(opt)
            return [{'label': i, 'value': i} for i in _options]

        @app.callback(
            Output('subsector-selector', 'value'),
            Input('subsector-selector', 'options'))
        def set_subsector_value(ss_options):
            return [ss_options[0]['value']]

        @app.callback(
            Output('comm-selector', 'options'),
            Input('sector-selector', 'value'),
            Input('subsector-selector', 'value'))
        def set_commodity_options(sector, selected_subsectors):
            _options = []
            for ss in selected_subsectors:
                if ss in comm_options[sector]:
                    for opt in comm_options[sector][ss]:
                        if opt not in _options:
                            _options.append(opt)
            return [{'label': i, 'value': i} for i in _options]

        @app.callback(
            Output('comm-selector', 'value'),
            Input('comm-selector', 'options'))
        def set_commodity_value(comm_options):
            return [comm_options[0]['value']]

        @app.callback(
            Output('indicator-graphic', 'figure'),
            Input('scenario-selector', 'value'),
            Input('region-selector', 'value'),
            Input('year-selector', 'value'),
            Input('sector-selector', 'value'),
            Input('subsector-selector', 'value'),
            Input('comm-selector', 'value'))
        def update_graph(scenario, region, year, sector, subsectors, comms):

            _dff = dff[
                (dff['sector'] == sector) &
                (dff['subsector'].isin(subsectors)) &
                (dff['scenario'] == scenario) &
                (dff['region'].isin(region)) &
                (dff['output_comm'].isin(comms)) &
                (dff['t_periods'] == year)]
            _tx_regions = []

            for txr in tx_regions:
                include = 0
                r1, r2 = txr.split('-')
                if r1 in region:
                    include += 1
                if r2 in region:
                    include += 1
                if include == 1:
                    _tx_regions.append(txr)

            _dfb = dff[
                (dff['sector'] == sector) &
                (dff['subsector'].isin(subsectors)) &
                (dff['scenario'] == scenario) &
                (dff['region'].isin(_tx_regions)) &
                (dff['output_comm'].isin(comms)) &
                (dff['t_periods'] == year)]
            _dfb['R1'] = _dfb.region.str.split('-').str[0]
            _dfb['R2'] = _dfb.region.str.split('-').str[1]

            units = set()
            for c in comms:
                units.add(self.commodity_info[self.commodity_info.name == comms[0]].units.values[0])

            if len(units) > 1:
                print('Warning: Attempting to plot technologies with different activity units:', units)

            unit = list(units)[0]
            yaxis_label = 'Activity Level [' + unit + ']'
            if len(region) == len(regions) and len(region) != 1:
                title = sector.capitalize() + ' Sector Activity for All Regions'
            else:
                title = sector.capitalize() + ' Sector Activity for '
                if len(region) == 1:
                    title += region[0]
                elif len(region) == 2:
                    title += region[0] + ' and ' + region[1]
                else:
                    for r in region[:-1]:
                        title += r
                        title += ', '
                    title += 'and ' + region[-1]

            fig = self.stacked_line_plot(_dff, _dfb, yaxis_label, title,
                                         techs, unit, super_categories)

            fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                              hovermode='x unified')
            return fig

        app.run_server(debug=True)

    def makeEmissionsDashboard(self, df):
        '''
        Creates the structure of the emissions dashboard
        '''
        scenarios = df.scenario.unique()
        regions = df.region.unique()
        sectors = df.sector.unique()
        emission_comms = df.emissions_comm.unique()
        years = df.t_periods.unique()

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app.layout = html.Div([
            html.Div([

                html.Div([
                    html.Label('Region(s):'),
                    dcc.Dropdown(
                        id='region-selector',
                        options=[{'label': i, 'value': i} for i in regions],
                        value=regions,
                        multi=True
                    ),
                    html.Label('Scenario:'),
                    dcc.Dropdown(
                        id='scenario-selector',
                        options=[{'label': i, 'value': i} for i in scenarios],
                        value=scenarios[0]
                    )
                ],
                    style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Sector(s):'),
                    dcc.Dropdown(
                        id='sector-selector',
                        options=[{'label': i.capitalize(), 'value': i} for i in sectors],
                        value=sectors,
                        multi=True
                    ),
                    html.Label('Commodity:'),
                    dcc.Dropdown(
                        id='commodity-selector',
                        options=[{'label': i, 'value': i} for i in emission_comms],
                        value=emission_comms[0]
                    )

                ],
                    style={'width': '48%', 'display': 'inline-block'})
            ]),

            dcc.Graph(id='indicator-graphic')
        ])

        @app.callback(
            Output('indicator-graphic', 'figure'),
            Input('sector-selector', 'value'),
            Input('scenario-selector', 'value'),
            Input('region-selector', 'value'),
            Input('commodity-selector', 'value'))
        def update_graph(sector, scenario, region, comm):

            dff = df[(df['sector'].isin(sector)) &
                     (df['region'].isin(region)) &
                     (df['scenario'] == scenario) &
                     (df['emissions_comm'] == comm)]
            unit = self.commodity_info[self.commodity_info.name == comm].units.values[0]
            yaxis_label = 'Emissions (kt)'
            years = df.t_periods.unique()
            years.sort()

            if len(region) == len(regions) and len(region) != 1:
                title = 'Annual ' + comm + ' Emissions for All Regions'
            else:
                title = 'Annual ' + comm + ' Emissions for '
                if len(region) == 1:
                    title += region[0]
                elif len(region) == 2:
                    title += region[0] + ' and ' + region[1]
                else:
                    for r in region[:-1]:
                        title += r
                        title += ', '
                    title += 'and ' + region[-1]

            fig = self.stacked_bar_plot(dff, years, yaxis_label, title,
                                        unit, 'sector', self.sector_info)

            fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                              hovermode='x unified',
                              xaxis=dict(
                tickmode='array',
                tickvals=years,
                ticktext=years
            ))

            return fig
        app.run_server(debug=True)

    def stacked_bar_plot(self, data, years, ylabel, title, unit, column_to_stack, stack_descriptor, super_categories=False):
        '''
        Returns a stacked bar plot.
        x-axis: Years
        y-axis: Tech capacities
        '''

        fig_data = []
        xlabel = 'Year'
        fig = go.Figure()
        if 'tech_order' in data:
            to_stack = data.sort_values(by=['tech_order', 'region_order'])[column_to_stack].unique()
        else:
            to_stack = data[column_to_stack].unique()
        for st in to_stack:
            _d = data[(data[column_to_stack] == st)]
            vals = np.asarray(([0.0]*years))

            for i, y in enumerate(years):
                __d = _d[_d.t_periods == y]
                vals[i] = sum(__d.value.values)

            if super_categories:
                tech_category = self.tech_info[self.tech_info.tech == st].tech_category.values[0]
                label = tech_category
                if label == '':
                    label = stack_descriptor[stack_descriptor[column_to_stack]
                                             == st].description.values[0]
            else:
                label = stack_descriptor[stack_descriptor[column_to_stack]
                                         == st].description.values[0]

            color = stack_descriptor[stack_descriptor[column_to_stack] == st].color.values[0]

            if matplotlib.colors.is_color_like(color):
                fig.add_trace(go.Bar(x=years,
                                     y=vals,
                                     marker_color=stack_descriptor[stack_descriptor[column_to_stack]
                                                                   == st].color.values[0],
                                     name=label,
                                     text=[label +
                                           ': '+str(round(_, 2))+' ' + unit for _ in vals],
                                     hoverinfo='text'
                                     ))

            else:
                fig.add_trace(go.Bar(x=years,
                                     y=vals,
                                     name=label,
                                     text=[label +
                                           ': '+str(round(_, 2))+' ' + unit for _ in vals],
                                     hoverinfo='text'
                                     ))

        fig.update_layout(barmode='relative',
                          title=title,
                          yaxis_title=ylabel,
                          xaxis_title=xlabel,
                          legend={'traceorder': 'reversed'})

        fig.update_traces(marker=dict(size=12,
                                      line=dict(width=0,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        return fig

    def stacked_line_plot(self, d_gen, d_tx, ylabel, title, techs, unit, super_categories):
        def individual_plot(_d_gen, _d_tx, sf):

            annual_to_date_modifier = 1/(sf.value.values[0] * 365 * 24)

            fig_data = []

            base = pd.DataFrame({'timestamp': _d_gen.timestamp.unique()}).set_index('timestamp')
            base['value'] = 0
            # Convert 'base' to a Series
            base = base.value

            _d_gen = _d_gen.sort_values(by=['tech_order', 'region_order'])
            _d_gen = _d_gen.reset_index(drop=True)

            for t in d_gen.sort_values(by=['tech_order', 'region_order']).tech.unique():

                # for r in d_gen.region.unique():
                _d = _d_gen[(_d_gen.tech == t)]  # & (_d_gen.region == r)]
                if 'exports' in _d.subsector.unique():
                    continue
                if t == 'E_EXPORT':
                    continue

                _d = _d.drop(columns=['scenario', 'sector', 'region', 't_periods', 'date', 'hour',
                                      'input_comm', 'output_comm', 'vflow_out', 'subsector'])
                _d = _d.groupby(_d.timestamp).agg({'tech': 'first',
                                                   'curtailment': sum,
                                                   'value': sum,
                                                   'tech_order': 'first',
                                                   'region_order': 'first'})
                vals = pd.DataFrame({'timestamp': _d_gen.timestamp.unique()}).set_index('timestamp')
                vals['value'] = 0
                vals2 = _d.value
                vals = pd.concat([vals, vals2], axis=1).fillna(0).sum(axis=1)
                vals = vals * annual_to_date_modifier

                if super_categories:
                    tech_category = self.tech_info[self.tech_info.tech == t].tech_category.values[0]
                    label = tech_category
                else:
                    label = self.tech_info[self.tech_info.tech == t].description.values[0]
                color = self.tech_info[self.tech_info['tech'] == t].color.values[0]

                if matplotlib.colors.is_color_like(color):
                    fig.add_trace(go.Bar(x=vals.index,
                                         y=vals.values,
                                         base=base.values,
                                         offsetgroup=0,
                                         marker_color=color,
                                         name=label,
                                         text=[label +
                                               ': '+str(round(_, 2))+' ' + unit for _ in vals],
                                         hoverinfo='text',
                                         legendgroup='group1',
                                         showlegend=not col
                                         ),
                                  row=1, col=col+1)

                else:
                    fig.add_trace(go.Bar(x=vals.index,
                                         y=vals.values,
                                         base=base.values,
                                         offsetgroup=0,
                                         name=label,
                                         text=[label +
                                               ': '+str(round(_, 2))+' ' + unit for _ in vals],
                                         hoverinfo='text',
                                         legendgroup='group1',
                                         showlegend=not col
                                         ),
                                  row=1, col=col+1)

                base = base.add(vals)

            # Imports
            for r2 in _d_tx.R2.unique():
                if r2 not in _d_gen.region.unique():
                    continue
                _d = _d_tx[_d_tx.R2 == r2]
                _d = _d.drop(columns=['scenario', 'sector', 'region', 't_periods', 'date', 'hour',
                                      'input_comm', 'output_comm', 'subsector'])
                _d = _d.groupby(_d.timestamp).agg({'tech': 'first',
                                                   'curtailment': sum,
                                                   'value': sum,
                                                   'tech_order': 'first',
                                                   'region_order': 'first',
                                                   'R1': 'first',
                                                   'R2': 'first'})

                vals = pd.DataFrame({'timestamp': _d_gen.timestamp.unique()}).set_index('timestamp')
                vals['value'] = 0
                vals2 = _d.value
                vals = pd.concat([vals, vals2], axis=1).fillna(0).sum(axis=1)
                vals = vals * annual_to_date_modifier

                label = 'Imports (Inter-regional)'
                color = '#cc76d6'

                fig.add_trace(go.Bar(x=vals.index,
                                     y=vals.values,
                                     base=base.values,
                                     offsetgroup=0,
                                     marker_color=color,
                                     name=label,
                                     text=[label +
                                           ': '+str(round(_, 2))+' ' + unit for _ in vals],
                                     hoverinfo='text',
                                     legendgroup='group1',
                                     showlegend=not col
                                     ),
                              row=1, col=col+1)
                base = base.add(vals)

            # Inter-regional Exports
            base2 = pd.DataFrame({'timestamp': _d_gen.timestamp.unique()}).set_index('timestamp')
            base2['value'] = 0
            # Convert 'base' to a Series
            base2 = base2.value
            for r1 in _d_tx.R1.unique():
                if r1 not in _d_gen.region.unique():
                    continue
                _d = _d_tx[_d_tx.R1 == r1]
                _d = _d.drop(columns=['scenario', 'sector', 'region', 't_periods', 'date', 'hour',
                                      'input_comm', 'output_comm', 'subsector'])
                _d = _d.groupby(_d.timestamp).agg({'tech': 'first',
                                                   'curtailment': sum,
                                                   'value': sum,
                                                   'tech_order': 'first',
                                                   'region_order': 'first',
                                                   'R1': 'first',
                                                   'R2': 'first'})
                vals = pd.DataFrame({'timestamp': _d_gen.timestamp.unique()}).set_index('timestamp')
                vals['value'] = 0
                vals2 = _d.value
                vals = pd.concat([vals, vals2], axis=1).fillna(0).sum(axis=1)
                vals = -vals * annual_to_date_modifier

                label = 'Exports (Inter-regional)'
                color = '#7d3c85'

                fig.add_trace(go.Bar(x=vals.index,
                                     y=vals.values,
                                     base=base2.values,
                                     offsetgroup=0,
                                     marker_color=color,
                                     name=label,
                                     text=[label +
                                           ': '+str(round(_, 2))+' ' + unit for _ in vals],
                                     hoverinfo='text',
                                     legendgroup='group1',
                                     showlegend=not col
                                     ),
                              row=1, col=col+1)

                base2 = base2.add(vals)

            # Exports to outside modelled regions.
            for t in d_gen[d_gen.subsector == 'exports'].sort_values(by=['tech_order', 'region_order']).tech.unique():

                _d = _d_gen[(_d_gen.tech == t)]

                _d = _d.drop(columns=['scenario', 'sector', 'region', 't_periods', 'date', 'hour',
                                      'input_comm', 'output_comm', 'vflow_out', 'subsector'])
                _d = _d.groupby(_d.timestamp).agg({'tech': 'first',
                                                   'curtailment': sum,
                                                   'value': sum,
                                                   'tech_order': 'first',
                                                   'region_order': 'first'})
                vals = pd.DataFrame({'timestamp': _d_gen.timestamp.unique()}).set_index('timestamp')
                vals['value'] = 0
                vals2 = _d.value
                vals = pd.concat([vals, vals2], axis=1).fillna(0).sum(axis=1)
                vals = -vals * annual_to_date_modifier

                label = 'Exports (Extra-regional)'
                color = 'black'

                fig.add_trace(go.Bar(x=vals.index,
                                     y=vals.values,
                                     base=base2.values,
                                     offsetgroup=0,
                                     marker_color=color,
                                     name=label,
                                     text=[label +
                                           ': '+str(round(_, 2))+' ' + unit for _ in vals],
                                     hoverinfo='text',
                                     legendgroup='group1',
                                     showlegend=not col
                                     ),
                              row=1, col=col+1)

                base2 = base2.add(vals)

            # Curtailment!
            _d = copy.copy(_d_gen)
            _d = _d.drop(columns=['scenario', 'sector', 'region', 't_periods', 'date', 'hour',
                                  'input_comm', 'output_comm', 'vflow_out', 'subsector'])
            _d = _d.groupby(_d.timestamp).agg({'tech': 'first',
                                               'curtailment': sum,
                                               'value': sum,
                                               'tech_order': 'first',
                                               'region_order': 'first'})
            vals = pd.DataFrame({'timestamp': _d_gen.timestamp.unique()}).set_index('timestamp')
            vals['value'] = 0
            vals2 = _d.curtailment
            vals = pd.concat([vals, vals2], axis=1).fillna(0).sum(axis=1)
            vals = -vals * annual_to_date_modifier
            label = 'Curtailment'
            color = '#828282'
            if sum(vals.values) >= 0.0001:
                fig.add_trace(go.Bar(x=vals.index,
                                     y=vals.values,
                                     base=base2.values,
                                     offsetgroup=0,
                                     marker_color=color,
                                     name=label,
                                     text=[label +
                                           ': '+str(round(_, 2))+' ' + unit for _ in vals],
                                     hoverinfo='text',
                                     legendgroup='group1',
                                     showlegend=not col
                                     ),
                              row=1, col=col+1)

                base2 = base2.add(vals)

        fig = make_subplots(rows=1, cols=self.ndays,
                            shared_yaxes='all', horizontal_spacing=0.01)

        for col, date in enumerate(self.segfrac.date.unique()):

            _d_gen = d_gen[d_gen.date == date]
            _d_tx = d_tx[d_tx.date == date]
            sf = self.segfrac[self.segfrac.date == date]

            individual_plot(_d_gen, _d_tx, sf)

        fig.update_layout(title_text=title, legend={'traceorder': 'reversed'})
        fig.update_yaxes(title_text=ylabel, row=1, col=1)

        return fig


# Function used for command line purposes. Parses arguments and then calls relevent functions.
def GeneratePlot(args):
    parser = argparse.ArgumentParser(description="Generate Output Plot")
    parser.add_argument('-i', '--input', action="store", dest="input",
                        help="Input Database Filename <path>", required=True)
    parser.add_argument('-p', '--plot-type', action="store", dest="type",
                        help="Type of Plot to be generated", choices=['capacity', 'flow', 'emissions'], required=True)
    parser.add_argument('--super', action="store_true", dest="super_categories",
                        help="Merge Technologies or not", default=False)

    options = parser.parse_args(args)
    result = OutputPlotGenerator(options.input, options.type)
    error = ''  # RV

    if (options.type == 'capacity'):
        error = result.generatePlotForCapacity(options.super_categories)
    elif (options.type == 'flow'):
        error = result.generatePlotForEnergyFlow(options.super_categories)
    elif (options.type == 'emissions'):
        error = result.generatePlotForEmissions()


begin = time.time()
def duration(): return time.time() - begin


if __name__ == '__main__':
    GeneratePlot(sys.argv[1:])
