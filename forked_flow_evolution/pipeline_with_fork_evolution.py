import pandapipes as pp 
import pandas as pd
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
import sys
import os 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent+'\\plotting_help')
import plotting_helper_functions as phf 
import pandapower.control as control
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapipes.timeseries import run_timeseries
#---------------------------------------

src_folder = "grid-data"
junctions_file = src_folder + "/junctions.csv"
pipes_file = src_folder + "/pipes.csv"
valves_file = src_folder + "/valves.csv"
sinks_file = src_folder + "/sinks.csv"
sources_file = src_folder + "/sources.csv"
pumps_file = src_folder + "/pumps.csv"
ext_grids_file = src_folder + "/external_grids.csv"

net = pp.create_empty_network(fluid='hgas')

junctions = pd.read_table(junctions_file, sep=",") # WICHTIG, dass "," als Trennzeichen genutzt wird!
for index, row in junctions.iterrows():
    junction_created = pp.create_junction(net,
                        pn_bar=row['pressure'],
                        name=row['name'],
                        tfluid_k=row['temp'], 
                        geodata=(row['xpos'], row['ypos']))
    if row['control'] == "Slack":
       pp.create_ext_grid(net, 
       					junction=junction_created,
       					vm_pu=1.0, name="Grid")

pipes = pd.read_table(pipes_file, sep=",") # WICHTIG, dass "," als Trennzeichen genutzt wird!
for index, row in pipes.iterrows():
    pipe_created = pp.create_pipe_from_parameters(net,
                        name=row['name'],
                        from_junction=row['fromj']-1,
                        #std_type=row['standardtype'],
                        to_junction=row['toj']-1,
                        length_km=row['length'], 
                        diameter_m=row['diam'],
                        k_mm=row['roughness'],
                        sections=5
                        )

sources = pd.read_table(sources_file, sep=",") 
for index, row in sources.iterrows():
    source_created = pp.create_source(net,
                        name=row['name'],
                        junction=row['atj']-1,
                        mdot_kg_per_s=row['enter_mass_flow']
                        )

sinks = pd.read_table(sinks_file, sep=",") 
for index, row in sinks.iterrows():
    sink_created = pp.create_sink(net,
                        name=row['name'],
                        junction=row['atj']-1,
                        mdot_kg_per_s=row['exit_mass_flow']
                        )

external_grids = pd.read_table(ext_grids_file, sep=",") 
for index, row in external_grids.iterrows():
    sink_created = pp.create_ext_grid(net, 
                        name=row['name'],
                        junction=row['atj']-1,
                        p_bar=row['pressure'], 
                        t_k=row['temp'],
                        )


sinks_profile = src_folder + "/time_series_sinks.csv"
profiles_sink = pd.read_table(sinks_profile, sep=",", index_col=0)
sources_profile = src_folder + "/time_series_sources.csv"            
profiles_source = pd.read_table(sources_profile, sep=",", index_col=0) 

ds_sink = DFData(profiles_sink)
ds_source = DFData(profiles_source)
print(ds_sink)

const_sink = control.ConstControl(net, element='sink', variable='mdot_kg_per_s',
                                  element_index=net.sink.index.values, data_source=ds_sink,
                                  profile_name=net.sink.index.values.astype(str))
const_source = control.ConstControl(net, element='source', variable='mdot_kg_per_s',
                                    element_index=net.source.index.values,
                                    data_source=ds_source,
                                    profile_name=net.source.index.values.astype(str))

time_steps = range(5)

log_variables = [('res_junction', 'p_bar'), ('res_pipe', 'v_mean_m_per_s'),
                 ('res_pipe', 'reynolds'), ('res_pipe', 'lambda'),
                 ('res_sink', 'mdot_kg_per_s'), ('res_source', 'mdot_kg_per_s'),
                 ('res_ext_grid', 'mdot_kg_per_s')]
ow = OutputWriter(net, time_steps, output_path=None, log_variables=log_variables)

run_timeseries(net, time_steps)
print("pressure:")
print(ow.np_results["res_junction.p_bar"])
print("mean velocity:")
print(ow.np_results["res_pipe.v_mean_m_per_s"])
print("reynolds number:")
print(ow.np_results["res_pipe.reynolds"])
print("lambda:")
print(ow.np_results["res_pipe.lambda"])
print("mass flow sink:")
print(ow.np_results["res_sink.mdot_kg_per_s"])
print("mass flow source:")
print(ow.np_results["res_source.mdot_kg_per_s"])
print("mass flow ext. grid:")
print(ow.np_results["res_ext_grid.mdot_kg_per_s"])