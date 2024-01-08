import pandapipes as pp
import pandas as pd
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent + "\\plotting_help")
import plotting_helper_functions as phf

# ---------------------------------------

src_folder = "grid-data"
junctions_file = src_folder + "/junctions.csv"
pipes_file = src_folder + "/pipes.csv"
valves_file = src_folder + "/valves.csv"
sinks_file = src_folder + "/sinks.csv"
sources_file = src_folder + "/sources.csv"
pumps_file = src_folder + "/pumps.csv"
ext_grids_file = src_folder + "/external_grids.csv"

net = pp.create_empty_network(fluid="hgas")

junctions = pd.read_table(
    junctions_file, sep=","
)  # WICHTIG, dass "," als Trennzeichen genutzt wird!
for index, row in junctions.iterrows():
    junction_created = pp.create_junction(
        net,
        pn_bar=row["pressure"],
        name=row["name"],
        tfluid_k=row["temp"],
        geodata=(row["xpos"], row["ypos"]),
    )
    if row["control"] == "Slack":
        pp.create_ext_grid(net, junction=junction_created, vm_pu=1.0, name="Grid")

pipes = pd.read_table(
    pipes_file, sep=","
)  # WICHTIG, dass "," als Trennzeichen genutzt wird!
for index, row in pipes.iterrows():
    pipe_created = pp.create_pipe_from_parameters(
        net,
        name=row["name"],
        from_junction=row["fromj"] - 1,
        # std_type=row['standardtype'],
        to_junction=row["toj"] - 1,
        length_km=row["length"],
        diameter_m=row["diam"],
        k_mm=row["roughness"],
        sections=5,
    )

sources = pd.read_table(sources_file, sep=",")
for index, row in sources.iterrows():
    source_created = pp.create_source(
        net,
        name=row["name"],
        junction=row["atj"] - 1,
        mdot_kg_per_s=row["enter_mass_flow"],
    )

sinks = pd.read_table(sinks_file, sep=",")
for index, row in sinks.iterrows():
    sink_created = pp.create_sink(
        net,
        name=row["name"],
        junction=row["atj"] - 1,
        mdot_kg_per_s=row["exit_mass_flow"],
    )

external_grids = pd.read_table(ext_grids_file, sep=",")
for index, row in external_grids.iterrows():
    sink_created = pp.create_ext_grid(
        net,
        name=row["name"],
        junction=row["atj"] - 1,
        p_bar=row["pressure"],
        t_k=row["temp"],
    )


pp.pipeflow(net)
print("net.res_junction: \n", net.res_junction, "\n")
print("net.res_pipe: \n", net.res_pipe, "\n")


cmap_list = [
    ((25, 28), "green"),
    ((28, 31), "yellowgreen"),
    ((31, 34), "yellow"),
    ((34, 37), "orange"),
    ((37, 40), "red"),
]
cmap, norm = phf.cmap_discrete(cmap_list)

jc = plot.create_junction_collection(
    net, net.junction.index, size=0.1, zorder=2, cmap=cmap, norm=norm
)

cmap_list = [
    ((-10, 0), "silver"),
    ((0, 5), "green"),
    ((5, 7), "yellowgreen"),
    ((7, 9), "yellow"),
    ((9, 11), "red"),
]
cmap, norm = phf.cmap_discrete(cmap_list)

pc = plot.create_pipe_collection(
    net, net.pipe.index, zorder=1, linewidths=4, cmap=cmap, norm=norm
)

plot.draw_collections([pc, jc], figsize=(4, 3))
plt.show()
