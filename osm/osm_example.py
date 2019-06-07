import OsmPy as osm
import networkx as nx
import pandas as pd


G = nx.gnp_random_graph(100, 0.02)

belief_update_filter = osm.cheat_dice_particle_filter(sample_num = 10, cheat_rate = 0.5)
#belief_update_filter = osm.cheat_dice_bayse_filter(cheat_rate = 0.5)
A = osm.cheat_dice_agents(G, accept_rate = 0.2, threshold = 0.9, belief_update_fileter = belief_update_filter)
E = osm.cheat_dice_environment(face_num = 100, cheat_rate = 0.5, cheat_state = 4)
#E = osm.cheat_dice_environment(face_num = 100, custom_distribute = {0:0.4, 1:0.3})
#E = osm.normal_environment(mu = 0, sigma = 1, size = 1000)
#E = osm.linear_regression_environment(intercept = 1, coefficient = 0.5)

S = A.get_random_sensors(num = 10)
#S = A.get_random_sensors(proportion = 0.1)

step_infos = osm.agents_step_infos()
init_agents_info = A.get_info(has_opinion = True, 
                             has_receive_number = True,
                             has_init_opinion = True,
                             has_init_belief = True)
step_infos.add(init_agents_info)

max_step = 3000
step_duration = 100
sensor_observe_rate = 0.1
sensor_observe_duration = 10
max_observe_number = 3 * len(S)

for s in range(0, max_step, step_duration):
    A = osm.update_step(step = step_duration,
                        agents = A, 
                        environment = E, 
                        sensors = S,
                        sensor_observe_rate = sensor_observe_rate,
                        sensor_observe_duration = sensor_observe_duration)

    step_infos.add(A.get_info(has_opinion = True, 
                              has_receive_number = True,
                              has_init_opinion = True,
                              has_init_belief = True))

