import os
import shutil

from utils.utils_sokoban import save_state


class GraphTracerSokoban:
    def __init__(self,
                 output_dir,
                 solution_node_color='green',
                 expanded_edge_color='rgb(255,168,7)',
                 unexpanded_edge_color='rgb(255,168,7)'):

        self.output_dir = output_dir
        self.solution_node_color = solution_node_color
        self.expanded_edge_color = expanded_edge_color
        self.unexpanded_edge_color = unexpanded_edge_color

    def draw_graph(self, solution, solver_nodes, edges, extra_edges):

        def node_color(node_id, solution_ids):
            if node_id in solution_ids:
                return self.solution_node_color
            else:
                return self.expanded_edge_color

        if solution is not None:
            solution_ids = [node.id for node in solution]
        else:
            solution_ids = []
        nodes = {node.id: {'label': f'{node.id}', 'level': node.level, 'color': node_color(node.id, solution_ids)} for
                 node in solver_nodes.values()}
        info = {}
        state_src = {}
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        save_state(solver_nodes[0].state, os.path.join(self.output_dir, 'images/root.png'))
        for id, node in solver_nodes.items():
            info[id] = 'Node value = {:>.4f}'.format(node.value)
            relative_state_path = f'images/state_{id}.png'
            full_state_path = os.path.join(self.output_dir, relative_state_path)
            save_state(node.state, full_state_path)

            state_src[id] = relative_state_path

        self.generate_website_solver_sokoban('Solver graph visualization', nodes, edges, extra_edges, state_src, info, self.output_dir)

    def draw_goal_generation(self, generator_nodes, edges):
        nodes = {node.id: {'label': f'{node.id}', 'level': node.level} for node in generator_nodes}
        info = {}
        partial_src = {}
        parent_src = {}
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        save_state(generator_nodes[0].input_board, os.path.join(self.output_dir, 'images/root.png'))
        for node in generator_nodes:
            id = node.id
            info[id] = 'Node probability = {:>.3f}'.format(node.p)
            relative_state_path = f'images/state_{id}.png'
            partial_state_path = os.path.join(self.output_dir, relative_state_path)
            save_state(node.condition, partial_state_path)

            relative_parent_path = f'images/parent_{id}.png'
            parent_path = os.path.join(self.output_dir, f'images/parent_{id}.png')
            if node.parent is not None:
                save_state(node.parent.condition, parent_path)
            else:
                save_state(node.condition, parent_path)

            partial_src[id] = relative_state_path
            parent_src[id] = relative_parent_path

            self.generate_website_goal_builder_sokoban('Goal building visualization', nodes, edges,  partial_src, parent_src,
                                              info, self.output_dir)


    def generate_website_solver_sokoban(self, title, nodes, edges, extra_edges, state_src, info, dump_folder):
        code = ''
        code += f'var title = "{title}"; \n'
        code += 'var state_src = { \n'
        for key, val in state_src.items():
            code += f'{key} : "{val}", \n'
        code += '}; \n'

        code += 'var info = { \n'
        for key, val in info.items():
            code += f'{key} : "{val}", \n'
        code += '}; \n'
        code += 'nodes = []; \n'
        code += 'edges = []; \n'

        for id, node in nodes.items():
            code += 'nodes.push({' + f'id: {id}, label: "' + \
                    f'{node["label"]}' + \
                    f'", level: {3*node["level"]}, color: "' + f'{node["color"]}' + '"}); \n'

        for edge in edges:
            color = 'black'
            code += 'edges.push({' + f'from: {edge[0]}, to: {edge[1]}, label: "{edge[2]}", arrows: "to", color: "' + f'{color}"' + '}); \n'

        for edge in extra_edges:
            color = 'gray'
            code += 'edges.push({' + f'from: {edge[0]}, to: {edge[1]}, label: "{edge[2]}", arrows: "to", color: "' + f'{color}"' + '}); \n'

        data_file = open(os.path.join(dump_folder, 'data.js'), 'w')
        data_file.write(code)


        shutil.copy2('graph_tracer/assets/solver_graph.html',
                     os.path.join(dump_folder, 'solver_graph.html'))
        shutil.copy2('graph_tracer/assets/vis-network.min.js',
                     os.path.join(dump_folder, 'vis-network.min.js'))



    def generate_website_goal_builder_sokoban(self, title, nodes, edges, partial_src, parent_src, info, dump_folder):
        code = ''
        code += f'var title = "{title}"; \n'
        code += 'var partial_src = { \n'
        for key, val in partial_src.items():
            code += f'{key} : "{val}", \n'
        code += '}; \n'

        code += 'var parent_src = { \n'
        for key, val in parent_src.items():
            code += f'{key} : "{val}", \n'
        code += '}; \n'

        code += 'var info = { \n'
        for key, val in info.items():
            code += f'{key} : "{val}", \n'
        code += '}; \n'

        code += 'nodes = []; \n'
        code += 'edges = []; \n'

        for id, node in nodes.items():
            code += 'nodes.push({' + f'id: {id}, label: "' + f'{node["label"]}' + f'", level: {node["level"]}'+ '}); \n'

        for edge in edges:
            code += 'edges.push({' + f'from: {edge[0]}, to: {edge[1]}, label: "{edge[2]}"' + '}); \n'

        data_file = open(os.path.join(dump_folder, 'data.js'), 'w')
        data_file.write(code)

        shutil.copy2('graph_tracer/assets/goal_generation.html', os.path.join(dump_folder, 'goal_generation.html'))
        shutil.copy2('graph_tracer/assets/vis-network.min.js', os.path.join(dump_folder, 'vis-network.min.js'))
