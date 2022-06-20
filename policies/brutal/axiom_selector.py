import re
from copy import deepcopy

from envs.int.theorem_prover_env import TheoremProverEnv
from logic.logic import Entity
from proof_system.logic_functions import necessary_logic_functions
from proof_system.meta_axiom import MetaAxiom
from proof_system.numerical_functions import necessary_numerical_functions
from proof_system.utils import is_entity, is_structured, substitution, search_operator_operands_in_gt, \
    side_of_an_entity, is_ls_type
from visualization.seq_parse import logic_statement_to_seq_string, entity_to_seq_string


def search_for_subgoal_bfs(state, subgoal_str, ground_truth, max_dist=2, debugg=False):
    env = TheoremProverEnv()
    current_state_str = logic_statement_to_seq_string(state['observation']['objectives'][0])
    seen_states = {current_state_str}
    queue = [(deepcopy(state), 0, [])]
    step = 0
    depth = 0
    while len(queue) > 0 and depth < max_dist+1:
        step += 1
        current_state_original, depth, parent_actions = queue.pop(0)
        current_state = deepcopy(current_state_original)
        current_state_str = logic_statement_to_seq_string(current_state['observation']['objectives'][0])
        all_actions = prepare_action_space(current_state['observation']['objectives'][0], subgoal_str, ground_truth, debugg)
        if debugg:
            print(f' All actions My state for actions = {logic_statement_to_seq_string(current_state["observation"]["objectives"][0])}')
            for action in all_actions:
                print(f'{action[0]} | inputs = {[entity_to_seq_string(x) for x in action[1:]]}')
            return True

        new_statement_str = ''
        new_state = current_state
        for action in all_actions:
            new_state_to_load = new_state
            if new_statement_str != current_state_str:
                new_state_to_load = deepcopy(current_state)
            env.load_problem_step(new_state_to_load)
            new_state, _, _, _ = env.step(action)
            new_statement = new_state['observation']['objectives'][0]
            new_statement_str = logic_statement_to_seq_string(new_statement)
            if new_statement_str == subgoal_str:
                return parent_actions + [action]
            elif new_statement_str not in seen_states:
                seen_states.add(new_statement_str)
                new_actions = parent_actions + [action]
                if depth+1 < max_dist:
                    queue.append((new_state, depth+1, new_actions))

    return None


def search_for_subgoal(state, subgoal):
    # seen_states = set()
    all_actions = prepare_action_space(state['observation']['objectives'][0])
    subgoal_str = logic_statement_to_seq_string(subgoal['observation']['objectives'][0])
    env = TheoremProverEnv()
    for action in all_actions:
        state = deepcopy(state)
        env.load_problem_step(state)
        new_obs, _, _, _ = env.step(action)
        new_state = new_obs['objectives'][0]
        if logic_statement_to_seq_string(new_state) == subgoal_str:
            print('Reached')
            print(logic_statement_to_seq_string(new_state))
            print(action)
            return True



def prepare_action_space(logic_statement, subgoal_statement_str, assumptions, debugg=False):
    entities = logic_statement.ent
    untouchables = find_untouchables(entities, subgoal_statement_str)
    # print(f'untouchables = {[entity_to_seq_string(ent) for ent in untouchables]}')

    actions = set()
    for entity in entities:
        if entity not in untouchables:
            for axiom in select_one_operand_axioms(entity):
                actions.add((axiom, entity))
    for ent_pair in equ_move_term_selector(entities, untouchables):
        actions.add(('EquMoveTerm', *ent_pair))
    for ent_triple in principle_of_equality_selector(entities, logic_statement, assumptions, debugg):
        actions.add(('PrincipleOfEquality', *ent_triple))

    return actions


def find_untouchables(entities, subgoal_str):
    untouchables_raw = []
    all_untouchables_set = set()

    current_occurrances = {}
    for entity in entities:
        entity_str = entity_to_seq_string(entity)
        if entity_str not in current_occurrances:
            current_occurrances[entity_str] = [1, 0, entity]
            current_occurrances[entity_str][1] = subgoal_str.count(entity_str)
        else:
            current_occurrances[entity_str][0] += 1

    for entity_str in current_occurrances:
        if current_occurrances[entity_str][0] == 1 and current_occurrances[entity_str][1]==1:
                untouchables_raw.append(current_occurrances[entity_str][2])

    for untouchable in untouchables_raw:
        all_untouchables_set = all_untouchables_set.union(all_children_of_entity(untouchable))

    return all_untouchables_set


def find_untouchables_old(entities, subgoal):
    untouchables_raw = []
    all_untouchables_set = set()
    def find_occurances(list_entities):
        current_occurrences = {}
        for entity in list_entities:
            entity_str = entity_to_seq_string(entity)
            if entity_str not in current_occurrences:
                current_occurrences[entity_str] = [1, entity]
            else:
                current_occurrences[entity_str][0] += 1
        return current_occurrences
    curr_occurrances = find_occurances(entities)
    subgoal_occurrances = find_occurances(subgoal.ent)
    for entity_str in curr_occurrances:
        if curr_occurrances[entity_str][0] == 1:
            if entity_str in subgoal_occurrances:
                if subgoal_occurrances[entity_str][0] == 1:
                    untouchables_raw.append(curr_occurrances[entity_str][1])

    for untouchable in untouchables_raw:
        all_untouchables_set = all_untouchables_set.union(all_children_of_entity(untouchable))

    return all_untouchables_set

def all_children(entities):
    all_children = {}
    for ent in entities:
        all_children[ent] = all_children_of_entity(ent)
    return all_children


def all_children_of_entity(entity):
    all_children = {entity}
    queue = [entity]
    while len(queue) > 0:
        current_ent = queue.pop()
        all_children.add(current_ent)
        operands = current_ent.operands
        if operands is not None:
            queue = queue + operands
    return all_children


def select_one_operand_axioms(entity):
    axioms = []
    if is_structured(entity, "add"):
        axioms.append('AdditionCommutativity')
    if is_structured(entity, "add") and is_structured(entity.operands[0], "add") or is_structured(entity, "add") and is_structured(entity.operands[1], "add"):
        axioms.append('AdditionAssociativity')
    if is_structured(entity, "add") and entity.operands[0].name == "0" or is_structured(entity, "add") and entity.operands[1].name == "0":
        axioms.append('AdditionZero')
    if is_structured(entity, "add") and is_structured(entity.operands[1], "opp"):
        axioms.append('AdditionSimplification')
    if is_structured(entity, "mul"):
        axioms.append('MultiplicationCommutativity')
    if is_structured(entity, "mul") and is_structured(entity.operands[0], "mul") or is_structured(entity, "mul") and is_structured(entity.operands[1], "mul"):
        axioms.append('MultiplicationAssociativity')
    if is_structured(entity, "mul") and entity.operands[0].name == "1" or is_structured(entity, "mul") and entity.operands[1].name == "1":
        axioms.append('MultiplicationOne')
    if is_structured(entity, "mul") and is_structured(entity.operands[1], "inv"):
        axioms.append('MultiplicationSimplification')
    if (is_structured(entity, "add") and is_structured(entity.operands[0], "mul") and
            is_structured(entity.operands[1], "mul") and entity.operands[0].operands[1].name == entity.operands[1].operands[1].name) or (
            is_structured(entity, "mul") and is_structured(entity.operands[0], "add")):
        axioms.append('AdditionMultiplicationLeftDistribution')
    if (is_structured(entity, "add") and is_structured(entity.operands[0], "mul") and is_structured(
            entity.operands[1], "mul")and entity.operands[0].operands[0].name == entity.operands[1].operands[0].name) or (
            is_structured(entity, "mul") and is_structured(entity.operands[1], "add")):
        axioms.append('AdditionMultiplicationRightDistribution')
    if is_structured(entity, "sqr") or (is_structured(entity, "mul") and entity.operands[0].name == entity.operands[1].name):
        axioms.append('SquareDefinition')
    return axioms

def equ_move_term_selector(entities, untouchables):
    possible_second = [ent for ent in entities if equ_move_term_selector_second_operand(ent)]
    # possible_first = [ent.operands[0] for ent in entities if equ_move_term_selector_first_pre_operand(ent)] + \
    #                  [ent.operands[1] for ent in entities if equ_move_term_selector_first_pre_operand(ent)]
    all = []
    for second in possible_second:
        if second not in untouchables:
            all_children_of_second = all_children_of_entity(second)
            all_to_try = [(ent, second) for ent in entities if ent not in all_children_of_second]
            all.extend(all_to_try)
    return all

def equ_move_term_selector_second_operand(entity):
    return  is_structured(entity, "add") and is_structured(entity.operands[1], "opp")

def equ_move_term_selector_first_pre_operand(entity):
    return  is_structured(entity, "add")

def principle_of_equality_selector(entities, logic_statement, assumptions, debugg=False):
    if debugg:
        print(f'poe debugg: statement = {logic_statement_to_seq_string(logic_statement)}')
        print(f'poe debugg ground = {[logic_statement_to_seq_string(x) for x in assumptions]}')

    all = []
    b_and_d_candidates = [entity for entity in entities if is_structured(entity, 'add')]
    if debugg:
        print(f' = b_and_d_candidates: {[entity_to_seq_string(x) for x in b_and_d_candidates]}')

    for b_and_d in b_and_d_candidates:

        d = b_and_d.operands[1]
        d_str = entity_to_seq_string(d)
        b_and_d_children = all_children_of_entity(b_and_d)

        if debugg:
            print(f'b_and_d = {entity_to_seq_string(b_and_d)}')

        for a_and_c_candidate in b_and_d_candidates:
            if a_and_c_candidate not in b_and_d_children:
                if debugg:
                    print(f'a_and_c = {entity_to_seq_string(a_and_c_candidate)}')
                    print(f'b_and_d_children = {[entity_to_seq_string(x) for x in b_and_d_children]}')
                for i in range(2):
                    a = a_and_c_candidate.operands[i]
                    c = a_and_c_candidate.operands[1-i]
                    c_str = entity_to_seq_string(c)
                    if c_str == d_str:
                        if debugg:
                            print(f'b_and_d = {entity_to_seq_string(b_and_d)}')
                            print(f'd_str = {d_str}')
                            print(f'c_str = {c_str}')
                        all.append((a, c, b_and_d))
                    else:
                        for assumption in assumptions:
                            ls = entity_to_seq_string(assumption.operands[0])
                            rs = entity_to_seq_string(assumption.operands[1])
                            if (c_str == ls and d_str == rs) or (c_str == rs and d_str == ls):
                                if (a, c, b_and_d) not in all:
                                    all.append((a, c, b_and_d))
                            if debugg:
                                print('')
                                print(f'b_and_d = {entity_to_seq_string(b_and_d)}')
                                print(f'd_str = {d_str}')
                                print(f'c_str = {c_str}')
                                print(f'ls = {ls} rs = {rs}')
                                print('')


    # print(f'poe len = {len(all)}')
    if len(all)==1:
        x1 = logic_statement_to_seq_string(logic_statement)
        x2 = [logic_statement_to_seq_string(x) for x in assumptions]
    return all






    lhs = logic_statement.operands[0]
    rhs = logic_statement.operands[1]

    if is_structured(lhs, "add") and is_structured(rhs, "add"):
        lhs1 = lhs.operands[0]
        lhs2 = lhs.operands[1]
        rhs1 = rhs.operands[0]
        rhs2 = rhs.operands[1]

        all.append((lhs1, lhs2, rhs))
        all.append((lhs2, lhs1, rhs))
        all.append((rhs1, rhs2, lhs))
        all.append((rhs2, rhs1, lhs))

    return all

# def equ_move_term_selector_old(entities, ground_truth, show):
#     possible_second = [ent for ent in entities if equ_move_term_selector_second_operand(ent)]
#     print(f'possible second = {[entity_to_seq_string(x) for x in possible_second]}')
#     ground_truth_str = [logic_statement_to_seq_string(ground) for ground in ground_truth]
#     print(f'ground = {ground_truth_str}')
#     selected = []
#     for second in possible_second:
#         selected_new = [(first, second) for first in entities if equ_move_term_selector_first_operand(first, second, ground_truth_str, show)]
#         selected += selected_new
#     return selected
#
#
#

#
# def equ_move_term_selector_first_operand(first_operand, second_operand, ground_truth, show=False):
#     # """
#     #           a + b = c, ls(a) => ls(c + (-b))
#     #           2 inputs: [a, c + (-b)]
#     #           """
#     a = first_operand
#     c = second_operand.operands[0]
#     b = second_operand.operands[1].operands[0]
#     a_and_b = necessary_numerical_functions["add"].execute_nf([a, b])
#     first_con = necessary_logic_functions["Equivalent"].execute_lf([a_and_b, c])
#     statement = logic_statement_to_seq_string(first_con)
#     if show:
#         print(f'a = {entity_to_seq_string(a)} statement = {statement}')
#     return statement in ground_truth or entity_to_seq_string(first_con.operands[0]) == entity_to_seq_string(first_con.operands[1])


#
# class PrincipleOfEquality(MetaAxiom):
#     def __init__(self):
#         input_no = 3
#         assumption_size, conclusion_size = 2, 1
#         assumption_types = ["Equivalent"]
#         super(PrincipleOfEquality, self).__init__(input_no=input_no,
#                                                   assumption_size=assumption_size,
#                                                   conclusion_size=conclusion_size,
#                                                   assumption_types=assumption_types)
#
#     def execute_th(self, operands, mode="generate"):
#         if mode == "generate":
#             """
#             If a=b, c=d, then a + c = b + d
#             :param operands: 4 inputs [a, b, c, d]
#             :return: dict(Assumptions, Conclusions)
#             """
#             a, b, c, d = operands
#             assumptions = [
#                 necessary_logic_functions["Equivalent"].execute_lf([a, b]),
#                 necessary_logic_functions["Equivalent"].execute_lf([c, d])
#             ]
#             a_and_c = necessary_numerical_functions["add"].execute_nf([a, c])
#             b_and_d = necessary_numerical_functions["add"].execute_nf([b, d])
#             conclusions = [necessary_logic_functions["Equivalent"].execute_lf([a_and_c, b_and_d])]
#
#         elif mode == "prove":
#             """
#             a=b, c=d，ls(a+c) => ls(b+d)
#             :param operands: 3 inputs [a, c, b+d]
#             :return: dict(Assumptions, Conclusions)
#             """
#             a, c, b_and_d = [deepcopy(op) for op in operands]
#             if is_entity(a) and is_entity(c) and is_entity(b_and_d) and \
#                     is_structured(b_and_d, "add"):
#                 b, d, = [deepcopy(op) for op in b_and_d.operands]
#                 a_and_c = necessary_numerical_functions["add"].execute_nf([a, c])
#                 first_con = necessary_logic_functions["Equivalent"].execute_lf([a, b])
#                 second_con = necessary_logic_functions["Equivalent"].execute_lf([c, d])
#                 third_con = substitution(b_and_d, a_and_c)
#                 assumptions = [first_con, second_con, third_con]
#                 conclusions = [b_and_d.root]
#             else:
#                 assumptions = []
#                 conclusions = []
#         else:
#             raise NotImplementedError
#
#         return {
#             "Assumptions": assumptions,
#             "Conclusions": conclusions
#         }
#
#     def extend_core_gt(self, core_gt, entities, transform_gt):
#         """
#         a = b (c=d) -> a + c = b + d
#         """
#         return {
#             "action": True,
#             "makeup": True,
#             "makeup_config": [{
#                 "requirement_type": "Equivalent",
#                 "a": random.choice(entities),
#                 "b": random.choice(entities),
#             }],
#             "operand_retrieval":
#                 lambda makeup_conclusions: core_gt.operands + makeup_conclusions[0].operands
#         }
#
#     @staticmethod
#     def original_coding():
#         lhs_coding = (0, 0)
#         rhs_coding = (1, 0)
#         return lhs_coding, rhs_coding
#
#     @staticmethod
#     def prove_operands(new_ls):
#         lhs, rhs, = new_ls.operands
#         a, c, = lhs.operands
#         return [a, c, rhs]
#
#
# class EquMoveTerm(MetaAxiom):
#     def __init__(self):
#         input_no = 2
#         assumption_size, conclusion_size = 2, 1
#         assumption_types = ["Equivalent"]
#         super(EquMoveTerm, self).__init__(input_no=input_no,
#                                           assumption_size=assumption_size,
#                                           conclusion_size=conclusion_size,
#                                           assumption_types=assumption_types)
#
#     def execute_th(self, operands, mode="generate"):
#         if mode == "generate":
#             """
#             If a + b = c, then a = c + (-b)
#             :param operands: [a, b, c]
#             :return: dict(Assumptions, Conclusions)
#             """
#             a, b, c, = [deepcopy(op) for op in operands]
#             a_and_b = necessary_numerical_functions["add"].execute_nf([a, b])
#             assumption = necessary_logic_functions["Equivalent"].execute_lf([a_and_b, c])
#             assumptions = [assumption]
#             neg_b = necessary_numerical_functions["opp"].execute_nf([b])
#             c_minus_b = necessary_numerical_functions["add"].execute_nf([c, neg_b])
#             conclusion = necessary_logic_functions["Equivalent"].execute_lf([a, c_minus_b])
#             conclusions = [conclusion]
#         elif mode == "prove":
#             """
#             a + b = c, ls(a) => ls(c + (-b))
#             2 inputs: [a, c + (-b)]
#             """
#             a, c_minus_b, = [deepcopy(op) for op in operands]
#             if is_entity(a) and is_entity(c_minus_b) and is_structured(c_minus_b, "add") \
#                     and is_structured(c_minus_b.operands[1], "opp"):
#                 c, minus_b, = c_minus_b.operands
#                 b, = minus_b.operands
#                 a_and_b = necessary_numerical_functions["add"].execute_nf([a, b])
#                 first_con = necessary_logic_functions["Equivalent"].execute_lf([a_and_b, c])
#                 second_con = substitution(c_minus_b, a)
#                 assumptions = [first_con, second_con]
#                 conclusions = [c_minus_b.root]
#             else:
#                 assumptions = []
#                 conclusions = []
#         else:
#             raise NotImplementedError
#
#         return {"Assumptions": assumptions,
#                 "Conclusions": conclusions}
#
#     @staticmethod
#     def transform_gt(core_gt, entities):
#         if is_ls_type(core_gt, "Equivalent") and is_structured(core_gt.operands[0], "add"):
#             return {
#                 "action": True,
#                 "makeup": False,
#                 "operands": core_gt.operands[0].operands + [core_gt.operands[1]],
#                 "transformed_side": "custom",
#                 "custom_function": lambda x, y: x,
#                 "original_coding": None,
#             }
#         else:
#             return {
#                 "action": False
#             }
#
#     def extend_core_gt(self, core_gt, entities, transform_gt):
#         """
#         x + y = b -> x = b + (-y)
#         """
#         return self.transform_gt(core_gt, entities)
#
#     @staticmethod
#     def original_coding():
#         return
#
#     @staticmethod
#     def prove_operands(new_ls):
#         lhs, rhs, = new_ls.operands
#         return [lhs, rhs]
#
#     @staticmethod
#     def transform_recover_first_name(substitution_operands):
#         return substitution_operands[0].name
#
#
# # field_axioms = {
# #     "AdditionCommutativity": AdditionCommutativity(),
# #     "AdditionAssociativity": AdditionAssociativity(),
# #     "AdditionZero": AdditionZero(),
# #     "AdditionSimplification": AdditionSimplification(),
# #     "MultiplicationCommutativity": MultiplicationCommutativity(),
# #     "MultiplicationAssociativity": MultiplicationAssociativity(),
# #     "MultiplicationOne": MultiplicationOne(),
# #     "MultiplicationSimplification": MultiplicationSimplification(),
# #     "AdditionMultiplicationLeftDistribution": AdditionMultiplicationLeftDistribution(),
# #     "AdditionMultiplicationRightDistribution": AdditionMultiplicationRightDistribution(),
# #     "SquareDefinition": SquareDefinition(),
# #     "PrincipleOfEquality": PrincipleOfEquality(),
# #     "EquMoveTerm": EquMoveTerm(),
# # }
