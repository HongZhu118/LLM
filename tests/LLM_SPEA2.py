import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__))+ "../..")

from common.commons import AlgoResultType
from platgo.algorithms import SPEA2

if __name__ == "__main__":
    # 内置问题
    optimization_problem = {
        "name": "DTLZ1",
        "n_var": 10,
        "algoResultType": 0,
        "lower": "0",
        "upper": "1",
    }
    print(optimization_problem)

    pop_size = 100
    max_fe = 10000
    options = {}

    evol_algo = SPEA2(
        pop_size=pop_size,
        options=options,
        optimization_problem=optimization_problem,
        # simulation_request_callback=None, # noqa
        control_cb=None,
        max_fe=max_fe,
        name="SPEA2-Thread",
        debug=True,
    )

    evol_algo.set_data_type(AlgoResultType.OBJECTIVE_SPACE)
    evol_algo.start()
    evol_algo.join()
    algo_result = evol_algo.get_external_algo_result()
    print("Done")
