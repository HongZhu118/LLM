class ExternalAlgoResultType:
    ALL_SOLUTION = 0
    NON_DOMINANT_SOLUTION = 1
    BEST_SOLUTION = 2


def ext_opt_prob_cb(algo_result):
    from suanpan.app import app
    from suanpan.node import node
    from suanpan.g import g

    app.send(
        {"outputData2": algo_result},
        message=g.context_queue.get().message, args=node.outargs)
