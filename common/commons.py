import enum


class AlgoMode(enum.Enum):
    ACADEMIC = 0
    ENGINEERING = 1


class SimParams:
    DesignId = "design_id"
    InputPrefix = "x"
    ObjectivePrefix = "f"
    ConstraintPrefix = "g"
    OutputObjective = "F"
    OutputConstraint = "G"
    OutputGradient = "dF"
    OutputConstraintGradient = "dG"


class ConvertionProbSim:
    """Convertion between problem and simulation."""
    @staticmethod
    def probres2simres(prob_res):
        """
        Convert an optimization problem result to a simulation result.
        e.g.
            * An optimization problem result: {"F": [obj1, obj2], "G": [cv1]},
            * A simulation result:
                {"f1": objv1, "f2": objv2, "g1": cv1}
        """
        sim_res = {}

        objv = prob_res[SimParams.OutputObjective]
        for i in range(len(objv)):
            sim_res[SimParams.ObjectivePrefix+str(i+1)] = objv[i]

        cv = prob_res[SimParams.OutputConstraint]
        for i in range(len(cv)):
            sim_res[SimParams.ConstraintPrefix+str(i+1)] = cv[i]

        return sim_res

    @staticmethod
    def simparams2x(sim_params):
        """
        Convert a simulation parameters to x.
        e.g.
            * Simulation parameters:
            {
                "x1": 0.1,
                "x2": 0.1,
                ...
                "x30": 0.1
            }
            * x:
            [0.1, 0.1, ..., 0.1]
        """
        x = [sim_params[SimParams.InputPrefix+str(i+1)]
             for i in range(len(sim_params))]
        return x


class AlgoState(enum.Enum):
    WAITING = 0
    RUNNING = 1
    PAUSING = 2
    PAUSED = 3
    STOPPING = 4
    STOPPED = 5
    COMPLETED = 6


class AlgoEvent(enum.Enum):
    START_ALGO = 0
    CONTINUE_ALGO = 1
    PAUSE_ALGO = 2
    STOP_ALGO = 3
    COMPLETE_ALGO = 4


class AlgoResultType(enum.Enum):
    OBJECTIVE_SPACE = 0
    DECISION_SPACE = 1
    HV = 2
    FEASIBLE_RATE = 3
    FINAL_RESULT = 4


class OptOrDoE:
    """优化/DoE模式常量"""

    OPT = 1
    DOE = 2

    DOE_Hypercube = 0
    DOE_FullFactorial = 1
    DOE_BoxBehnken = 2


class runState:
    Waiting = 0
    Running = 1
    Pause = 2
    Finished = 3


class DoEType:
    LatinHypercube = 0
    FullFactorial = 1
    BoxBehnken = 2
    OrthogonalDesign = 3
