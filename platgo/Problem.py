import os
import re
import random
import types
import math
import json
import copy
import numpy as np
import pandas as pd
from .Population import Population


def import_code(code, name):
    module = types.ModuleType(name)
    module.__dict__.update({
        "np": np,
        "pd": pd,
        "random": random,
        "math": math,
        "Population": Population,
    })
    exec(code, module.__dict__)
    return module


class FuncName:
    LOWER_BOUND = "lower bound"
    UPPER_BOUND = "upper bound"
    DATA_SET = "data set"
    VARIABLE = "variable"
    INIT_FUNCTION = "init function"
    FIX_FUNCTION = "fix function"
    OBJECTIVE_FUNCTION = "objective function"
    CONSTRAINT_FUNCTION = "constraint function"
    OBJECTIVE_GRADIENT_FUNCTION = "objective gradient function"
    CONSTRAINT_GRADIENT_FUNCTION = "constraint gradient function"


# noinspection PyPep8Naming,PyPep8Naming
class Problem:
    """
    type = {
        "n_obj": {"single", "multi", "many"},
        "encoding": {
            "real",
            "binary",
            "permutation",
            "label",
            "two_permutation",
            "vrp",
        },
        "special": {
            "constrained",
            "constrained/none",
            "large",
            "large/none",
            "expensive",
            "expensive/none",
            "preference",
            "preference/none",
            "multimodal",
            "multimodal/none",
            "sparse",
            "sparse/none",
            "gradient",
            "gradient/none",
        }
    }
    """

    def __init__(self, external_optimization_problem, debug=False):
        """
        optimizationProblem: dict
        * for known problem like ZDT1
        {
            "name": "ZDT1",
            "n_var": 30
        }
        * for custom defined problem
        {
            "name": "custom_problem",
            "mode": 0,
            "encoding": "real",
            "n_var": 30,
            "lower": "0",
            "upper": "1",
            "variables": [
                {"param": "g", "formula": "1 + 9 * mean(x[1:])"},
                {"param": "h", "formula": "1 - sqrt(x[0] / g)"}
            ],
            "dataSet": [],
            "initFcn": [],
            "decFcn": [],
            "objFcn": [
                "x[0]",
                "g * h"],
            "conFcn": [],
            "objGradFcn": [],
            "conGradFcn": []
        }
        """
        optimization_problem = copy.deepcopy(external_optimization_problem)

        self.pop_size = None
        # encoding of decision variable
        self.encoding = optimization_problem["encoding"]
        # number of variable
        self.n_var = optimization_problem["n_var"]
        # lower bound
        self.lb = optimization_problem.get("lower", [0]*self.n_var)
        # upper bound
        self.ub = optimization_problem.get("upper", [1]*self.n_var)
        # number of objectives
        self.n_obj = len(
            optimization_problem.get("objFcn", [])
        ) or optimization_problem.get("n_obj")
        # number of constraints
        self.n_constr = len(optimization_problem.get("conFcn", []))
        # middle variables
        self.variables = optimization_problem.get("variables", [])
        # middle variable names
        self.variable_names = [item["param"] for item in self.variables]
        # data set
        self.data = optimization_problem.get("dataSet", [])
        # initial functions
        self.init_funcs = optimization_problem.get("initFcn", [])
        # decision functions
        self.dec_funcs = optimization_problem.get("decFcn", [])
        # objective functions
        self.obj_funcs = optimization_problem.get("objFcn", [])
        # constraint functions
        self.con_funcs = optimization_problem.get("conFcn", [])
        # objective gradient functions
        self.obj_grad_funcs = optimization_problem.get("objGradFcn", [])
        # constraint gradient functions
        self.con_grad_funcs = optimization_problem.get("conGradFcn", [])
        # external optimization problem
        self.is_external = True \
            if "algoResultType" in optimization_problem else False

        if not debug:
            self._download_files()

        self._eval_bounds()
        self._eval_data_set()
        self._convert_funcs()
        self._override_methods()

    def _override_methods(self):
        if self.init_funcs:
            if callable(self.init_funcs[0]):
                setattr(Problem, "init_pop", self.init_funcs[0])

        if self.dec_funcs:
            if callable(self.dec_funcs[0]):
                setattr(Problem, "fix_decs", self.dec_funcs[0])

    def init_pop(self, N: int = None):
        """
        根据问题的要求初始化种群染色体
        :param n: 种群大小，即决策矩阵大小
        """
        if N is None:
            N = self.pop_size
        if self.encoding == "real":
            if N is not None:
                lb = np.tile(self.lb, (N, 1))  # 将lb矩阵向0维坐标方向重复n次
                ub = np.tile(self.ub, (N, 1))  # 将ub矩阵向0维坐标方向重复n次
            else:
                # 将lb矩阵向0维坐标方向重复n次
                lb = np.tile(self.lb, (self.pop_size, 1))
                # 将ub矩阵向0维坐标方向重复n次
                ub = np.tile(self.ub, (self.pop_size, 1))
            decs = np.random.uniform(lb, ub)  # todo 这里种群初始化的时候要考虑编码方式
        elif self.encoding == "binary":
            decs = np.random.randint(0, high=2, size=(N, self.n_var))
        elif self.encoding == "label":
            lb = np.ones((N, self.n_var)).astype(int)
            rnd = np.random.randint(1, high=self.n_var + 1, size=(N, 1))
            ub = np.tile(rnd, (1, self.n_var))
            decs = np.round(np.random.uniform(lb, ub)).astype(int)
        elif self.encoding == "two_permutation":
            tmp = np.argsort(
                np.random.random((N, int(self.n_var / 2))), axis=1
            )
            tmp1 = np.argsort(
                np.random.random((N, int(self.n_var / 2))), axis=1
            )
            decs = np.hstack((tmp, tmp1))
        elif self.encoding == "vrp":
            decs = np.zeros(shape=(N, self.n_var))
            for i in range(N):
                if self.n_var % 2 == 0:
                    nc = int(self.n_var / 2) - 1  # nc(客户数)
                else:
                    nc = int((self.n_var - 1) / 2)
                a = np.arange(1, nc + 1)
                b = np.zeros(self.n_var - nc - 2)  # 染色体内部的零
                c = np.hstack((a, b))
                random.shuffle(c)
                c = np.hstack((0, np.hstack((c, 0))))
                decs[i, :] = c

        else:
            decs = np.argsort(np.random.random((N, self.n_var)), axis=1)
        pop = Population(decs=decs)
        return pop

    def fix_decs(self, pop):
        if self.encoding == "real":
            pop.decs = np.fmax(np.fmin(pop.decs, self.ub), self.lb)

    def _verify_bounds(self):
        assert (
            len(self.lb) == self.n_var
        ), f"{FuncName.LOWER_BOUND} should be a 1*{self.n_var} vector"
        assert (
            len(self.ub) == self.n_var
        ), f"{FuncName.UPPER_BOUND} should be a 1*{self.n_var} vector"

    def _convert_func_expr(self, func_expr):
        func_expr = func_expr.lower()
        pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
        match_names = set(re.findall(pattern, func_expr))
        valid_func_expr = func_expr
        for match_name in match_names:
            if match_name not in {"x", "data"}:
                if match_name not in self.variable_names:
                    # 替换函数
                    valid_func_expr = valid_func_expr.replace(
                        match_name, "np." + match_name
                    )
        return valid_func_expr

    def _download_files(self):
        # download set data files
        for obj_or_filename_or_expr in self.data:
            if isinstance(obj_or_filename_or_expr, str):
                _, file_ext = os.path.splitext(obj_or_filename_or_expr)
                if file_ext in [".csv", ".txt", ".json"]:
                    filename = obj_or_filename_or_expr

        # download function files
        for funcs in [
            self.init_funcs,
            self.dec_funcs,
            self.obj_funcs,
            self.con_funcs,
            self.obj_grad_funcs,
            self.con_grad_funcs,
        ]:
            for filename_or_expr in funcs:
                _, file_ext = os.path.splitext(filename_or_expr)
                if file_ext in [".py"]:
                    filename = filename_or_expr

    def _eval_data_set(self):
        """
        Evaluate data set, could be a mathematical expression, a data set
        file (.csv/.txt/.json), or an Object:
        * if a mathematical expression, convert it into corresponding object
        (e.g. number, string, ndarray, etc);
        * if a data set file (.csv/.txt/.json), convert it into corresponding
        object (ndarray or dict);
        * if an Object, do nothing;
        """
        for i, expr_filename_or_obj in enumerate(self.data):
            if isinstance(expr_filename_or_obj, str):
                if os.path.isfile(expr_filename_or_obj):
                    try:
                        filename = expr_filename_or_obj
                        _, ext = os.path.splitext(filename)
                        if ext in (".csv", ".txt"):
                            data_set = pd.read_csv(filename, header=None)
                            data_set = data_set.to_numpy()
                            self.data[i] = data_set
                        elif ext == ".json":
                            with open(filename, "r") as f:
                                self.data[i] = json.load(f)
                    except Exception as err:
                        raise Exception(f"Failed to define the \
                            {FuncName.DATA_SET} {i+1}, since: {err}")
                else:  # a mathematical expression
                    data_set_expr = expr_filename_or_obj
                    data_set = self._eval_func_expr(
                        data_set_expr, f"{FuncName.DATA_SET} {i+1}")
                    self.data[i] = data_set

    def _eval_bounds(self):
        """
        Evaluate lower & upper bounds, could be a mathematical expression
        or an Object:
        * if a mathematical expression, convert it into ndarray;
        * if an Object, do nothing;
        """
        if isinstance(self.lb, str):
            try:
                self.lb = [float(self.lb) for _ in range(self.n_var)]
            except ValueError:
                self.lb = self._eval_func_expr(self.lb, FuncName.LOWER_BOUND)

        if isinstance(self.ub, str):
            try:
                self.ub = [float(self.ub) for _ in range(self.n_var)]
            except ValueError:
                self.ub = self._eval_func_expr(self.ub, FuncName.UPPER_BOUND)

    def _convert_funcs(self):
        """
        Convert function string, could be a mathematical expression or a
        Python file path:
        * if a mathematical expression, do nothing;
        * if a Python file path, convert it into a callable;
        """
        funcs_li = [
            self.init_funcs,
            self.dec_funcs,
            self.obj_funcs,
            self.con_funcs,
            self.obj_grad_funcs,
            self.con_grad_funcs]
        func_names = [
            FuncName.INIT_FUNCTION,
            FuncName.FIX_FUNCTION,
            FuncName.OBJECTIVE_FUNCTION,
            FuncName.CONSTRAINT_FUNCTION,
            FuncName.OBJECTIVE_GRADIENT_FUNCTION,
            FuncName.CONSTRAINT_GRADIENT_FUNCTION]
        for funcs, func_name in zip(funcs_li, func_names):
            for i, filename_or_expr in enumerate(funcs):
                if os.path.isfile(filename_or_expr):
                    filename = filename_or_expr
                    with open(filename, "r") as f:
                        try:
                            content = f.read()
                            module = import_code(content, "_")
                            main = getattr(module, "main")
                            funcs[i] = main
                        except Exception as err:
                            raise Exception(f"Failed to define the \
                                {func_name} {i+1}, since: {err}")

    def _eval_func_expr(self, func_expr_or_callable, func_name, x=None,
                        var_values={}):
        try:
            if callable(func_expr_or_callable):
                res = func_expr_or_callable(x, self.data)
            else:
                valid_func_expr = self._convert_func_expr(
                    func_expr_or_callable)
                res = eval(
                    valid_func_expr,
                    {"np": np, "data": self.data, "x": x, **var_values})
            return res
        except Exception as err:
            raise Exception(f"Failed to define the {func_name}, since: {err}")

    def validate(self):
        self._verify_bounds()
        try:
            pop = self.init_pop(N=2)
        except Exception as err:
            func_name = "init_pop"
            raise Exception(f"Failed to define the {func_name}, since: {err}")

        try:
            self.fix_decs(pop)
        except Exception as err:
            func_name = "fix_decs"
            raise Exception(f"Failed to define the {func_name}, since: {err}")

        self.compute(pop)

    def compute(self, pop) -> None:
        objv = np.zeros((pop.pop_size, self.n_obj))
        cv = np.zeros((pop.pop_size, self.n_constr))

        for i in range(pop.pop_size):
            x = pop.decs[i, :]
            var_values = {}
            for variable in self.variables:
                name = variable["param"]
                variable_expr = variable["formula"]
                var_values[name] = self._eval_func_expr(
                    variable_expr, f"{FuncName.VARIABLE} {name}",
                    x=x, var_values=var_values)

            for j in range(len(self.obj_funcs)):
                func_expr_or_callable = self.obj_funcs[j]
                objv[i, j] = self._eval_func_expr(
                    func_expr_or_callable,
                    f"{FuncName.OBJECTIVE_FUNCTION} f{j+1}",
                    x=x, var_values=var_values)

            for j in range(len(self.con_funcs)):
                func_expr_or_callable = self.con_funcs[j]
                cv[i, j] = self._eval_func_expr(
                    func_expr_or_callable,
                    f"{FuncName.CONSTRAINT_FUNCTION} g{j+1}",
                    x=x, var_values=var_values)

        pop.objv = objv
        pop.cv = cv

