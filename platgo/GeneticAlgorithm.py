import threading
import importlib
import time
import math
import asyncio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import logging

from .Track import Track
from .Population import Population
from .Problem import Problem
from .metrics import hv
from .utils import nd_sort
from common.commons import AlgoResultType, AlgoState, AlgoMode
from common.external_optimization_problem import ExternalAlgoResultType


class StopException(Exception):
    pass


class GeneticAlgorithm(threading.Thread):
    type = {
        "n_obj": {"single", "multi", "many"},
        "encoding": {
            "real",
            "binary",
            "permutation",
            "label",
            "vrp",
            "two_permutation",
        },  # noqa
        "special": {
            "constrained",
            "constrained/none",
            "large",
            "large/none",
            "expensive",
            "expensive/none",  # noqa
            "preference",
            "preference/none",
            "multimodal",
            "multimodal/none",
            "sparse",
            "sparse/none",
            "gradient",
            "gradient/none",
        },  # noqa
    }

    def __init__(
            self,
            pop_size,
            options,
            optimization_problem,
            control_cb,
            max_fe=10000,
            name=None,
            show_bar=False,
            sim_req_cb=None,
            ext_opt_prob_cb=None,
            debug=False,
    ):
        threading.Thread.__init__(self, name=name)
        self._options = options

        # 关联问题
        self._algo_mode = AlgoMode(optimization_problem.get("mode", 0))
        self._control_cb = control_cb

        # 进化部分
        self._gen = 0
        self._max_fe = max_fe
        self._track = Track()

        # 显示
        self._is_show_bar = show_bar
        self._p_bar = tqdm(total=100)  # 进度条
        self._p_bar_inc = 0
        self._p_bar_inc_res = 0.0
        self._p_bar_value = 0
        self._p_bar_values_his = []
        self._data_type = AlgoResultType.OBJECTIVE_SPACE
        self._frame = 0
        self._iter_start_time = None
        self._exec_time = 0.0

        self._first_cal_obj = True
        self._refer_array = None
        self._cur_hv = 0
        self._max_hv = 10

        # 进度控制
        self._state_lock = threading.Lock()
        self._state = AlgoState.WAITING
        self._start_flag = False

        # 本地调试
        if not debug:
            from common.exception import exception_decorator

            self.run = exception_decorator(self.run)

        self._debug = debug

        problem = self.init_problem(
            optimization_problem, sim_req_cb=sim_req_cb, debug=debug
        )
        problem.pop_size = pop_size
        self.problem = problem

        # 外部系统数据回调函数
        self._ext_opt_prob_cb = ext_opt_prob_cb
        self._algo_result_type = optimization_problem.get("algoResultType")

    @property
    def state(self):
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new_state):
        with self._state_lock:
            self._state = new_state

    def reset_iter_start_time(self):
        self._iter_start_time = time.time()

    def _reset_exec_time(self):
        self._exec_time = 0.0

    def update_exec_time(self):
        iter_exec_time = time.time() - self._iter_start_time
        self._exec_time += iter_exec_time

    def set_data_type(self, data_type):
        self._data_type = data_type

    def set_frame_from_bar_value(self, slidebar_value):
        frame_interp = interp1d(
            self._p_bar_values_his,
            range(len(self._p_bar_values_his)),
            kind="nearest-up",
            bounds_error=False,
            fill_value="extrapolate",
        )
        self._frame = int(frame_interp(slidebar_value))

    def reset_frame(self):
        self._frame = len(self._track.objv_his) - 1

    def init_problem(self, optimization_problem, sim_req_cb=None, debug=False):
        if self._algo_mode is AlgoMode.ACADEMIC:
            class_name = optimization_problem["name"]
            try:
                problems = importlib.import_module(
                    "platgo.problems"
                )
                problem_class = getattr(problems, class_name)
                problem = problem_class(optimization_problem)
            except AttributeError:
                problem = Problem(optimization_problem, debug=debug)
        else:  # algo_mode is AlgoMode.ENGINEERING:
            problem = EngineeringProblem(
                optimization_problem, sim_req_cb=sim_req_cb, debug=debug
            )

        return problem

    def run_algorithm(self):
        """继承的子类需实例化此方法，否则会报错。
        参考scipy_algo.py

        Raises:
            NotImplementedError: [子类未实例化此方法]
        """
        raise NotImplementedError(
            "Subclasses should implement run_algorithm method!"
        )  # noqa: E501

    def run(self):
        """线程run函数，子类请勿集成"""
        # 执行搜索
        if self._debug:
            self.run_algorithm()
            if self._algo_mode is AlgoMode.ENGINEERING:
                self.problem.sim_req_cb({})
        else:
            while True:
                while not self._start_flag:
                    time.sleep(1e-3)
                try:
                    self.reset_iter_start_time()
                    self._reset_exec_time()
                    self.run_algorithm()
                    if (
                            self._algo_mode is AlgoMode.ACADEMIC
                            and self.problem.is_external
                    ):
                        ext_algo_res = self.get_external_algo_result()
                        self._ext_opt_prob_cb(ext_algo_res)
                    if self._algo_mode is AlgoMode.ENGINEERING:
                        self.problem.sim_req_cb({})
                except StopException as e:
                    logging.info(e)

    def re_init(self):
        self._gen = 0
        self._track = Track()
        self._p_bar = tqdm(total=100)  # 进度条
        self._p_bar_inc = 0
        self._p_bar_inc_res = 0.0
        self._p_bar_value = 0
        self._p_bar_values_his = []
        self._frame = 0
        self._first_cal_obj = True
        self._refer_array = None
        self._cur_hv = 0
        self._max_hv = 10

    def is_start(self):
        with self._state_lock:
            return self._start_flag

    def set_start_flag(self, flag):
        with self._state_lock:
            self._start_flag = flag

    def not_terminal(self, pop: Population) -> bool:
        """检查算法是否到结束条件"""
        if not self._debug:
            logging.info(f"Current function evaluations: {self._gen}")
        else:
            print(f"Current function evaluations: {self._gen}")

        self._track.objv_his.append(pop.objv)
        self._track.finalresult_his.append(pop.finalresult)
        self._track.decs_his.append(pop.decs)
        self._track.cv_his.append(pop.cv)
        feasible_rate = sum(1 for i in pop.cv if np.all(i) <= 0) / pop.pop_size
        self._track.feasible_rate_his.append([self._gen, feasible_rate])
        pop_best, non_dominant_pops = self.pop_best(pop)
        self._track.pop_best_his.append(pop_best)
        self._track.non_dominant_his.append(non_dominant_pops)

        self._p_bar_value += self._p_bar_inc
        self._p_bar_values_his.append(self._p_bar_value)
        if self.problem.n_obj == 1:
            self._track.soea_median.append(np.median(self._track.objv_his[-1]))
            self._track.soea_best.append(np.min(self._track.objv_his[-1]))

        if self._gen >= self._max_fe:
            if self._debug:
                self.reset_frame()
                self._p_bar.update(self._p_bar_inc)
                self._p_bar.close()

                self._draw()
                plt.ioff()
                plt.show()
                plt.pause(2)
                plt.close()
            else:
                self._control_cb(self)
                self._complete_algo()
            return False
        else:
            if self._debug:
                self.reset_frame()
                self._p_bar.update(self._p_bar_inc)
                self._draw()
            else:
                self._control_cb(self)
            self._p_bar_inc = 0
            return True

    def pop_best(self, pop):
        frontno, _ = nd_sort(pop.objv, pop.cv, self.problem.pop_size)
        Current = np.argwhere(frontno == 1).flatten()
        PopObj = pop.objv
        N = PopObj.shape[0]
        M = PopObj.shape[1]
        if len(Current) <= M:
            Rank = np.argsort(-PopObj[Current, :], axis=0)
            return pop[Current, :][int(Rank[0][0])], pop[frontno == 1]
        else:
            Distance = np.zeros(N)
            # Find the extreme points
            Rank = np.argsort(-PopObj[Current, :], axis=0)
            Extreme = np.zeros(M, dtype=int)
            Extreme[0] = Rank[0, 0]
            for j in range(1, len(Extreme)):
                k = 0
                Extreme[j] = Rank[k, j]
                while Extreme[j] in Extreme[0:j]:
                    k = k + 1
                    Extreme[j] = Rank[k, j]
            # Calculate the hyperplane
            temp = PopObj[Current[Extreme], :]
            try:
                Hyperplane = np.linalg.solve(
                    temp, np.ones((len(Extreme), 1))
                )  # noqa
            except:  # noqa
                print("警告: 矩阵接近奇异值，或者缩放错误。结果可能不准确")
                Hyperplane = np.dot(
                    np.linalg.pinv(temp), np.ones((len(Extreme), 1))
                )  # noqa
            # Calculate the distance of each solution to the hyperplane
            temp1 = (np.dot(PopObj[Current, :], Hyperplane) - 1).flatten()
            temp2 = np.sqrt(np.sum(Hyperplane ** 2))
            Distance[Current] = -temp1 / temp2
            Rank = np.argsort(-Distance[Current])
            pop_best = pop[Current][int(Rank[0])]
            return pop_best, pop[frontno == 1]

    def _draw(self):
        plt.clf()
        if self.problem.n_obj == 1:
            plt.plot(self._track.soea_median, label="median objective")
            plt.plot(self._track.soea_best, label="best objective")
            plt.xlabel("number of generation")
            plt.ylabel("value of objective")
            plt.xlim(left=0)
            plt.legend()

        elif self.problem.n_obj == 2:
            # 取最后一代种群目标值
            data = self._track.objv_his[-1]
            objv1 = data[:, 0]
            objv2 = data[:, 1]
            plt.xlim(min(objv1), max(objv1))
            plt.ylim(min(objv2), max(objv2))
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.scatter(objv1, objv2)

        elif self.problem.n_obj == 3:
            ax = plt.subplot(projection="3d")
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_zlabel("f3")
            data = self._track.objv_his[-1]
            ax.scatter3D(
                data[:, 0],
                data[:, 1],
                data[:, 2],
                marker="o",
                depthshade=False,
            )
            ax.view_init(35, 50)

        else:
            data = self._track.objv_his[-1]
            obj_dim = np.arange(1, data.shape[1] + 1)
            plt.xlabel("dimension on")
            plt.ylabel("f")
            plt.plot(obj_dim, data.T, c="b")
            plt.xlim(1, data.shape[1])

        # plt.pause(0.01)

    def cal_obj(self, pop):
        # if self._gen>=1000:  # 跑SatellkiteRCPSP2_2_Strategy时需要设置1000，其余的不需要
        #     self.problem.fix_decs(pop)
        self.problem.fix_decs(pop)
        if self._algo_mode is AlgoMode.ACADEMIC:
            self.problem.compute(pop)
        else:
            asyncio.run(self.problem.compute(pop))
        self._gen += len(pop)

        if self._first_cal_obj:
            self._first_cal_obj = False
            self._refer_array = np.max(pop.objv, axis=0)

        self.cal_hv(pop)
        self._p_bar_inc_res += 100 * len(pop) / self._max_fe - math.floor(
            100 * len(pop) / self._max_fe
        )
        if self._p_bar_inc_res >= 0.999999:
            self._p_bar_inc += math.floor(100 * len(pop) / self._max_fe) + 1
            self._p_bar_inc_res -= 1
        else:
            self._p_bar_inc += math.floor(100 * len(pop) / self._max_fe)

    def cal_hv(self, pop):
        if_eval_hv = self._gen >= (self._max_fe * self._cur_hv / self._max_hv)

        if if_eval_hv:
            score = hv(pop, self._refer_array)
            self._track.hv_his.append([self._gen, score])
            self._cur_hv += 1

    def pause_algo(self):
        while True:
            if self.state is AlgoState.PAUSING:
                self.state = AlgoState.PAUSED
                logging.info("Algorithm is paused.")
            elif self.state is AlgoState.RUNNING:
                break
            elif self.state is AlgoState.STOPPING:
                self.stop_algo()
            time.sleep(1e-6)

    def stop_algo(self):
        self.state = AlgoState.STOPPED
        raise StopException("Algorithm is manually stopped!")

    def _complete_algo(self):
        self.state = AlgoState.COMPLETED
        self.set_start_flag(False)
        self._control_cb(self)
        logging.info("Algorithm is completed.")

    def post_processing(self):
        frame = self._frame

        data = None
        if self._data_type is AlgoResultType.OBJECTIVE_SPACE:
            if self.problem.n_obj == 1:
                history_data = [
                    [d1, d2]
                    for d1, d2 in zip(
                        self._track.soea_median, self._track.soea_best
                    )
                ]
                data = history_data[: frame + 1]
            else:
                history_data = self._track.objv_his
                data = history_data[frame]
                data = data.tolist()
        elif self._data_type is AlgoResultType.DECISION_SPACE:
            history_data = self._track.decs_his
            data = history_data[frame]
            data = data.tolist()
        elif self._data_type is AlgoResultType.FINAL_RESULT:
            history_data = self._track.finalresult_his
            data = history_data[frame]
            data = data.tolist()
        elif self._data_type is AlgoResultType.HV:
            history_data = self._track.hv_his
            data = history_data[: frame + 1]
        elif self._data_type is AlgoResultType.FEASIBLE_RATE:
            history_data = self._track.feasible_rate_his
            data = history_data[: frame + 1]

        res = {
            "n_obj": self.problem.n_obj,
            "data": data,
            "slidebarIncrement": self._p_bar_inc,
            "dataType": self._data_type.value,
            "execTime": self._exec_time,
            "curFE": self._gen,
        }

        return res

    def get_external_algo_result(self):
        frame = self._frame
        res = []

        headers = []
        headers.extend([f"Variable {i + 1}" for i in range(self.problem.n_var)])

        headers.extend([f"Objective {i + 1}" for i in range(self.problem.n_obj)])
        headers.extend(
            [f"Constraint {i + 1}" for i in range(self.problem.n_constr)]
        )  # noqa
        headers.extend([f"FinalResult"])
        res.append(headers)

        if self._algo_result_type is ExternalAlgoResultType.ALL_SOLUTION:
            data = np.concatenate(
                (
                    self._track.decs_his[frame],
                    self._track.objv_his[frame],
                    self._track.cv_his[frame],
                    self._track.finalresult_his[frame],
                ),
                axis=1,
            )
            data = data.tolist()
        elif (
                self._algo_result_type
                is ExternalAlgoResultType.NON_DOMINANT_SOLUTION
        ):  # noqa
            non_dominant_pops = self._track.non_dominant_his[frame]

            data = []
            for non_dominant_pop in non_dominant_pops:
                temp = []
                temp.extend(non_dominant_pop.decs.flatten().tolist())
                temp.extend(non_dominant_pop.objv.flatten().tolist())
                temp.extend(non_dominant_pop.cv.flatten().tolist())
                temp.extend(non_dominant_pop.finalresult.flatten().tolist())
                data.append(temp)
        elif self._algo_result_type is ExternalAlgoResultType.BEST_SOLUTION:
            pop_best = self._track.pop_best_his[frame]
            data = []
            data.extend(pop_best.decs.flatten().tolist())
            data.extend(pop_best.objv.flatten().tolist())
            data.extend(pop_best.cv.flatten().tolist())
            data.extend(pop_best.finalresult.flatten().tolist())
        else:
            raise RuntimeError(
                f"Unknown external algorithm result type: {self._algo_result_type}"
            )  # noqa

        res.extend(data)
        res2 = self.post_processing()  # 为了获取HV的值增加的
        output = {
            "data": res,
            "hv1": res2,  # 为了获取HV的值增加的
            "n_var": self.problem.n_var,
            "n_obj": self.problem.n_obj,
            "n_constr": self.problem.n_constr,
        }
        return output

    def get_current_frame_data(self):
        """
        返回当前帧种群的数据，顺序分别为：决策变量、目标值和约束值。
        """
        frame = self._frame

        data = np.concatenate(
            (
                self._track.decs_his[frame],
                self._track.objv_his[frame],
                self._track.cv_his[frame],
                self._track.finalresult_his[frame],
            ),
            axis=1,
        )
        data = data.tolist()

        current_frame_data = {
            "data": data,
            "n_var": self.problem.n_var,
            "n_obj": self.problem.n_obj,
            "n_constr": self.problem.n_constr,
        }

        return current_frame_data

    def get_gif_data(self):
        """
        返回所有帧种群的数据，仅当前数据类型
        """
        if self._data_type is AlgoResultType.OBJECTIVE_SPACE:
            if self.problem.n_obj == 1:
                history_data = [
                    [d1, d2]
                    for d1, d2 in zip(
                        self._track.soea_median, self._track.soea_best
                    )
                ]
                data = history_data
            else:
                history_data = self._track.objv_his
                data = [frame_data.tolist() for frame_data in history_data]
        elif self._data_type is AlgoResultType.DECISION_SPACE:
            history_data = self._track.decs_his
            data = [frame_data.tolist() for frame_data in history_data]
        elif self._data_type is AlgoResultType.FINAL_RESULT:
            history_data = self._track.finalresult_his
            data = [frame_data.tolist() for frame_data in history_data]
        elif self._data_type is AlgoResultType.HV:
            history_data = self._track.hv_his
            data = history_data
        elif self._data_type is AlgoResultType.FEASIBLE_RATE:
            history_data = self._track.feasible_rate_his
            data = history_data
        else:
            data = None

        gif_data = {
            "data": data,
            "n_var": self.problem.n_var,
            "n_obj": self.problem.n_obj,
            "n_constr": self.problem.n_constr,
        }

        return gif_data

    def get_all_frames_data(self):
        """
        返回所有帧种群的数据
        """
        data = []
        for decs,  objv, cv, finalresult in zip(
                self._track.decs_his, self._track.objv_his, self._track.cv_his, self._track.finalresult_his
        ):
            temp = np.concatenate((decs, objv, cv, finalresult), axis=1)
            temp = temp.tolist()
            data.append(temp)

        all_frames_data = {
            "data": data,
            "n_var": self.problem.n_var,
            "n_obj": self.problem.n_obj,
            "n_constr": self.problem.n_constr,
        }
        return all_frames_data

    def get_current_frame_best_population_data(self):
        """
        返回当前帧的最优解数据，顺序分别为：决策变量、目标值和约束值。
        """
        frame = self._frame
        pop_best = self._track.pop_best_his[frame]

        data = []
        data.extend(pop_best.decs.flatten().tolist())
        data.extend(pop_best.objv.flatten().tolist())
        data.extend(pop_best.cv.flatten().tolist())
        data.extend(pop_best.finalresult.flatten().tolist())

        current_frame_best_population_data = {
            "data": data,
            "n_var": self.problem.n_var,
            "n_obj": self.problem.n_obj,
            "n_constr": self.problem.n_constr,
        }
        return current_frame_best_population_data

    def get_current_frame_non_dominant_data(self):
        """
        返回当前帧的非支配数据，顺序分别为：决策变量、目标值和约束值。
        """
        frame = self._frame
        non_dominant_pops = self._track.non_dominant_his[frame]

        data = []
        for non_dominant_pop in non_dominant_pops:
            temp = []
            temp.extend(non_dominant_pop.decs.flatten().tolist())
            temp.extend(non_dominant_pop.objv.flatten().tolist())
            temp.extend(non_dominant_pop.cv.flatten().tolist())
            temp.extend(non_dominant_pop.finalresult.flatten().tolist())
            data.append(temp)

        current_frame_non_dominant_data = {
            "data": data,
            "n_var": self.problem.n_var,
            "n_obj": self.problem.n_obj,
            "n_constr": self.problem.n_constr,
        }

        return current_frame_non_dominant_data
