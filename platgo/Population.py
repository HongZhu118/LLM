import numpy as np
from typing import Union


class Population:
    """
    种群类
    """

    def __init__(
            self,
            pop_size: int = None,
            decs: np.ndarray = None,
            cv: np.ndarray = None,
            objv: np.ndarray = None,
            finalresult: np.ndarray = None,
            cgradv: np.ndarray = None,
            objgradv: np.ndarray = None,
            vel: np.ndarray = None,
    ) -> None:
        """
        种群中的属性包含了优化的所有解，对于多目标优化来说通常不存在唯一解，只有相对合适的解集。
        种群初始化
        可以传入决策变量矩阵decs初始化或是根据pop_size和problem自动初始化
        :param pop_size: 种群大小
        :param decs:     决策矩阵
        :param cv:       约束违反矩阵
        :param objv:     目标值矩阵
        :param cgradv:   约束梯度违反矩阵
        :param objgradv: 目标梯度值矩阵
        :param vel:      速度
        """
        # 初始化检查,条件为False时触发异常
        assert (
                decs is None or decs.ndim == 2
        ), "Population initiate error, decs must be 2-D array"
        assert (
                cv is None or cv.ndim == 2
        ), "Population initiate error, cv must be 2-D array"
        assert (
                objv is None or objv.ndim == 2
        ), "Population initiate error, objv must be 2-D array"
        assert (
                finalresult is None or finalresult.ndim == 2
        ), "Population initiate error, finalresult must be 2-D array"
        assert (
                cgradv is None or cgradv.ndim == 3
        ), "Population initiate error, cgradv must be 3-D array"
        assert (
                objgradv is None or objgradv.ndim == 3
        ), "Population initiate error, objgradv must be 3-D array"
        assert (
                pop_size is not None or decs is not None
        ), "Population initiate error, at least one of pop_size and decs is not None"  # noqa: E501 N和decs至少有一个不为None， 否则不能确定种群大小
        self.decs: np.ndarray = decs.copy()
        self.cv: np.ndarray = cv.copy() if cv is not None else None
        self.objv: np.ndarray = objv.copy() if objv is not None else None
        self.finalresult: np.ndarray = finalresult.copy() if finalresult is not None else None
        self.cgradv: np.ndarray = cgradv.copy() if cgradv is not None else None
        self.objgradv: np.ndarray = objgradv.copy() if objgradv is not None else None  # noqa
        # TODO 类中其他方法均没有增加关于vel的代码，需要用到的时候再进行添加
        self.vel: np.ndarray = vel.copy() if vel is not None else None

    @property
    def pop_size(self) -> int:
        return self.decs.shape[0]

    def add(self, Add):
        if self.vel is None:
            self.vel = Add

    def copy(self):
        """返回对象副本"""
        new_decs = self.decs.copy()
        new_cv = self.cv.copy() if self.cv is not None else None
        new_objv = self.objv.copy() if self.objv is not None else None
        new_finalresult = self.finalresult.copy() if self.finalresult is not None else None
        new_cgradv = self.cgradv.copy() if self.cgradv is not None else None
        new_objgradv = self.objgradv.copy() if self.objgradv is not None else None  # noqa
        new_vel = self.vel.copy() if self.vel is not None else None  # 添加额外参数
        pop = Population(
            decs=new_decs, cv=new_cv, objv=new_objv, finalresult=new_finalresult, cgradv=new_cgradv,
            objgradv=new_objgradv, vel=new_vel
        )  # 添加额外参数
        return pop

    def __getitem__(self, ind: Union[int, list, np.ndarray, slice]):
        """
        种群切片，根据下标选择部分个体生成新的种群
        ndarray索引分为int下标索引和bool索引，计算N的方式不同
        :param ind: 新种群的索引，接受int, list, ndarray, slice
        """
        if self.decs is None:
            raise RuntimeError("The population has not been initialized")
        if type(ind) == int:
            ind = [ind]
        if type(ind) == np.ndarray:
            # 索引的类型只能是int32或bool或int64
            assert ind.dtype in [np.int32, np.int64, np.bool8]
            # 索引的维度只能是 (n,) 或是 (1,n)
            assert ind.ndim == 1 or ind.ndim == 2
            if ind.ndim == 2:
                assert 1 in ind.shape
                ind = ind.flatten()

        new_decs = self.decs[ind]
        new_cv = self.cv[ind] if self.cv is not None else None
        new_objv = self.objv[ind] if self.objv is not None else None
        new_finalresult = self.finalresult[ind] if self.objv is not None else None
        new_cgradv = self.cgradv[ind] if self.cgradv is not None else None
        new_objgradv = self.objgradv[ind] if self.objgradv is not None else None  # noqa
        new_vel = self.vel[ind] if self.vel is not None else None
        new_pop = Population(
            decs=new_decs, cv=new_cv, objv=new_objv, finalresult=new_finalresult, cgradv=new_cgradv,
            objgradv=new_objgradv, vel=new_vel
        )  # noqa: E501
        return new_pop

    def __setitem__(
            self, item: Union[int, list, np.ndarray, slice], pop
    ) -> None:  # noqa: E501
        """
        为种群内的部分个体赋值，支持多对一
        population[[0,1]] = pop
        :param item: 下标
        :param pop: instance of Population
        :return:
        """
        # TODO 两个种群需要进行检查，要么n->1，要么n->n，不允许n->m
        if self.decs is not None:
            self.decs[item] = pop.decs
        if self.cv is not None:
            self.cv[item] = pop.cv
        if self.objv is not None:
            self.objv[item] = pop.objv
        if self.finalresult is not None:
            self.finalresult[item] = pop.finalresult
        if self.cgradv is not None:
            self.cgradv[item] = pop.cgradv
        if self.objgradv is not None:
            self.objgradv[item] = pop.objgradv

    def __add__(self, pop):
        """
        合并种群,不更改原来的两种群，而是返回新的种群
        不会重新计算目标函数值和约束
        :param pop:
        :return:
        """
        # TODO cv和objv可以是空，但是两种群必须一致
        new_decs = np.vstack([self.decs, pop.decs])
        new_cv = np.vstack([self.cv, pop.cv]) if self.cv is not None else None
        new_objv = np.vstack([self.objv, pop.objv]) if self.objv is not None else None  # noqa: E501
        new_finalresult = np.vstack(
            [self.finalresult, pop.finalresult]) if self.finalresult is not None else None  # noqa: E501
        new_cgradv = np.vstack([self.cgradv, pop.cgradv]) if self.cgradv is not None else None  # noqa
        new_objgradv = np.vstack([self.objgradv, pop.objgradv]) if self.objgradv is not None else None  # noqa: E501
        new_pop = Population(
            decs=new_decs, cv=new_cv, objv=new_objv, finalresult=new_finalresult, cgradv=new_cgradv,
            objgradv=new_objgradv)
        return new_pop

    def __len__(self):
        return self.decs.shape[0]
