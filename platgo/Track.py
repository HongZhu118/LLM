class Track:
    """
    用于追踪进化过程

    这个类还没想好，等用到再完善
    """

    def __init__(self):
        # 用于记录每一代种群目标值信息
        self.objv_his = []
        # 用于记录每一代种群对应的结果
        self.finalresult_his = []
        # 记录单目标每一代种群目标值的均值，用于单目标算法的绘图
        self.soea_median = []
        self.soea_best = []
        # 记录每一代的决策变量
        self.decs_his = []
        # 记录每一代的约束值
        self.cv_his = []
        # 记录HV
        self.hv_his = []
        # 记录可行解比例
        self.feasible_rate_his = []
        # 记录每一代的最优解
        self.pop_best_his = []
        # 记录每一代的非支配解
        self.non_dominant_his = []
