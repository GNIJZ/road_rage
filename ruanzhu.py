# import numpy as np
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl
# image_data=1;
# physio_data=2;
# eeg_data=3;
# def model(a,b,c):
#     return  np.random.rand()
# def fuzzy_inference(trust_value):
#     """
#     进行模糊推理，计算控制权重比例
#     参数:
#         trust_value: 驾驶员的信任值
#     返回值:
#         driver_weight: 驾驶员控制权重
#         machine_weight: 机器控制权重
#     """
#     # 定义模糊变量
#     driver_trust = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'driver_trust')
#     machine_control = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'machine_control')
#     weight_ratio = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'weight_ratio')
#
#     # 定义模糊集
#     driver_trust.automf(3)
#     machine_control.automf(3)
#     weight_ratio.automf(3)
#
#     # 定义规则
#     rules = [
#         ctrl.Rule(driver_trust['poor'], weight_ratio['good']),
#         ctrl.Rule(driver_trust['average'], weight_ratio['average']),
#         ctrl.Rule(driver_trust['good'], weight_ratio['poor']),
#     ]
#
#     # 创建控制系统
#     weight_ratio_ctrl = ctrl.ControlSystem(rules)
#     weight_ratio_simulation = ctrl.ControlSystemSimulation(weight_ratio_ctrl)
#
#     # 输入值
#     weight_ratio_simulation.input['driver_trust'] = trust_value
#     weight_ratio_simulation.compute()
#
#     driver_weight = weight_ratio_simulation.output['weight_ratio']
#     machine_weight = 1 - driver_weight  # 机器的权重
#
#     return driver_weight, machine_weight
#
# # 假设信任值和机器控制权重的计算
# while True:
#     trust_value = model(image_data, physio_data, eeg_data)
#     machine_value = 1 - trust_value
#
#     # 进行模糊推理
#     control_weight,machine_weight= fuzzy_inference(trust_value)
#     print(f"当前驾驶员的控制权重比例: {control_weight:.1f}",f"当前机器的控制权重比例: {machine_weight:.1f}",)
