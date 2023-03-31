import torch
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
n_layers = 10
n_ops = 8

""" weights = torch.ones(n_layers,n_ops)*0.125
weights[-1][6] = 0.01
weights[-1][7] = 0.01 """

weights = [
    [0.1224, 0.1261, 0.1178, 0.1259, 0.1257, 0.1279, 0.1273, 0.1270],
    [0.1236, 0.1296, 0.1173, 0.1282, 0.1304, 0.1261, 0.1198, 0.1250],
    [0.1203, 0.1338, 0.1280, 0.1255, 0.1151, 0.1290, 0.1220, 0.1263],
    [0.1247, 0.1270, 0.1225, 0.1221, 0.1259, 0.1224, 0.1368, 0.1186],
    [0.1277, 0.1285, 0.1173, 0.1170, 0.1291, 0.1325, 0.1216, 0.1264],
    [0.1222, 0.1217, 0.1245, 0.1278, 0.1177, 0.1255, 0.1335, 0.1270],
    [0.1222, 0.1217, 0.1245, 0.1278, 0.1177, 0.1255, 0.1335, 0.1270],
    [0.1222, 0.1217, 0.1245, 0.1278, 0.1177, 0.1255, 0.1335, 0.1270],
    [0.1222, 0.1217, 0.1245, 0.1278, 0.1177, 0.1255, 0.1335, 0.1270],
    [0.1222, 0.1217, 0.1245, 0.1278, 0.1177, 0.1255, 0.01335, 0.01270]
]

#convert weights to tensor
weights = torch.tensor(weights)



model_args = {
    'num_class': 60, 
    'num_point': 25,
    'num_person': 2,
    'graph': 'graph.ntu_rgb_d.Graph',
    'graph_args': {
        'labeling_mode': 'spatial'
    },
    "weights": weights,
}
output_device = 0


model_label = "model.dynamic_model.Model"
Model = import_class(model_label)
model = Model(**model_args).cuda(output_device)
print(model)