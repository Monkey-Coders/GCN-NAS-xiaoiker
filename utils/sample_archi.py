import csv
import random
import re 
path_to_archi = "./Run6/run/work_dir/ntu/xview/agcn_joint_Srch_CTN/architextures.txt"

def format_archi():
    with open(path_to_archi) as f:
        data = f.read()

    data = re.sub(r",\n[\s]*", ",", data)
    data = re.sub(r"tensor\(", "", data)
    data = re.sub(r",device='cuda:[0-9]*'\)", "", data)

    with open(path_to_archi, "w") as f:
        f.write(data)

def open_archi_as_csv():
    #HEADERS: {epoch}|{accuracy}|{loss}|{weights}
    
    with open(path_to_archi) as f:
        archi = csv.reader(f, delimiter="|")
        archi = list(archi)
        idx = random.randint(0,len(archi))
        print(archi[idx])
        
format_archi()
open_archi_as_csv()

#['24', '0.6578059071729958', '1.1208003926314885', "tensor([[0.1224, 0.1261, 0.1178, 0.1259, 0.1257, 0.1279, 0.1273, 0.1270],[0.1236, 0.1296, 0.1173, 0.1282, 0.1304, 0.1261, 0.1198, 0.1250],[0.1203, 0.1338, 0.1280, 0.1255, 0.1151, 0.1290, 0.1220, 0.1263],[0.1247, 0.1270, 0.1225, 0.1221, 0.1259, 0.1224, 0.1368, 0.1186],[0.1277, 0.1285, 0.1173, 0.1170, 0.1291, 0.1325, 0.1216, 0.1264],[0.1222, 0.1217, 0.1245, 0.1278, 0.1177, 0.1255, 0.1335, 0.1270]],device='cuda:0')"]