import train, test, investigate_factors as invf
from parameter import model_hyper_parameter as hp, training_parameter as trp, testing_parameter as tep, data_analysis_parameter as dp
from tools import set_params
import numpy as np

'''
def pipeline(argv):
    set_params(argv, hp)
    trp["save_path"] = "save/s"+str(hp["seed"])+'/'
    (trp["load_path"], tep["load_path"], dp["load_path"]) = (trp["save_path"], trp["save_path"], trp["save_path"])
    
    train.main([], hp.copy(), trp.copy())
    test.main([], hp.copy(), tep.copy())
    invf.main([], hp.copy(), dp.copy())
    
    trp["training_step"] = 400
    trp["use_BCI_readout"], tep["use_BCI_readout"] = True, True
    perturbation_num = 25
    path = trp["save_path"]
    
    trp["perturbation_type"] = "within_manifold"
    pl_list = []
    for i in range(perturbation_num):
        trp["pick_perturbation"] = i
        trp["save_path"] = path+"p"+str(i)+'/'
        try:
            pl = train.main([], hp.copy(), trp.copy())
        except:
            print("Not enough within manifold perturbations, only {} avalible!!\n".format(i))
            continue
        pl_list.append(pl)
    np.save(path+"within_manifold_perturbation_learning_curves", np.array(pl_list))
    
    trp["perturbation_type"] = "outside_manifold"
    pl_list = []
    for i in range(perturbation_num):
        trp["pick_perturbation"] = i
        trp["save_path"] = path+"op"+str(i)+'/'
        try:
            pl = train.main([], hp.copy(), trp.copy())
        except:
            print("Not enough outside manifold perturbations, only {} avalible!!\n".format(i))
            continue
        pl_list.append(pl)
    np.save(path+"outside_manifold_perturbation_learning_curves", np.array(pl_list))
'''

def pipeline(argv):
    set_params(argv, hp)
    path = "save/s"+str(hp["seed"])+'/'
    tep["load_path"] = path
    
    #test.main([], hp.copy(), tep.copy())
    
    trp["use_BCI_readout"], tep["use_BCI_readout"] = True, True
    perturbation_num = 25
    ve_list, ove_list = [], []
    
    for i in range(perturbation_num):
        
        try:
            tep["perturbation_type"] = "within_manifold"
            tep["load_path"], dp["load_path"] = path+"p"+str(i)+'/', path+"p"+str(i)+'/'
            test.main([], hp.copy(), tep.copy())
            ve_list += [invf.main([], hp.copy(), dp.copy())]
            tep["perturbation_type"] = "outside_manifold"
            tep["load_path"], dp["load_path"] = path+"op"+str(i)+'/', path+"op"+str(i)+'/'
            test.main([], hp.copy(), tep.copy())
            ove_list += [invf.main([], hp.copy(), dp.copy())]
        except:
            print("error at pert {}".format(i))
            break
    np.save(path+"ve_list", [np.array(ve_list), np.array(ove_list)])


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    pipeline(["seed", "1042"])