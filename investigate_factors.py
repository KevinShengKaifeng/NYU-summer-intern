import numpy as np
import matplotlib.pyplot as plt
from model import Model
import tensorflow as tf
from parameter import data_analysis_parameter as dp
from tools import z_score, canonical_angle, simple_plot, show_distribution, show_matrix,\
                 gene_mat, save_message, set_params, vector_to_angle, angle_diff, moving_window_smoothing,\
                 assess_performance, draw_learning_curve
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, kstest
import time


def draw_latent_variables(_z, save_path, sort=False):
    '''
    input _z has shape [time_step, num_direction, batch_size, num_facotr]
    '''
    print("ploting latent variables")
    plt.figure(figsize=(10,15))
    num_loc = _z.shape[1]
    num_fac = dp["optimal_num_factor"]
    colors = ['orange', 'c', 'm', 'r', 'g', 'b', 'k', 'gray']
    if sort:
        variance = [z_score(np.reshape(_z[:, :, :, f], [-1]))[1] for f in range(num_fac)]
        order = np.searchsorted(np.sort(variance), variance)
        ita = np.zeros([num_fac, num_fac])
        for f in range(num_fac):
            ita[f, order[f]] = 1
        _z = _z @ ita
    limit = 20
    ax = plt.gca()
    for f in range(num_fac):
        #plt.plot([-limit, limit], [f, f], c='black')
        ax.axhline(y=f, color='black')
        for d in range(num_loc):
            mean, sigma = z_score(np.reshape(_z[:, d, :, num_fac-f-1], [-1]))
            plt.plot([mean-sigma, mean+sigma], [(d+1)/(num_loc+2)+f, (d+1)/(num_loc+2)+f], c=colors[d], linewidth=4)
            plt.plot([mean, mean], [(d+1)/(num_loc+2)+f-0.025, (d+1)/(num_loc+2)+f+0.025], c=colors[d], linewidth=3)
    
    ax.set_ylim([0,num_fac])
    #ax.set_xlim([-limit,limit])
    ax.set_yticks([])
    if dp["save_figures"]:
        plt.savefig(save_path+'f'+str(num_fac)+"-latent_variables.png", dpi=1000)
    if sort:
        return ita
    plt.figure(figsize=(3*4,num_fac/3*4))
    for f in range(num_fac):
        ax = plt.subplot(np.ceil(num_fac/3),3,f+1,projection='polar')
        points = list(np.mean(_z, axis=(0,2))[:,f])
        points.append(points[0])
        plt.polar(np.arange(num_loc+1)*2*np.pi/num_loc, points)
    

def draw_trail_with_activity(activity, output_weight, axis1, axis2):
    if dp["Dale's law"]:
        base_vector = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        output = activity @ output_weight @ base_vector
    else:
        output = activity @ output_weight
    (step, num_loc, batch_size, num_cell) = activity.shape
    cursor_trails = [np.zeros([num_loc, batch_size, 2])]
    for t in range(step):
        cursor_trails.append(cursor_trails[-1]+dp["delta_t"]/1000*output[t])
    cursor_trails = np.array(cursor_trails)
    for d in range(num_loc):
        points = np.mean(cursor_trails[:,d,:,:], axis=1).T
        axis1.plot(points[0], points[1], c=(d/num_loc, 1-d/num_loc, 0))
        axis1.add_artist(plt.Circle([np.cos(d/num_loc*2*np.pi), np.sin(d/num_loc*2*np.pi)], 0.2, fill=False, color=(d/num_loc, 1-d/num_loc, 0), clip_on=False))
    axis1.set_xlabel("mean trail across batch")
    axis1.set_xlim([-1.2, 1.2])
    axis1.set_ylim([-1.2, 1.2])
    for d in range(num_loc):
        for i in range(batch_size):
            points = cursor_trails[:,d,i,:].T
            axis2.plot(points[0], points[1], c=(d/num_loc, 1-d/num_loc, 0))
    axis2.set_xlabel("all trails from the test set")
    axis2.set_xlim([-1.2, 1.2])
    axis2.set_ylim([-1.2, 1.2])
    return cursor_trails


def draw_trails(firing_rate_data, estimated_activity, output_weight, save_path):
    print("ploting trails")
    fig, [[axis11, axis12], [axis21, axis22]] = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    cursor_trails = draw_trail_with_activity(firing_rate_data, output_weight, axis11, axis12)
    axis11.set_ylabel("actual output")
    if estimated_activity is not None:
        draw_trail_with_activity(estimated_activity, output_weight, axis21, axis22)
        axis21.set_ylabel("estimated output")
    if dp["save_figures"]:
        plt.savefig(save_path+'f'+str(dp["optimal_num_factor"])+"-estimated_trails.png", dpi=1000)
    if estimated_activity is not None:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        output_est_error = np.linalg.norm((firing_rate_data-estimated_activity)@output_weight, axis=3)
        ax1.plot(np.mean(output_est_error, axis=(0,2)))
        ax1.set_ylabel("distance")
        ax1.set_xlabel("target direction")
        ax1.set_title("mean error between actual output and estimated output")
        ax2.set_xlabel("time step")
        ax2.plot(np.mean(output_est_error, axis=(1,2)))
    plt.show()
    return cursor_trails


def draw_error(estimated_activity, firing_rate_data, save_path):
    print("ploting relative error")
    fig, [axis1, axis2] = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    error_list=[]
    num_loc = estimated_activity.shape[1]
    for d in range(num_loc):
        error = np.linalg.norm(estimated_activity[:,d,:,:]-firing_rate_data[:,d,:,:], axis=2)
        mod = np.linalg.norm(firing_rate_data[:,d,:,:], axis=2)
        mean_relative_error_through_time = np.mean(error/mod, axis=1)
        axis1.plot(mean_relative_error_through_time, c=(d/num_loc, 1-d/num_loc, 0))
        error_list.append(np.mean(mean_relative_error_through_time))
    axis2.plot(error_list)
    axis1.set_ylim([0,1])
    axis2.set_ylim([0,1])
    axis1.set_title("relative error across time")
    axis1.set_xlabel("time_step")
    axis2.set_title("mean relative error for different target direction")
    axis2.set_xlabel("target direction")
    if dp["save_figures"]:
        plt.savefig(save_path+'f'+str(dp["optimal_num_factor"])+"-relative_error.png", dpi=1000)


def draw_pushing_vectors(L, output_weight, save_path='save/'):
    print("ploting pushing vectors")
    plt.figure(figsize=(5,5))
    if dp["Dale's law"]:
        base_vector = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        pushing_vectors = L @ output_weight @ base_vector
    else:
        pushing_vectors = L @ output_weight
    num_fac = dp["optimal_num_factor"]
    for v in range(num_fac):
        plt.plot([0, pushing_vectors[v, 0]], [0, pushing_vectors[v, 1]], label=v+1)
    plt.legend()
    ax = plt.gca()
    ax.set_title("pushing vectors")
    if dp["save_figures"]:
        plt.savefig(save_path+'f'+str(num_fac)+"-pushing_vectors.png", dpi=1000)


def fit_cos_tuning(tuning_curve):
    def cos(theta, A, b, psi):
        before_clip = np.cos((theta-psi)/360*2*np.pi)*A+b
        return np.where(before_clip>0, before_clip, 0)
    xdata = np.linspace(0, 360, len(tuning_curve))
    plt.figure(figsize=(5,5))
    opt_psi = []
    for i in range(len(tuning_curve[0])):
        popt, pcov = curve_fit(cos, xdata, tuning_curve[:,i], bounds=([0, -np.inf, 0], [np.inf, np.inf, 360]))
        opt_psi.append(popt[2])
        if i==1:
            plt.plot(xdata, tuning_curve[:,i], label="data"+str(i+1))
            xdata1 = np.linspace(0, 360, 1000)
            plt.plot(xdata1, cos(xdata1, *popt), label="fit"+str(i+1))
    plt.legend()
    plt.show()
    print("kstest of the tuning direction fitting uniform distribution: {}".format(kstest(np.array(opt_psi)/360, 'uniform')))
    return np.array(opt_psi)


def draw_tuning_curves(activity, W_in, delta_t=5.0, save_path="save/"):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    selected_neurons = np.argsort(z_score(np.reshape(activity, [-1, activity.shape[-1]]))[1])[-10:]
    activity = activity[:,:,:,selected_neurons]
    tuning_curve = np.mean(activity, axis=(0,2))
    tuning_curve = np.concatenate([tuning_curve, tuning_curve[0:1]])
    for c in range(activity.shape[-1]):
        ax1.plot(np.arange(0, activity.shape[0]*delta_t, delta_t), np.mean(activity[:,:], axis=(1,2))[:,c])
        ax2.plot(np.linspace(0, 360, len(tuning_curve)), tuning_curve[:,c])
    ax1.set_title("time tuning curves")
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("mean firing rate")
    ax2.set_title("target direction tuning curves")
    ax2.set_xlabel("target direction (degree)")
    direction_tuning = fit_cos_tuning(tuning_curve)
    o_c = W_in[:2, selected_neurons].T
    o_t = W_in[2:, selected_neurons].T
    angle_c = np.arccos(o_c[:,0]/np.linalg.norm(o_c,axis=1))/np.pi*180*np.where(o_c[:,1]<0, -1, 1)+ np.where(o_c[:,1]<0, 360, 0)
    angle_t = np.arccos(o_t[:,0]/np.linalg.norm(o_t,axis=1))/np.pi*180*np.where(o_t[:,1]<0, -1, 1)+ np.where(o_t[:,1]<0, 360, 0)
    for angle in (angle_c, angle_t):
        for i in range(len(angle)):
            if (angle[i] - direction_tuning[i])>180:
                angle[i] -= 360
            if (angle[i] - direction_tuning[i])<-180:
                angle[i] += 360
    r_c, p_value = pearsonr(angle_c, direction_tuning)
    r_t, p_value = pearsonr(angle_t, direction_tuning)
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    plt.scatter(direction_tuning, angle_c, marker='.', label='cursor input')
    plt.scatter(direction_tuning, angle_t, marker='.', label='target input')
    ax.set_ylabel("input pushing vector direction (\N{DEGREE SIGN})")
    ax.set_xlabel("preferred direction (\N{DEGREE SIGN})")
    plt.axis('equal')
    plt.plot([0,360], [0,360])
    plt.text(-100, 300, "R_c: {0:.2f}\nR_t: {1:.2f}".format(r_c, r_t), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
    plt.legend()
    plt.show()
    
    if dp["save_figures"]:
        plt.savefig(save_path+"tuning_curves.png", dpi=1000)
    return direction_tuning


def draw_repertoire_on_output_space(activity, output_weight, delta_t=5, save_path="save/"):
    if dp["Dale's law"]:
        base_vector = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        output = activity @ output_weight @ base_vector
    else:
        output = activity @ output_weight
    num_loc = activity.shape[1]
    off_set = np.array([0.85, 0.85])
    thetas = np.linspace(0, np.pi*2, num_loc+1)[:-1]
    centers = (np.array([np.cos(thetas), np.sin(thetas)])*0.08).T+off_set
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    for d in range(num_loc):
        points = np.reshape(output[:, d, :, :], [-1, 2])
        ax1.scatter(points[:,0], points[:,1], marker='.')
        points2 = np.mean(output[:,d,:,:], axis=1)
        _l, = ax2.plot(points2[:,0], points2[:,1], label=str(d))
        ax1.add_artist(plt.Circle(centers[d], 0.02, fill=True, transform=ax1.transAxes, color=_l.get_color()))
        ax2.add_artist(plt.Circle(centers[d], 0.02, fill=True, transform=ax2.transAxes, color=_l.get_color()))
    ax1.add_artist(plt.Circle(off_set, 0.006, fill=False, transform=ax1.transAxes, color='black'))
    ax2.add_artist(plt.Circle(off_set, 0.006, fill=False, transform=ax2.transAxes, color='black'))
    ax1.set_xlim([-4,4])
    ax1.set_ylim([-4,4])
    ax2.set_xlim([-4,4])
    ax2.set_ylim([-4,4])
    ax1.set_xlabel("output x velocity")
    ax1.set_ylabel("output y velocity")
    ax1.set_title("neural repertorie on output space")
    ax2.set_xlabel("output x velocity")
    ax2.set_ylabel("output y velocity")
    ax2.set_title("neural dynamics on output space")
    
    if dp["save_figures"]:
        plt.savefig(save_path+"neural_repertoire.png", dpi=1000)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    off_set = np.array([0.85, 0.85])
    thetas = np.linspace(0, np.pi*2, num_loc+1)[:-1]
    centers = (np.array([np.cos(thetas), np.sin(thetas)])*0.08).T+off_set
    for d in range(output.shape[1]):
        _l, = ax.plot(np.arange(output.shape[0])*delta_t, np.mean(np.linalg.norm(output, axis=3), axis=2)[:,d])
        ax.add_artist(plt.Circle(centers[d], 0.02, fill=True, transform=ax.transAxes, color=_l.get_color()))
    ax.add_artist(plt.Circle(off_set, 0.006, fill=True, transform=ax.transAxes, color='black'))
    ax.plot(np.arange(output.shape[0])*delta_t, np.mean(np.linalg.norm(output, axis=3), axis=(1,2)), c='black', linewidth=2)
    ax.set_title("speed")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("v (1/s)")
    


def plot_variance_explained(_z, _L, _cov, save_path=""):
    plt.figure()
    z_mean, z_variance = z_score(_z)
    factor_norm = np.linalg.norm(_L, axis=1)
    total_var = np.trace(_cov)
    var_exp = z_variance**2*factor_norm**2/total_var
    cumulative_var = np.sort(var_exp)[::-1]
    plt.plot(np.arange(1, len(var_exp)+1), cumulative_var, marker='.')
    for i in range(1, len(var_exp)):
        cumulative_var[i] += cumulative_var[i-1]
    plt.plot(np.arange(1, len(var_exp)+1), cumulative_var, marker='.')
    #print(cumulative_var)
    ax = plt.gca()
    ax.set_ylim([0, 1 if max(cumulative_var)<1 else max(cumulative_var)])
    if dp["save_figures"]:
        plt.savefig(save_path+"variance_explained.png", dpi=1000)
    return cumulative_var


def plot_angles_of_vectors_in_space(space, vectors):
    plt.figure()
    for row in range(len(vectors)):
        angles = np.array([canonical_angle(space[:i+1], vectors[row:row+1]) for i in range(len(space))])
        plt.plot(angles/np.pi*180, label="vector "+str(row+1))
    ax = plt.gca()
    ax.set_xlabel("dim of space")
    ax.set_ylabel("angle between vectors and projection into space")
    plt.legend()


def draw_latent_variables_dynamic(z, f1, f2=None, delta_t=5):
    '''
    z.shape=[time_step, num_loc, batch_size, num_fac]
    '''
    z = np.mean(z, axis=2)
    num_loc = z.shape[1]
    off_set = np.array([0.15, 0.15])
    thetas = np.linspace(0, np.pi*2, num_loc+1)[:-1]
    centers = (np.array([np.cos(thetas), np.sin(thetas)])*0.08).T+off_set
    if f2 is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        for d in range(num_loc):
            line, = ax.plot(np.arange(0, z.shape[0]*delta_t, delta_t), z[:,d,f1], label=d)
            ax.add_artist(plt.Circle(centers[d], 0.02, fill=True, transform=ax.transAxes, color=line.get_color()))
        ax.add_artist(plt.Circle(off_set, 0.006, fill=False, transform=ax.transAxes, color='black'))
        ax.set_xlabel("time")
        ax.set_ylabel("factor {}".format(f1+1))
    else:    
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        for d in range(num_loc):
            line, = ax2.plot(z[:,d,f1], z[:,d,f2], label=d)
            points = ax1.scatter(z[:,d,f1], z[:,d,f2], c=[i*delta_t for i in range(z.shape[0])], cmap='jet', marker='.')
            ax2.add_artist(plt.Circle(centers[d], 0.02, fill=True, transform=ax2.transAxes, color=line.get_color()))
        ax2.add_artist(plt.Circle(off_set, 0.006, fill=False, transform=ax2.transAxes, color='black'))
        cax = fig.add_axes([0.15, 0.15, 0.02, 0.36])
        fig.colorbar(points, cax=cax, orientation='vertical')
        ax1.set_xlabel("factor {}".format(f1+1))
        ax2.set_xlabel("factor {}".format(f1+1))
        ax1.set_ylabel("factor {}".format(f2+1))
        plt.axis('equal')
        '''
        ax1.set_xlim([-4, 4])
        ax1.set_ylim([-4, 4])
        ax2.set_xlim([-4, 4])
        ax2.set_ylim([-4, 4])
        '''


def show_weight_diff(weight_diff, save_path=''):
    fig = plt.figure(figsize=(5, 5))
    weight_diff = np.array(weight_diff).T
    plt.plot(weight_diff[0], marker='.', label='within manifold difference')
    plt.plot(weight_diff[1], marker='.', label='outside manifold difference')
    plt.plot(weight_diff[2], marker='.', label='prediction weight outside manifold part')
    plt.legend()
    plt.show()
    if dp["save_figures"]:
        plt.savefig(save_path+"output_weight_difference.png", dpi=1000)
        

def show_angle_diff(angle_diff, save_path=''):
    fig = plt.figure(figsize=(5, 5))
    angle_diff = np.array(angle_diff).T
    line, = plt.plot(angle_diff[0], marker='.', label='within manifold angle 1')
    plt.plot(angle_diff[1], marker='.', label='within manifold angle 2', color=line.get_color())
    line, = plt.plot(angle_diff[2], marker='.', label='outside manifold angle 1')
    plt.plot(angle_diff[3], marker='.', label='outside manifold angle 2', color=line.get_color())
    line, = plt.plot(angle_diff[4], marker='.', label='prediction weight & manifold 1')
    plt.plot(angle_diff[5], marker='.', label='prediction weight & manifold 2', color=line.get_color())
    plt.legend()
    plt.show()
    if dp["save_figures"]:
        plt.savefig(save_path+"output_weight_difference.png", dpi=1000)


def activity_to_cursor_trail():
    pass


def all_perm(dim):
    if dim == 1:
        return [[0]]
    else:
        n_perm = []
        for p in all_perm(dim-1):
            for i in range(dim):
                n_perm.append(p[:i]+[dim-1]+p[i:])
        return n_perm


def choose_perturbed_readout(firing_rate_data, output_weight, num_fac=10, min_angle=15, max_angle=60, save_path="save/"):
    transformer = PCA()
    transformer.fit_transform(np.reshape(firing_rate_data, [-1, 500]))
    L = transformer.components_[:num_fac]
    o_list = []
    W_para = L.T @ L @ output_weight
    perm_list = all_perm(num_fac)
    activity = np.mean(firing_rate_data[25:125], axis=(0,2))
    La = L @ activity.T
    nL = np.zeros(La.shape)
    Lo = L @ output_weight
    for p in perm_list:
        nL[p] = La
        o_list.append(nL.T @ Lo) 
    angle = vector_to_angle(activity @ W_para)
    angle_p = vector_to_angle(np.array(o_list))
    abs_diff = angle_diff(angle, angle_p, if_abs=True)
    show_distribution(np.mean(abs_diff, axis=1))
    compatibility = np.sum(np.where(abs_diff>min_angle, 1, 0)*np.where(abs_diff<max_angle, 1, 0), axis=1)
    eligible_p_idx = np.argwhere(compatibility==8)[:,0]
    if len(eligible_p_idx) == 0:
        print("NO eligible within manifold perturbation!!")
        return
    np.random.shuffle(eligible_p_idx)
    mean_angle_diff = np.mean(abs_diff[eligible_p_idx], axis=1)
    W_pert_list = []
    for p in eligible_p_idx:
        W_pert_list.append(L.T@gene_mat(perm_list[p])@L@output_weight)
    show_distribution(mean_angle_diff)
    W_pert_list = np.array(W_pert_list)
    np.save(save_path+"W_pert_list", W_pert_list)
    np.save(save_path+"W_angle_diff_list", mean_angle_diff)
    print("number of eligible within manifold perturbation: {0}, mean angle difference: {1:.2f}\N{DEGREE SIGN}".format(W_pert_list.shape[0], np.mean(mean_angle_diff)))
    return W_pert_list


def choose_outside_manifold_readout(firing_rate_data, output_weight, num_fac=10, cell_per_group=15, min_angle=15, max_angle=60, save_path="save/"):
    mean, var = z_score(np.reshape(firing_rate_data, [-1, 500]))
    rng = np.random.RandomState(0)
    chosen_cells = np.argsort(var)[-cell_per_group*num_fac:]
    rng.shuffle(chosen_cells)
    groups = chosen_cells.reshape([num_fac, cell_per_group])
    transformer = PCA()
    transformer.fit_transform(np.reshape(firing_rate_data, [-1, 500]))
    L = transformer.components_[:num_fac]
    o_list = []
    W_para = L.T @ L @ output_weight
    W_para_permuted = np.copy(W_para)
    perm_list = all_perm(num_fac)
    activity = np.mean(firing_rate_data[25:125], axis=(0,2))
    for perm in perm_list:
        for i in range(num_fac):
            W_para_permuted[groups[perm[i]]] = W_para[groups[i]]
        o_list.append(activity@W_para_permuted)
    angle = vector_to_angle(activity @ W_para)
    angle_p = vector_to_angle(np.array(o_list))
    abs_diff = angle_diff(angle, angle_p, if_abs=True)
    show_distribution(np.mean(abs_diff, axis=1))
    compatibility = np.sum(np.where(abs_diff>min_angle, 1, 0)*np.where(abs_diff<max_angle, 1, 0), axis=1)
    eligible_p_idx = np.argwhere(compatibility==8)[:,0]
    if len(eligible_p_idx) == 0:
        print("NO eligible outside manifold perturbation!!")
        return
    np.random.shuffle(eligible_p_idx)
    W_pert_list = []
    mean_angle_diff = np.mean(abs_diff[eligible_p_idx], axis=1)
    for perm in eligible_p_idx:
        for i in range(num_fac):
            W_para_permuted[groups[perm_list[perm][i]]] = W_para[groups[i]]
        W_pert_list.append(W_para_permuted.copy())
    show_distribution(mean_angle_diff)
    W_pert_list = np.array(W_pert_list)
    np.save(save_path+"W_out_pert_list", W_pert_list)
    np.save(save_path+"W_out_angle_diff_list", mean_angle_diff)
    print("number of eligible outside manifold perturbation: {0}, mean angle difference: {1:.2f}\N{DEGREE SIGN}".format(W_pert_list.shape[0], np.mean(mean_angle_diff)))
    return W_pert_list


def choose_time_period(activity, W_out):
    v_t = np.linalg.norm(np.mean(activity@W_out, axis=(1,2)), axis=-1)
    v_max = np.max(v_t)
    period = np.max(np.argwhere(v_t/v_max>0.1)[:,0])
    return period


def my_PCA():
    pass


def main(argv=[], hp=None, dp=None):
    if hp is None:
        from parameter import model_hyper_parameter as hp
    if dp is None:
        from parameter import data_analysis_parameter as dp
    dp = set_params(argv, dp)
    
    
    load_path = dp["load_path"]
    num_loc = dp["number_of_location"]
    num_fac = dp["optimal_num_factor"]
    filename = dp["method"]+"_results_"+str(num_fac)+"_components.npy"
    '''
    _z, _Lambda, _Psi, _activity_mean, _cov = np.load(load_path+filename)
    am = np.expand_dims(_activity_mean, axis=0)
    _z = np.reshape(_z, [dp["time_step"], num_loc, -1, num_fac])
    q, r = np.linalg.qr(_Lambda.T)
    _L_orth = q.T
    _z_orth = _z @ r.T
    estimated_activity = _z @ _Lambda + _activity_mean
    '''
    
    firing_rate_data = np.load(load_path[:load_path.find('o' if 'o' in load_path else 'p')]+"default_test-activity.npy")#[:dp["time_step"],:,:dp["cell_number"]]
    firing_rate_data = np.reshape(firing_rate_data, [dp["time_step"], num_loc, -1, dp["cell_number"]])
    
    firing_rate_data1 = np.load(load_path+"default_test-activity_p.npy")
    firing_rate_data1 = np.reshape(firing_rate_data1, [dp["time_step"], num_loc, -1, dp["cell_number"]])
    
    '''
    firing_rate_data2 = np.load(load_path+"default_test-activity_pb.npy")
    firing_rate_data2 = np.reshape(firing_rate_data2, [dp["time_step"], num_loc, -1, dp["cell_number"]])
    firing_rate_data3 = np.load(load_path+"default_test-activity_w.npy")
    firing_rate_data3 = np.reshape(firing_rate_data3, [dp["time_step"], num_loc, -1, dp["cell_number"]])
    '''
    '''
    plt.figure(figsize=(15,10))
    plt.scatter(*z_score(firing_rate_data), marker='.', label="intuitive mapping")
    plt.scatter(*z_score(firing_rate_data2), marker='.', label="perturbed mapping, before training")
    plt.scatter(*z_score(firing_rate_data1), marker='.', label="perturbed mapping, after training")
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel("mean firing rate")
    ax.set_ylabel("firing rate variance")
    '''
    
    '''
    model = Model(hp)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_model(sess, load_path+"prediction_checkpoint.ckpt")#"within_manifold_perturbation_prediction_checkpoint.ckpt"
        weight_list = sess.run(model.weight_list[0])
        input_weight = weight_list[0]
        output_weight = weight_list[2]
        W_BCI = weight_list[-1]
        model.load_model(sess, load_path+"within_manifold_perturbation_prediction_checkpoint.ckpt")
        weight_list1 = sess.run(model.weight_list[0])
        output_weight1 = weight_list1[2]
        W_BCI1 = weight_list1[-1]
    '''
        
        
    '''
    W_L = _L_orth.T 
    W_ = _L_orth @ output_weight
    W_out_para = _L_orth.T@_L_orth@output_weight
    W_out_perp = output_weight - W_out_para
    '''
    
    #plot_angles_of_vectors_in_space(_Lambda, output_weight.T)
    #print([np.tan(canonical_angle(_Lambda, output_weight.T[i:i+1])) for i in range(2)])
    #print([np.linalg.norm(W_out_para.T[i]) for i in range(2)])
    #print([np.linalg.norm(W_out_perp.T[i]) for i in range(2)])
    '''
    
    canonical_angle(am, _Lambda)
    canonical_angle(am@_Lambda.T@_Lambda, W_out_para.T[1:2])
    canonical_angle(am - am@_Lambda.T@_Lambda, W_out_perp.T[1:2])
    print(_activity_mean@output_weight, _activity_mean@W_out_para, (am - am@_Lambda.T@_Lambda)@W_out_perp)
    [canonical_angle(_Lambda, output_weight.T[i:i+1]) for i in range(2)]
    canonical_angle(_Lambda, output_weight.T)
    canonical_angle(input_weight, _Lambda)
    canonical_angle((np.array([[1,0,-1,0],[0,1,0,-1]])@input_weight)[0:1], _Lambda)
    canonical_angle(output_weight.T, input_weight)
    canonical_angle(output_weight.T, (np.array([[1,0,-1,0],[0,1,0,-1]])@input_weight))
    '''
    #canonical_angle(np.array([[1,0,-1,0],[0,1,0,-1]])@output_weight.T, _L_orth)
    #canonical_angle(np.array([[1,0,-1,0],[0,1,0,-1]])@output_weight.T, np.array([[1,0,-1,0],[0,1,0,-1]])@input_weight)
    #canonical_angle(np.load(load_path+"FA_results_"+str(10)+"_components.npy")[1], _Lambda)
    #plot_variance_explained(np.reshape(_z,[-1, num_fac]), _Lambda, _cov, load_path)
    #draw_latent_variables(_z, load_path)
    #ita = draw_latent_variables(_z_orth, load_path+'o-', True)
    #plot_variance_explained(np.reshape(_z_orth,[-1, num_fac]), _L_orth, _cov, load_path+'o-')

    #draw_repertoire_on_output_space(estimated_activity-_z @ _Lambda, output_weight, load_path)
    #draw_pushing_vectors(_Lambda, output_weight, load_path)
    #draw_pushing_vectors(ita.T@_L_orth, output_weight, load_path+'o-')
    
    #draw_trails(firing_rate_data, estimated_activity, W_BCI, load_path)
    #draw_trails(firing_rate_data, estimated_activity, W_out_perp, load_path)
    #draw_tuning_curves(firing_rate_data, load_path)
    #draw_error(estimated_activity, firing_rate_data, load_path)
    #tail_activity = np.mean(firing_rate_data[-100:], axis=(0,2))
    data = firing_rate_data
    mean, sigma = z_score(np.reshape(data, [-1, 500]))
    #show_distribution(sigma)
    #data = (data-mean)/(sigma+1e-2)
    time_period = 400#choose_time_period(data, W_BCI)
    time_period = 125 if time_period<125 else time_period
    save_message("default", "time_period_length-{}\n".format(time_period))
    transformer = PCA()
    num_fac = 10
    z = transformer.fit_transform(np.reshape(data[:time_period], [-1, 500]))
    z = np.reshape(z[:,:num_fac], [time_period, num_loc, -1, num_fac])
    L = transformer.components_[:num_fac]
    cov = transformer.get_covariance()
    np.save(load_path+filename, np.array([z, L, mean]))
    #np.save(load_path+"W_para", L.T@L@W_BCI)
    #np.save(load_path+"W_para_pred", L.T@L@output_weight)
    

    
    transformer1 = PCA()
    mean1, sigma1 = z_score(np.reshape(firing_rate_data1, [-1, 500]))
    #firing_rate_data1 = (firing_rate_data1 - mean1)/(sigma1+1e-2)
    z1 = transformer1.fit_transform(np.reshape(firing_rate_data1[:], [-1, 500]))
    z1 = np.reshape(z1[:,:num_fac], [dp["time_step"], num_loc, -1, num_fac])  
    L1 = transformer1.components_[:num_fac]
    cov1 = transformer1.get_covariance()
    #simple_plot([canonical_angle(L1, L[i]) for i in range(10)])
    before_ve = plot_variance_explained(np.reshape(firing_rate_data@L.T,[-1, 10]), L, cov, load_path)
    after_ve = plot_variance_explained(np.reshape(firing_rate_data1@L.T,[-1, 10]), L, cov1, load_path)
    relative_ve = after_ve/before_ve*100
    return before_ve, after_ve, relative_ve
    #simple_plot(np.arange(1, 11), relative_ve, ylabel="relative variance explained (%)")
    
    #print(np.mean(np.expand_dims(z, axis=-1)@np.expand_dims(z, axis=3), axis=(0,1,2)).trace())
    #print(np.mean(np.expand_dims(data, axis=-1)@np.expand_dims(data, axis=3), axis=(0,1,2)).trace())
    #print(np.mean(np.expand_dims(firing_rate_data1@L.T, axis=-1)@np.expand_dims(firing_rate_data1@L.T, axis=3), axis=(0,1,2)).trace())
    #print(np.mean(np.expand_dims(firing_rate_data1, axis=-1)@np.expand_dims(firing_rate_data1, axis=3), axis=(0,1,2)).trace())
    #print((((mean/sigma)@L.T@L)*sigma)@W_BCI, np.mean(firing_rate_data, axis=(0,2))[0]@W_BCI)
    covar1 = np.mean(np.expand_dims(firing_rate_data1@L.T, axis=-1)@np.expand_dims(firing_rate_data1@L.T, axis=3), axis=(0,1,2))
    covar = np.mean(np.expand_dims(z, axis=-1)@np.expand_dims(z, axis=3), axis=(0,1,2))
    delta_var = (np.diag(covar1) - np.diag(covar))/np.diag(covar)
    
    delta_pushing_magnitude = np.linalg.norm(L@W_BCI1, axis=1) - np.linalg.norm(L@W_BCI, axis=1)
    #simple_plot(delta_pushing_magnitude, delta_var, method="scatter")
    #print(np.mean(angle_diff(vector_to_angle(L@output_weight), vector_to_angle(L@W_BCI1))))
    #print(np.mean(angle_diff(vector_to_angle(L@output_weight1), vector_to_angle(L@W_BCI1))))
    #print(np.linalg.norm(L@ (output_weight-W_BCI1)))
    #draw_pushing_vectors(L, W_BCI1)
    #print(np.linalg.norm(L@ (output_weight1-W_BCI1)))

    #show_weight_diff(weight_diff)
    #show_angle_diff(angle_diff)
    #simple_plot(moving_window_smoothing(activity_angle_list, 20))
    #simple_plot(np.linalg.norm(np.reshape(firing_rate_data1@L.T@L, [-1, 500]), axis=1), np.linalg.norm(np.reshape(firing_rate_data1-firing_rate_data1@L.T@L, [-1, 500]), axis=1), equal_axis=True, method="scatter")
    
    
    #draw_latent_variables_dynamic(z, 0, 2)
    #draw_latent_variables(z, load_path)
    #draw_repertoire_on_output_space(data, W_BCI, dp["delta_t"], load_path)

    #draw_trails(firing_rate_data3, firing_rate_data3@L.T@L, W_BCI, load_path)
    
    #draw_trails(data, data@L.T@L, output_weight, load_path)
    
    #print(np.linalg.norm(L.T@L@W_BCI),np.linalg.norm(L.T@L@output_weight),np.linalg.norm(L.T@L@(W_BCI-output_weight)))
    #draw_trails(firing_rate_data3, firing_rate_data3@L.T@L, output_weight1, load_path)
    #draw_tuning_curves(firing_rate_data, input_weight, delta_t=dp["delta_t"], save_path=load_path)
    #W_pert_list = np.load("save/W_pert_list.npy")
    #W_out_pert_list = np.load("save/W_out_pert_list.npy")
    #simple_plot([canonical_angle(W_pert.T[1], L, if_print=False) for W_pert in W_out_pert_list])
    #print(output_weight[:,0]@output_weight[:,1],(output_weight[:,0]@output_weight[:,0]),(output_weight[:,1]@output_weight[:,1]))
    
    
    ve = transformer.explained_variance_ratio_[:20]*100
    cumulative_ve = [0]
    for i in range(len(ve)):
        if i==0:
            cumulative_ve.append(transformer.explained_variance_ratio_[i]*100)
        else:
            cumulative_ve.append(cumulative_ve[-1]+transformer.explained_variance_ratio_[i]*100)
    #print(cumulative_ve)
    #simple_plot(np.arange(1, len(ve)+1), ve, xlabel="factor", ylabel="variance explained (%)")
    #simple_plot(cumulative_ve, xlabel="factor", ylabel="acumulative variance explained (%)", ylimit=[0,100])
    np.save(load_path+"variance_explained", np.array([ve, cumulative_ve]))
    
    '''
    with open("investigate_factors.txt", 'a') as file:
        file.write(str(cumulative_ve))
        file.write(',\n')
    
    #plot_variance_explained(np.reshape(_z@_Lambda@L1.T, [-1, num_fac]), L1, _cov)
    canonical_angle(_Lambda, L1)
    #simple_plot([canonical_angle(_Lambda, L1[i:i+1], False) for i in range(12)], ylimit=[0,90])
    canonical_angle(output_weight.T, W_BCI.T[0])
    canonical_angle(output_weight.T, W_BCI.T[1])
    canonical_angle(L1, tail_activity)
    #print(np.linalg.norm(np.linalg.qr((tail_activity@L1.T@L1).T)[0].T, axis=1))
    #simple_plot([canonical_angle(W_out_para.T[1], _Lambda[i:i+1], False) for i in range(10)], ylimit=[0,90])
    transformer = PCA()
    transformer.fit_transform(tail_activity@L1.T@L1)
    #simple_plot([canonical_angle(L1.T@L1@W_BCI.T[0], transformer.components_[i:i+1], False) for i in range(12)], ylimit=[0,90])
    #show_matrix(np.concatenate((L1, (W_out_para/np.linalg.norm(W_out_para, axis=0)).T))@_Lambda.T)
    canonical_angle(W_out_para.T[0], L1)
    canonical_angle(W_out_para.T[1], L1)
    canonical_angle(W_out_para.T, tail_activity)
    #simple_plot(tail_activity)
    '''
    
    #choose_perturbed_readout(firing_rate_data[:time_period], W_BCI, save_path=load_path)
    #W_pert_list = np.load("save/W_pert_list.npy")
    #draw_trails(firing_rate_data, None, W_pert_list[2], load_path)
    #choose_outside_manifold_readout(firing_rate_data[:time_period], W_BCI, save_path=load_path)
    
    
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])