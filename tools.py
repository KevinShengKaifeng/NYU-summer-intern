import matplotlib.pyplot as plt
import numpy as np


#-----------------------  *visualization*  -----------------------#
if_display = True

def show_activity(activity_list, num_shown=None, save_path=""):
    plt.figure()
    activity_list = np.array(activity_list)
    batch_size = activity_list.shape[1]
    if num_shown is None or num_shown > batch_size:
        num_shown = batch_size
    subplot_per_row = 1 if num_shown is 1 else 2
    for i in range(num_shown):
        plt.subplot(np.ceil(num_shown/subplot_per_row), subplot_per_row, i+1)
        plt.imshow(activity_list[:,i,:].T)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path+"activity.png", dpi=1000)
    if if_display:
        plt.show()
        plt.close()


def show_distribution(weight, column_num=100, save_path="", name=""):
    plt.figure()
    weight = np.reshape(np.array(weight), [-1])
    min_value = np.min(weight)
    max_value = np.max(weight)
    value_range = max_value - min_value
    distribution = np.zeros(column_num)
    for value in weight:
        idx = int(np.floor((value - min_value)/value_range*column_num))
        if idx > column_num - 1:
            idx = column_num - 1
        distribution[idx] += 1
    plt.plot((np.arange(min_value, max_value, value_range/column_num) + value_range/column_num/2)[:column_num], distribution)
    if name:
        ax=plt.gca()
        ax.set_title("distribution of "+name)
    if save_path:
        plt.savefig(save_path+"distribution.png", dpi=1000)
    if if_display:
        plt.show()
        plt.close()


def show_connection(weight_list, save_path=""):
    '''
    weight_list should compose of (input_weight, recurrent_weight, output_weight)
    '''
    assert len(weight_list)>=3
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(weight_list[0])
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(weight_list[1])
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(weight_list[2])
    plt.colorbar()
    if save_path:
        plt.savefig(save_path+"connection.png", dpi=1000)
    if if_display:
        plt.show()
        plt.close()
    show_distribution(weight_list[1], save_path=save_path, name="recurrent weight")


def show_cursor_trail(cursor_trail, num_shown=None, target_location=None, target_size=0.2, delta_t=5, name="", save_path=""):
    plt.figure()
    cursor_trail = np.array(cursor_trail)
    batch_size = cursor_trail.shape[1]
    time_step = cursor_trail.shape[0]
    if len(target_location.shape) == 2:
        target_location = np.expand_dims(target_location, axis=0)
    if num_shown is None or num_shown > batch_size:
        num_shown = batch_size
    subplot_per_row = 1 if num_shown is 1 else 2
    for i in range(num_shown):
        plt.subplot(np.ceil(num_shown/subplot_per_row), subplot_per_row, i+1)
        step_size = int(batch_size/num_shown)
        if target_location is not None:
            ax = plt.gca()
            radius = 1.5*np.linalg.norm(target_location[0, 0])
            ax.set_xlim((-radius, radius))
            ax.set_ylim((-radius, radius))
            ax.add_artist(plt.Circle(target_location[0, i*step_size], target_size, fill=False, color='blue', clip_on=False))
            ax.add_artist(plt.Circle(target_location[-1, i*step_size], target_size, fill=False, color='red', clip_on=False))
        plt.scatter(cursor_trail[:,i*step_size,0], cursor_trail[:,i*step_size,1], c=[i*delta_t for i in range(time_step)], cmap='jet', marker='.')
        plt.colorbar()
    if name:
        plt.set_title(name)
    if save_path:
        plt.savefig(save_path+name+"cursor_trail.png", dpi=1000)
    if if_display:
        plt.show()
        plt.close()


def show_loss(loss_list, loss_list2=None, loss_list3=None, loss_list4=None, loss_threshold=None, save_path=""):
    plt.figure()
    bin_size = 30
    line, = plt.plot(loss_list, marker='.', alpha=0.10)
    plt.plot(moving_window_smoothing(loss_list, bin_size), label="target error", color=line.get_color(), linewidth=2)
    if loss_list2 is not None:
        line, = plt.plot(loss_list2, marker='.', alpha=0.10)
        plt.plot(moving_window_smoothing(loss_list2, bin_size), label="prediction error", color=line.get_color(), linewidth=2)
    if loss_list3 is not None:
        line, = plt.plot(loss_list3, marker='.', alpha=0.10)
        plt.plot(moving_window_smoothing(loss_list3, bin_size), label="alignment error", color=line.get_color(), linewidth=2)
    if loss_list4 is not None:
        line, = plt.plot(loss_list4, marker='.', alpha=0.10)
        plt.plot(moving_window_smoothing(loss_list4, bin_size), label="total loss", color=line.get_color(), linewidth=2)
    if loss_threshold is not None:
        plt.plot([0, len(loss_list)], [loss_threshold, loss_threshold], c='r', label="threshold")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.gca().set_ylim((0, 1.0))
    if save_path:
        plt.savefig(save_path+"loss.png", dpi=1000)
    if if_display:
        plt.show()
        plt.close()


def show_weight_norm(weight_norm_list, save_path=""):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for i in range(len(weight_norm_list)//2):
        ax1.plot(weight_norm_list[i], label="delta weight "+str(i)+", total")
        ax2.plot(weight_norm_list[i+len(weight_norm_list)//2], label="delta weight "+str(i)+", last step")
    ax1.legend()
    ax2.legend()
    if save_path:
        plt.savefig(save_path+"weight_norm.png", dpi=1000)
    if if_display:
        plt.show()


def show_each_row(data, save_path=''):
    num_row = len(data)
    plt.figure()
    for i in range(num_row):
        plt.subplot(np.ceil(num_row/2), 2, i+1)
        plt.plot(data[i])
    if save_path:
        plt.savefig(save_path+".factors.png", dpi=1000)
    if if_display:
        plt.show()
        plt.close()


def simple_plot(x, *arg, c=None, xlimit=None, ylimit=None, xlabel=None, ylabel=None, equal_axis=False, method="plot"):
    plt.figure(figsize=(5,5))
    if method == "plot":
        plot = plt.plot
    elif method == "scatter":
        plot = plt.scatter
    if not arg:
        plot(x, marker='.', c=c)
    else:
        for i in range(len(arg)):
            plot(x, arg[i], marker='.', label=i+1, c=c)
        if len(arg) > 1:
            plt.legend()
    ax = plt.gca()
    if xlimit is not None:
        ax.set_xlim(xlimit)
    if ylimit is not None:
        ax.set_ylim(ylimit)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if equal_axis:
        plt.axis('equal')
    if if_display:
        plt.show()
        plt.close()
        

def show_matrix(matrix):
    plt.figure()
    plt.imshow(matrix)
    plt.colorbar()
    if if_display:
        plt.show()
        plt.close()
    

def moving_window_smoothing(data, bin_size):
    data = np.array(data)
    assert len(data.shape) == 1
    o_data = data.copy()
    for i in range(len(data)):
        if i < bin_size//2:
            data[i] = np.nanmean(o_data[:i+bin_size//2])
        elif i >= len(data)-bin_size//2:
            data[i] = np.nanmean(o_data[i-bin_size//2:])
        else:
            data[i] = np.nanmean(o_data[i-bin_size//2:i+bin_size//2])
    return data
    

def performance_list_preprocessing(performance_list, baseline, washout, bin_size):
    if len(np.array(performance_list).shape) == 3:
        pl = []
        for p in performance_list:
            pl.append(performance_list_preprocessing(p, baseline, washout, bin_size))
        return np.array(pl)
    pl = np.array(performance_list).T
    #num_trial = len(pl[0])
    #print("target not reached in {} trials out of {} trials".format(len(np.argwhere(pl[1]==0)), num_trial))
    pl[1][pl[1]==0] = np.nan
    b, t, w = pl[:, :baseline], pl[:, baseline:-washout], pl[:, -washout:]
    for block in [b, t, w]:
        block[0] = moving_window_smoothing(block[0], bin_size)
        block[1] = moving_window_smoothing(block[1], bin_size)
    pl = np.concatenate((b,t,w), axis=1)
    return pl


def draw_learning_curve(performance_list, baseline, washout, max_time=2000, bin_size=30, name="", save_path=""):
    pl = performance_list_preprocessing(performance_list, baseline, washout, bin_size)
    pl_mean = None
    if len(pl.shape) == 2:
        pl = np.expand_dims(pl, axis=0)
    else:
        pl_mean = np.nanmean(pl, axis=0)
    num_trial = pl.shape[-1]
    fig = plt.figure()
    ax1 = plt.gca()
    color = 'black'
    ax1.set_xlabel('trial')
    ax1.set_ylabel('success rate (%)', color=color)
    xrange = np.arange(-baseline, num_trial-baseline)
    for p in pl:
        ax1.plot(xrange, p[0]*100, color=color, alpha=0.5)        
    ax1.set_ylim([0,100*1.05])
    ax1.set_xlim([-baseline, num_trial-baseline])
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('acquisition time (ms)', color=color)
    ax2.set_ylim([0,max_time*1.05])
    for p in pl:
        ax2.plot(xrange, p[1], color=color, alpha=0.5)
    if pl_mean is not None:
        ax1.plot(xrange, pl_mean[0]*100, color='orange')
        ax2.plot(xrange, pl_mean[1], color='red')
    ax2.tick_params(axis='y', labelcolor=color)
    if baseline != 0:
        plt.axvline(x=0, color='red', linestyle='--')
    if washout != 0:
        plt.axvline(x=num_trial-washout-baseline, color='red', linestyle='--')
    #ax2.axhline(y=1000, color='red', linestyle=':')
    if name:
        ax1.set_title(name)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path+"learning_curve.png", dpi=1000)
    if if_display:
        plt.show()
        plt.close()


def access_learning(performance_list, baseline, washout, bin_size=50):
    pl = performance_list_preprocessing(performance_list, baseline, washout, bin_size)
    mean, sigma = z_score(pl.T)
    zscored_pl = (pl.T - mean)/sigma
    P_B = np.nanmean(zscored_pl[:baseline], axis=0)
    P_P1 = zscored_pl[baseline]
    L_max = P_B - P_P1
    L_raw = zscored_pl[baseline:-washout] - P_P1
    L_proj = L_raw@L_max/(L_max@L_max)
    P_A = zscored_pl[-washout]
    after_effect = np.linalg.norm(P_B - P_A)
    relative_after_effect = (P_B - P_A)@L_max/(L_max@L_max)
    amount_of_learning = np.max(L_proj)
    learning_time = np.argmax(L_proj)+1
    return amount_of_learning, learning_time, after_effect, relative_after_effect
    

def compare_histogram(h1, h2, xname, save_path=""):
    plt.figure(figsize=(5,5))
    h1, h2 = h1[~np.isnan(h1)], h2[~np.isnan(h2)]
    mean1, sigma1 = z_score(h1)
    mean2, sigma2 = z_score(h2)
    sigma1 = sigma1/np.sqrt(len(h1))
    sigma2 = sigma2/np.sqrt(len(h2))
    plt.hist(h1, bins=20, alpha=0.5, color='red', label="within")
    plt.hist(h2, bins=20, alpha=0.5, color='blue', label="outside")
    ax = plt.gca()
    plt.axvline(mean1, color='red', linestyle='--')
    plt.axvline(mean2, color='blue', linestyle='--')
    y = ax.transLimits.inverted().transform((0, 0.9))[1]
    plt.plot([mean1-sigma1, mean1+sigma1], [y, y], color='red', linewidth=3)
    plt.plot([mean2-sigma2, mean2+sigma2], [y+0.1, y+0.1], color='blue', linewidth=3)
    plt.legend()
    ax.set_xlabel(xname)
    if save_path:
        plt.savefig(save_path+xname+" histogram", dpi=1000)
    

#--------------------------------------------------------------------#
def generate_target(tp, rng, set_target_position=None):
    '''
    generate target array with shape [batch_size, 2], using the given rng
    pass argument set_target_position to specifiy a single target for the whole batch
    '''
    location_num = tp["number_of_location"]
    theta_range = [np.pi*2*i/location_num for i in range(location_num)]
    if "no_bias_test" in tp and tp["no_bias_test"]:
        assert tp["batch_size"]%location_num==0
        theta_choice = np.concatenate([np.ones(int(tp["batch_size"]/location_num))*theta for theta in theta_range])
    elif set_target_position is not None:
        assert 0 <= set_target_position <= location_num
        theta_choice = np.ones(tp["batch_size"]) * theta_range[set_target_position]
    else:
        theta_choice = rng.choice(theta_range, tp["batch_size"])
    target = np.array([np.cos(theta_choice), np.sin(theta_choice)]).T * tp["initial_target_distance"]
    if "moving_target" in tp and tp["moving_target"]:
        velocity_theta = rng.rand(tp["batch_size"])*2*np.pi
        target_velocity = np.array([np.cos(velocity_theta), np.sin(velocity_theta)]).T * tp["moving_velocity"]
        time_step = int(tp["trial_time"]/tp["delta_t"])
        target = np.array([target + target_velocity*step*tp["delta_t"]/1000 for step in range(time_step)])
    return target


def assess_performance(trail, target_location, tp):
    '''
    arg:
        trail: cursor trail for a batch
        target_location: target location for the batch
        tp: testing parameter, "target_size" and "delta_t" are used
    return:
        success rate and mean acquisition time for the test batch
    '''
    if "moving_target" not in tp or not tp["moving_target"]:
        target_location = np.expand_dims(target_location, axis=0)
    distance = np.sqrt(np.sum(np.square(trail - target_location), axis=2))
    if_outside_target = (distance - np.array([[tp["target_size"]]], dtype=np.float32)).T
    success_count = 0
    acquisition_time_list = []
    for each_trial in if_outside_target:
        reached = np.argwhere(each_trial<=0)
        if len(reached) > 0:
            success_count += 1
            acquisition_time_list.append(reached[0, 0]*tp["delta_t"])
    success_rate = success_count/if_outside_target.shape[0]
    if len(acquisition_time_list) == 0:
        mean_acquisition_time = 0
    else:
        mean_acquisition_time = np.mean(acquisition_time_list)
    return success_rate, mean_acquisition_time


def perm_mat(dim, rng):
    perm_array = rng.permutation(dim)
    return gene_mat(perm_array)


def gene_mat(perm_array):
    dim = len(perm_array)
    mat = np.zeros([dim, dim])
    for i in range(dim):
        mat[i, perm_array[i]] = 1
    return mat


def z_score(data):
    mean = np.nanmean(data, axis=0)
    sigma = np.sqrt(np.nanmean((data - mean)**2, axis=0))
    return mean, sigma


def canonical_angle(L1, L2, if_print=True, result="degree"):
    '''
    input: L1 and L2 are the loading matrix of two latent spaces, dim_L1 <= dim L2
    output: canonical angle between these two spaces
    '''
    if len(L1.shape)==1:
        L1 = np.expand_dims(L1, axis=0)
    if len(L2.shape)==1:
        L2 = np.expand_dims(L2, axis=0)
    if len(L1) > len(L2):
        L1, L2 = L2, L1
    orthn_L1 = np.linalg.qr(L1.T)[0].T
    orthn_L2 = np.linalg.qr(L2.T)[0].T
    M = orthn_L1  @ orthn_L2.T
    cos_theta = np.sqrt(np.linalg.det(M @ M.T))
    if cos_theta > 1:
        theta = 0
    else:
        theta = np.arccos(cos_theta)
    if if_print:
        print("theta = {0:.2f}".format(theta/np.pi*180)+'\N{DEGREE SIGN}')
    if result == "radian":
        return theta
    elif result == "degree":
        return theta * 180 / np.pi


def save_message(job_id, message, path='save/batch_job_message.txt'):
    with open(path, 'a+') as file:
        file.write("{}: {}".format(job_id, message))


def set_params(argv, params):
    if argv:
        assert len(argv)%2 == 0
    else:
        return params
    para_dict = [(argv[i], argv[i+1]) for i in range(len(argv)//2)]
    for para, value in para_dict:
        if value == "True":
            value = True
        elif value == "False":
            value = False
        elif '.' in value:
            value = float(value)
        elif value.isdigit():
            value = int(value)
        params[para] = value
    return params


def vector_to_angle(v):
    v = v.T
    angle = np.arccos(v[0]/np.linalg.norm(v,axis=0))/np.pi*180*np.where(v[1]<0, -1, 1)+ np.where(v[1]<0, 360, 0)
    return angle.T


def angle_diff(angle1, angle2, if_abs=True):
    angle_diff = angle2 - angle1
    angle_diff -= np.where(angle_diff>180, 360, 0)
    angle_diff += np.where(angle_diff<-180, 360, 0)
    return np.abs(angle_diff) if if_abs else angle_diff


if __name__ == "__main__":
    pl = np.load("save/within_manifold-p0-prediction_performance_list.npy")
    #pl2 = np.load("save/performance_list.npy")
    #draw_learning_curve(pl2,100,100, name="outside manifold perturbation")
    draw_learning_curve(pl, 200, 200, name="within manifold")
    access_learning(pl, 200, 200)