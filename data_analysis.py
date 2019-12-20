import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis, PCA
from parameter import data_analysis_parameter as dp, model_hyper_parameter as hp
from tools import show_each_row
import time
from model import Model

"""
def show_points(data, save_path=''):
    plt.figure()
    plt.scatter(data.T[0], data.T[1], marker='.', )
    plt.show()


def my_factor_analysis(data, num_factor, rng=None):
    '''
    generate a normal distribution represented by (z, Lambda, Psi) that maximize the log
    likelihood L = ln(P(u|z, Lambda, Psi)) that the data is generated from the distribution
    ln(P(ui|z, Lambda, Psi)) = sum(-1/2*ln(2*pi*Psi^2)) + sum(-1/2*(ui - z @ Lambda)^2/Psi^2)
    
    z.shape: [batch_size, num_factor]
    Lambda.shape: [num_factor, cell_number]
    Psi.shape: [cell_number]
    
    args:
        data: shape of [batch_size, dim_data], the data matrix used for factor analysis
        num_factor: expected latent space dimension
        rng: random number generator
    return:
        Lambda, Psi, z
    '''
    if rng is None:
        rng = np.random.RandomState()
    z = tf.Variable(rng.normal(size=(data.shape[0], num_factor)), dtype=tf.float32)
    z_mean = tf.reduce_mean(z, axis=0, keepdims=True)
    sigma_z = tf.sqrt(tf.reduce_mean((z - z_mean)**2, axis=0, keepdims=True))
    z = (data - z_mean)/sigma_z
    Lambda = tf.Variable(rng.normal(size=(num_factor, data.shape[1])), dtype=tf.float32)
    sigma = np.sqrt(np.mean(data**2, axis=0) - np.mean(data, axis=0)**2)
    Psi = tf.Variable(sigma, dtype=tf.float32)
    Psi = tf.nn.relu(Psi) + 1e-6
#    data = z_score(data, "numpy")
    u = tf.constant(data, dtype=tf.float32)
    loss = tf.reduce_mean(2*tf.log(Psi)) + tf.reduce_mean((u-z@Lambda)**2*tf.expand_dims(Psi**(-2), 0))
    optimize = tf.train.AdamOptimizer(0.01).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            _step, _loss, _Lambda, _Psi, _z = sess.run([optimize, loss, Lambda, Psi, z])
            if (i+1)%100 == 0:
                print("step: {0}, loss: {1:.3f}".format(i+1, _loss))
    show_points(data)
    show_points(_z@_Lambda)
    return _Lambda, _Psi, _z


def generate_test_data():
    z = np.random.normal(size=(200, 2))
    Lambda = np.array([[1,0,0],[0,1,0]])
    noise = np.random.normal(scale=0.1, size=(200, 3))
    data = z @ Lambda + noise
    return data, Lambda, z
"""

def data_analysis(activity, num_factor, output_weight=None, save_path=''):
    start_time = time.time()
    print("data size: {}".format(activity.shape))
    print("start "+dp["method"]+", {} factors".format(num_factor))
    _activity_mean = np.mean(activity, axis=0)
    if dp["method"] == "FA":
        method = FactorAnalysis
    elif dp["method"] == "PCA":
        method = PCA
    transformer = method(n_components=num_factor, random_state=0)
    save_path = save_path+dp["method"]+"_results_"+str(num_factor)+'_components'
    _z = transformer.fit_transform(activity)
    _Lambda = transformer.components_
    if num_factor == dp["optimal_num_factor"]:
        show_each_row(_Lambda, save_path)
        _Psi = transformer.noise_variance_
        _cov = transformer.get_covariance()
        np.save(save_path, np.array([_z, _Lambda, _Psi, _activity_mean, _cov]))
    if output_weight is not None:
        log_likelihood = transformer.score(activity)
        est_relative_error = np.mean(np.linalg.norm(activity-_z@_Lambda-_activity_mean, axis=-1)/np.linalg.norm(activity, axis=-1))
        est_out_error = np.mean(np.linalg.norm((activity-_z@_Lambda-_activity_mean)@output_weight, axis=-1))
        est_relative_out_error = np.mean(np.linalg.norm((activity-_z@_Lambda-_activity_mean)@output_weight, axis=-1)/np.linalg.norm(activity@output_weight, axis=-1))
        print("num_factor: {0}, average LL: {1:.2f}, time: {2:.1f}s".format(num_factor, log_likelihood, time.time()-start_time))
        return [log_likelihood, est_relative_error, est_out_error, est_relative_out_error]


def DA_on_range(firing_rate_data, output_weight, load_path):
    num_factor_list = np.array(range(1, 20, 1))
    performance = np.array([data_analysis(firing_rate_data, num_factor, output_weight, load_path) for num_factor in num_factor_list])
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    ax1.plot(num_factor_list, performance[:,0])
    ax2.plot(num_factor_list, performance[:,1])
    ax3.plot(num_factor_list, performance[:,2])
    ax4.plot(num_factor_list, performance[:,3])
    ax3.set_xlabel("factor number")
    ax1.set_ylabel("average LL")
    ax2.set_ylabel("relative activity estimation error")
    ax3.set_ylabel("output estimation error")
    ax4.set_ylabel("relative output estimation error")
    plt.savefig(load_path+dp["method"]+"_LL_plot.png", dpi=1000)


def main(load_path):
    load_path = "save/"+load_path+'/' if load_path else "save/"
    firing_rate_data = np.load(dp["load_path"]+dp["activity_filename"])[:200,:,:dp["cell_number"]]
    firing_rate_data = np.reshape(firing_rate_data, [-1, firing_rate_data.shape[-1]])
    
    model = Model(hp)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_model(sess, load_path+"checkpoint.ckpt")
        output_weight = sess.run(model.weight_list[0][2])
        if dp["Dale's law"]:
            base_vector = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
            output_weight = output_weight @ base_vector
    
    data_analysis(firing_rate_data, dp["optimal_num_factor"], save_path=load_path)
    #DA_on_range(firing_rate_data, output_weight, load_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        main("")
        #main("-b50-Dy-s01-in")
    else:
        main(sys.argv[1])
    '''
    data, L, z = generate_test_data()
    transformer = FactorAnalysis(n_components=2)
    _z = transformer.fit_transform(data)
    components = transformer.components_
    noise = transformer.noise_variance_
    error_var = np.sqrt(np.mean((data - _z @ components)**2, axis=0))
    print("error:", error_var)
    print(transformer.score(data))
    print("actual factor:", L)
    print("FA components:", components)
    print("FA noise:", noise)
    print(transformer.get_covariance())
    show_points(z)
    show_points(_z)
    '''

