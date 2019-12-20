import tensorflow as tf
import numpy as np
from model import Model
from tools import show_activity, show_cursor_trail, generate_target, simple_plot,\
                     assess_performance, set_params, show_distribution, vector_to_angle, angle_diff
import time


def test(model, tp, message=''):
    print("building computational graph")
    model.run(int(tp["trial_time"]/tp["delta_t"]))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt_name = "prediction_checkpoint.ckpt" if tp["predict_cursor_location"] else "checkpoint.ckpt"
        if tp["use_BCI_readout"]:
            if tp["test_before_training"]:
                model.load_model(sess, tp["load_path"]+str(tp["perturbation_type"])+"_perturbation_before_training_"+ckpt_name)
            else:
                model.load_model(sess, tp["load_path"]+str(tp["perturbation_type"])+"_perturbation_"+ckpt_name)
        else:
            if tp["test_washout"]:
                intuitive_BCI_readout = np.load(tp["load_path"]+"W_para.npy")
                model.load_model(sess, tp["load_path"]+str(tp["perturbation_type"])+"_perturbation_"+ckpt_name)
                sess.run(tf.assign(model.network[0].BCI_readout, tf.constant(intuitive_BCI_readout)))
            else:
                model.load_model(sess, tp["load_path"]+ckpt_name)
        print("start testing")
        start_time = time.time()
        rng = np.random.RandomState(model.hp["seed"] + 1)
        target_location = generate_target(tp, rng, None)
        _activity_list, _cursor_trail, _weight_list = sess.run([model.activity_list, model.get_cursor_trail, model.weight_list],
                                                feed_dict={model.target_location:target_location})
        if tp["predict_cursor_location"]:
            _p_cursor_trail = sess.run(model.get_predicted_cursor_trail,
                                                feed_dict={model.target_location:target_location})
        test_save_path = tp["load_path"]+message+'-'
        _activity = np.array(_activity_list[0])
        
        if tp["no_bias_test"]:
            _activity = np.reshape(_activity, [_activity.shape[0], tp["number_of_location"], int(_activity.shape[1]/tp["number_of_location"]), -1])
            show_activity(_activity[:,:,0,:], 1, test_save_path)
            max_activity = np.amax(np.mean(_activity, axis=(0,2)), axis=0)
            W_out = np.array(_weight_list[0][3] if tp["predict_cursor_location"] else _weight_list[0][2])
            o = np.mean((_activity@W_out)[25:125], axis=(0,2))
            init_output_angle = vector_to_angle(o)
            target_angle = np.arange(0,360,360/len(init_output_angle))
            simple_plot(target_angle, angle_diff(target_angle, init_output_angle), init_output_angle)
            simple_plot(o[:,0], o[:,1])
        else:
            show_activity(_activity, 4, test_save_path)
        max_activity = np.amax(np.mean(_activity, axis=0), axis=0)
        show_distribution(max_activity)
        cutting_threshold = np.mean(max_activity)/5
        activated_neurons = np.sum(np.where(max_activity>cutting_threshold, 1, 0))
        print("total neuron number: {0}, activated neuron number(at least at one condition): {1}, cutting threshold: {2}".format(max_activity.shape[-1], activated_neurons, cutting_threshold))
        if tp["predict_cursor_location"]:
            show_cursor_trail(_p_cursor_trail, 10, target_location=target_location, target_size=tp["target_size"], save_path=test_save_path)
        show_cursor_trail(_cursor_trail, 10, target_location=target_location, target_size=tp["target_size"], save_path=test_save_path)
    if tp["save_activity"]:
        if tp["use_BCI_readout"]:
            suffix = "_pb" if tp["test_before_training"] else "_p"
        else:
            suffix = "_w" if tp["test_washout"] else ""
        np.save(test_save_path+"activity"+suffix, _activity)
    success_rate, mean_acquisition_time = assess_performance(_cursor_trail, target_location, tp)
    with open(tp["load_path"]+'test-results.txt', 'a+') as results_file:
        results_file.write(message+'\n')
        result_line = "success rate: {0:.1f}%, mean acquisition time: {1:.0f}ms\n".format(success_rate*100, mean_acquisition_time)
        results_file.write(result_line)
    print(result_line)
    print("test time: {0:.2f}s".format(time.time()-start_time))


def test_on_one_param(tp, hp, param='', value=None):
    message = "default_test"
    if param:
        tp[param] = value
        if param in hp:
            hp[param] = value
        message = param+'-'+str(value)
    model = Model(hp)
    test(model, tp, message)


def main(argv=[], hp=None, tp=None):
    if hp is None:
        from parameter import model_hyper_parameter as hp
    if tp is None:
        from parameter import testing_parameter as tp
    tp = set_params(argv, tp)
    
    
    if tp["no_noise"]:
        hp["input_noise_std"], hp["output_noise_std"] = 0, 0
        hp["recurrent_noise_std"] = 0
        tp["batch_size"] = tp["number_of_location"]
    hp["delta_t"] =  tp["delta_t"]
    hp["batch_size"] = tp["batch_size"]
    hp["number_of_location"] = tp["number_of_location"]
    hp["moving_target"] = tp["moving_target"]
    hp["add_perturbation_to_cursor"] = tp["add_perturbation_to_cursor"]
    hp["trial_time"] = tp["trial_time"]
    hp["firing_rate_scale"] = tp["firing_rate_scale"]
    hp["seed"] += 20
    hp["output_bias"] = tp["output_bias"]
    hp["use_BCI_readout"] = tp["use_BCI_readout"]
    hp["predict_cursor_location"] = tp["predict_cursor_location"]
    hp["predict_velocity"] = tp["predict_velocity"]
    with open(tp["load_path"]+'test-params.txt', 'w+') as params_file:
        params_file.write("hp:\n")
        params_file.write(str(hp))
        params_file.write("\ntp:\n")
        params_file.write(str(tp))
    
    test_on_one_param(tp, hp)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])