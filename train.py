from model import Model
import tensorflow as tf
import numpy as np
import time
from tools import show_cursor_trail, show_loss, show_connection, set_params, access_learning,\
                 show_activity, generate_target, show_weight_norm, assess_performance, draw_learning_curve
import sys, os


def compute_error(cursor_trail, target_location, tp):
    '''
    use cursor trail record and target location to compute error
    tp: training parameter
    two avaliable methods:
        1. "delayed_location": based on distance between cursor and target after a target reaching time
        2. "desired_velocity": based on L2 norm square of difference between actual velocity and desired velocity at all time steps
    return:
        error
    '''
    if not tp["moving_target"]:
        target_location = tf.expand_dims(target_location, axis=0)
    
    if tp["error_type"] == "delayed_location":
        # error at each time step is the batch mean of relu(cursor_target_distance - target_size)
        
        distance = tf.reduce_sum((cursor_trail - target_location)**2, axis=2)
        error = tf.reduce_mean(tf.nn.relu(distance - tf.constant([[tp["target_size"]]], dtype=tf.float32)), axis=1, keepdims=True)
        
        # only count error at time later than target_reaching_time
        mask = tf.constant(np.diag([0 if idx < tp["target_reaching_time"]/tp["delta_t"] else 1 for idx in range(cursor_trail.shape[0])]), dtype=tf.float32)
        error = tf.reduce_sum(mask @ error)
        return error
    if tp["error_type"] == "delayed_location_2":
        # error at each time step is the batch mean of (cursor_target_distance)**2
        distance = tf.reduce_sum((cursor_trail - target_location)**2, axis=2)
        error = tf.reduce_mean(distance, axis=1)
        
        # only count error at time later than target_reaching_time
        #mask = tf.constant(np.diag([0 if idx < tp["target_reaching_time"]/tp["delta_t"] else 1 for idx in range(cursor_trail.shape[0])]), dtype=tf.float32)
        error = tf.reduce_mean(error[int(tp["target_reaching_time"]/tp["delta_t"]):])
        return error
    elif tp["error_type"] == "desired_velocity":
        beta = tp["max_velocity"]/tp["initial_target_distance"]
        desired_velocity = beta * (cursor_trail - target_location)
        time_step_num = cursor_trail.shape[0]
        differential_matrix = -np.eye(time_step_num) + np.concatenate((np.zeros((time_step_num, 1)), np.concatenate((np.eye(time_step_num-1), np.zeros((1, time_step_num-1))))), axis=1)
        actual_velocity = tf.reshape(differential_matrix @ tf.reshape(cursor_trail, [time_step_num, -1]), cursor_trail.shape)/tp["delta_t"]
        delta_v = tf.reduce_sum(tf.square(actual_velocity - desired_velocity), axis=2)
        '''
        # use mask1 to mask delta_v when cursor is inside the target
        distance = tf.sqrt(tf.reduce_sum(tf.square(cursor_trail - tf.expand_dims(target_location, axis=0)), axis=2))
        mask1 = tf.sign(tf.nn.relu(distance - tf.constant([[tp["target_size"]]], dtype=tf.float32)))
        delta_v = mask1 * delta_v
        '''
        error = tf.reduce_mean(delta_v, axis=1, keepdims=True)
        # use mask to remove the last time step
        mask = tf.constant(np.diag(np.concatenate((np.ones(time_step_num-1), [0]))), dtype=tf.float32)
        error = tf.reduce_sum(mask @ error)
        return error
    else:
        raise(RuntimeError("error computing method not defined"))


def validation(tp, sess, error, model, rng):
    validation_error_list = [sess.run(error, feed_dict={model.target_location:generate_target(tp, rng, i), model.close_noise:True}) for i in range(tp["number_of_location"])]
    return np.mean(validation_error_list)


def train(model, tp):
    '''
    train the given model with training parameters tp
    '''
    print("building computational graph")
    step_num = int(tp["trial_time"]/tp["delta_t"])
    model.run(step_num)
    if not tp["predict_cursor_location"]:
        target_error = compute_error(model.get_cursor_trail, model.target_location, tp)
        error = target_error
    else:
        if tp["assisted_prediction_training"]:
            model.change_saver("M1_output_weight:0")
        prediction_error = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum((model.get_predicted_cursor_trail-model.get_cursor_trail)**2, axis=2), axis=1))
        target_error = compute_error(model.get_predicted_cursor_trail, model.target_location, tp)
        error = target_error + tp["lambda_pred"]*prediction_error
    alignment_error = model.alignment_error/step_num
    error = error + tp["lambda_align"]*alignment_error
    mean_recurrent_weight = tf.reduce_mean([tf.reduce_mean(tf.math.abs(layer_weight[1])) for layer_weight in model.weight_list])
    mean_activity = tf.reduce_mean([tf.reduce_mean(layer_activity) for layer_activity in model.activity_list])
    #lambda_fr = tf.cond(error > 20, lambda: 0.0, lambda: tp["lambda_fr"])
    loss = error + tp["lambda_rec"]*mean_recurrent_weight + tp["lambda_fr"]*mean_activity
    learning_rate = tf.cond(loss > tp["learning_rate_downscale_threshold"], lambda: tp["learning_rate"], lambda: tp["learning_rate"]/5)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.875)
    if not tp["seperate_training"]:
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, global_norm = tf.clip_by_global_norm(gradients, tp["maximum_gradient_norm"])
        optimize = optimizer.apply_gradients(zip(gradients, variables))
    else:
        gradients1, variables = zip(*optimizer.compute_gradients(tp["lambda_pred"]*prediction_error))
        gradients2, variables = zip(*optimizer.compute_gradients(target_error + tp["lambda_align"]*alignment_error))
        gradients1, global_norm1 = tf.clip_by_global_norm(gradients1, tp["maximum_gradient_norm"])
        gradients2, global_norm2 = tf.clip_by_global_norm(gradients2, tp["maximum_gradient_norm"])
        optimize = optimizer.apply_gradients(zip((gradients2[0],gradients1[1],gradients2[2],gradients2[3]),variables))
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    print("working directory: "+tp["load_path"])
    with tf.Session(config=config) as sess:
        sess.run(init)
        ckpt_name = "prediction_checkpoint.ckpt" if tp["predict_cursor_location"] else "checkpoint.ckpt"
        if tp["use_BCI_readout"]:
            model.load_model(sess, tp["load_path"]+ckpt_name)
            #sess.run(tf.assign(model.network[0].output_weight, model.network[0].intuitive_pred_readout))
        if tp["assisted_prediction_training"]:
            model.change_saver("M1_output_weight:0", model.network[0].output_weight)
        model.update_weight(sess)
        if tp["use_BCI_readout"]:
            model.save_model(sess, tp["save_path"]+str(tp["perturbation_type"])+"_perturbation_before_training_"+ckpt_name)
        if not tp["baseline_block"] and not tp["washout_block"]:
            #sess.graph.finalize()
            pass
        
        rng = np.random.RandomState(tp["seed"])
        performance_list = []
        weight_record = []
        activity_record = []
        if tp["baseline_block"]:
            print("\nstart baseline block\n")
            assign_BCI = model.network[0].assign_BCI_readout
            model.network[0].assign_BCI_readout = tf.assign(model.network[0].BCI_readout, model.network[0].intuitive_BCI_readout)
            model.update_weight(sess)
            for i in range(tp["baseline_trial"]):
                target_location = generate_target(tp, rng)
                _train_step, _cursor_trail = sess.run([optimize, model.get_cursor_trail],\
                                                      feed_dict={model.target_location:target_location})
                model.update_weight(sess)
                if tp["save_record"]:
                    _activity_list, _weight_list = sess.run([model.activity_list, model.weight_list],\
                                                            feed_dict={model.target_location:target_location})
                    _weight_list = np.array(_weight_list[0])
                    weight_record.append(_weight_list)
                    activity_record.append(_activity_list[0])
                _sr, _at = assess_performance(_cursor_trail, target_location, tp)
                performance_list.append([_sr, _at])
            model.network[0].assign_BCI_readout = assign_BCI
            model.update_weight(sess)
        
        print("start training")
        start_time = time.time()
        loss_list = []
        prediction_error_list = []
        target_error_list = []
        alignment_error_list = []
        validation_error_list = []
        _init_weight_list = np.array(sess.run(model.weight_list)[0])
        if tp["use_BCI_readout"]:
            W_BCI_im = tp["L"].T@tp["L"]@_init_weight_list[-1]
            W_BCI_om = _init_weight_list[-1] - W_BCI_im
            
        _last_weight_list = _init_weight_list
        weight_norm_list = [[],[],[],[],[],[],[],[]]
        
        target_location = generate_target(tp, rng, 6)
        print("trainable variables:")
        print([variables[i].name for i in range(len(variables))])
        init_loss, iae, ma, init_trail = sess.run([loss, alignment_error, mean_activity, model.get_cursor_trail], feed_dict={model.target_location:target_location})
        loss_list.append(init_loss)
        alignment_error_list.append(iae)
        '''
        for i in range(len(_gradients)):
            print("_gradient {}: {}".format(i, _gradients[i]))
        '''
        if tp["predict_cursor_location"]:
            _p_error, _p_cursor_trail = sess.run([prediction_error, model.get_predicted_cursor_trail], feed_dict={model.target_location:target_location})
            show_cursor_trail(_p_cursor_trail, 10, target_location, tp["target_size"])
            prediction_error_list.append(_p_error)
            print("initial prediction error:{0:.2f}".format(_p_error))
        show_cursor_trail(init_trail, 10, target_location, tp["target_size"])
        print("initial loss: {0:.2f}, initial alignment error: {1:.2f}, mean activity: {2:.2f}".format(init_loss, iae, ma))
        validation_error = validation(tp, sess, error, model, rng)
        validation_error_list.append(validation_error)
        print("initial validation error:{0:.3f}\n".format(validation_error))
        learning_step = tp["training_step"]
        for i in range(tp["training_step"]):
            target_location = generate_target(tp, rng)
            if tp["predict_cursor_location"]:
                _p_error = sess.run(prediction_error, feed_dict={model.target_location:target_location})
                prediction_error_list.append(_p_error)
            
            _loss, _te, _ae, _mean_recurrent_weight, _mean_activity, _train_step =\
                sess.run([loss, target_error, alignment_error, mean_recurrent_weight, mean_activity, optimize],\
                          feed_dict={model.target_location:target_location})
            model.update_weight(sess)
            loss_list.append(_loss)
            target_error_list.append(_te)
            alignment_error_list.append(_ae)
            
            
            if tp["save_record"]:
                _activity_list, _weight_list = sess.run([model.activity_list, model.weight_list],\
                          feed_dict={model.target_location:target_location})
                _weight_list = np.array(_weight_list[0])
                weight_record.append(_weight_list)
                activity_record.append(_activity_list[0])
                delta_w = _weight_list - _last_weight_list
                _last_weight_list = _weight_list
                for w in range(len(_weight_list)):
                    weight_norm_list[w].append(np.linalg.norm(_weight_list[w]-_init_weight_list[w]))
                    weight_norm_list[w+len(_weight_list)].append(np.linalg.norm(delta_w[w]))
            
            '''
            if tp["use_BCI_readout"]:
                weight_diff = _weight_list[2]-_weight_list[3]
                output_im = tp["L"].T@tp["L"]@_weight_list[2]
                output_om = _weight_list[2] - output_im
                def angle(v1, v2):
                    if np.linalg.norm(v1)<1e-2 or np.linalg.norm(v2)<1e-2:
                        return 90.0
                    else:
                        return np.arccos(v1@v2/np.linalg.norm(v1)/np.linalg.norm(v2))/np.pi*180
                angle_diff_list += [[angle(output_im.T[0], W_BCI_im.T[0]), angle(output_im.T[1], W_BCI_im.T[1]),\
                                 angle(output_om.T[0], W_BCI_om.T[0]), angle(output_om.T[1], W_BCI_om.T[1]),\
                                 angle(tp["L"].T@tp["L"]@(_weight_list[2].T[0]), _weight_list[2].T[0]), angle(tp["L"].T@tp["L"]@(_weight_list[2].T[1]), _weight_list[2].T[1])]]
                weight_diff_list += [[np.linalg.norm(tp["L"].T@tp["L"]@weight_diff), np.linalg.norm(output_om-W_BCI_om), np.linalg.norm(output_om)]]
                
                _activity_list = sess.run(model.activity_list, feed_dict={model.target_location:target_location})
                activity_on_manifold = np.linalg.norm(np.array(_activity_list[0])@tp["L"].T@tp["L"], axis=1)
                activity_angle = np.mean(np.arccos(activity_on_manifold/np.linalg.norm(np.array(_activity_list[0]), axis=1))/np.pi*180)
                activity_angle_list += [activity_angle]
            '''
            
            if tp["validate"]:
                validation_error = validation(tp, sess, error, model, rng)
                validation_error_list.append(validation_error)
                if validation_error < tp["validation_threshold"]:
                    learning_step = i+1
                    break
            if tp["assess_performance"]:
                _cursor_trail = sess.run(model.get_cursor_trail, feed_dict={model.target_location:target_location})
                _sr, _at = assess_performance(_cursor_trail, target_location, tp)
                performance_list.append([_sr, _at])
            if (i+1) % 20 == 0:
                print("step: {0}, training time: {1:.2f}s, mean_recurrent_weight: {2:.3f}, mean_activity: {3:.3f}".format(i+1, time.time()-start_time, _mean_recurrent_weight, _mean_activity))
                if tp["predict_cursor_location"]:
                    print("prediction error:{0:.2f}".format(_p_error))
                print("loss: {0:.3f}, target error: {1:.3f}, alignment error: {2:.3f}\n".format(_loss, _te, _ae))
                if tp["validate"]:
                    print("validation error:{0:.3f}\n".format(validation_error))
        target_location = generate_target(tp, rng, 6)
        _cursor_trail, _activity_list, _weight_list = sess.run([model.get_cursor_trail, model.activity_list, model.weight_list],\
                          feed_dict={model.target_location:target_location})
        if tp["predict_cursor_location"]:
            _p_cursor_trail = sess.run(model.get_predicted_cursor_trail, feed_dict={model.target_location:target_location})
            show_cursor_trail(_p_cursor_trail, 10, target_location, tp["target_size"], save_path=tp["save_path"]+'predicted_')
        show_cursor_trail(_cursor_trail, 10, target_location, tp["target_size"], save_path=tp["save_path"])
        _weight_list = np.array(_weight_list[0])
        if tp["predict_cursor_location"]:
            show_loss(target_error_list, prediction_error_list, alignment_error_list, loss_list, loss_threshold=tp["learning_rate_downscale_threshold"], save_path=tp["save_path"])
        else:
            show_loss(loss_list, validation_error_list if tp["validate"] else None, loss_threshold=tp["learning_rate_downscale_threshold"], save_path=tp["save_path"])
        if tp["save_record"]:
            np.save(tp["save_path"]+tp["perturbation_type"]+"_weight_record", weight_record)
            np.save(tp["save_path"]+tp["perturbation_type"]+"_activity_record", activity_record)
            show_weight_norm(weight_norm_list, save_path=tp["save_path"]+(str(tp["perturbation_type"])+'_' if tp["use_BCI_readout"] else "original_"))
        show_connection(_weight_list, save_path=tp["save_path"])
        _activity = np.array(_activity_list[0])
        show_activity(_activity, 1, save_path=tp["save_path"])
        if tp["save"]:
            if tp["use_BCI_readout"]:
                model.save_model(sess, tp["save_path"]+str(tp["perturbation_type"])+"_perturbation_"+ckpt_name)
            else:
                model.save_model(sess, tp["save_path"]+ckpt_name)
        
        if tp["washout_block"]:
            print("\nstart washout block\n")
            model.network[0].assign_BCI_readout = tf.assign(model.network[0].BCI_readout, model.network[0].intuitive_BCI_readout)
            model.update_weight(sess)
            for i in range(tp["washout_trial"]):
                target_location = generate_target(tp, rng)
                _train_step, _cursor_trail = sess.run([optimize, model.get_cursor_trail],\
                                                      feed_dict={model.target_location:target_location})
                model.update_weight(sess)
                if tp["save_record"]:
                    _activity_list, _weight_list = sess.run([model.activity_list, model.weight_list],\
                                                            feed_dict={model.target_location:target_location})
                    _weight_list = np.array(_weight_list[0])
                    weight_record.append(_weight_list)
                    activity_record.append(_activity_list[0])
                _sr, _at = assess_performance(_cursor_trail, target_location, tp)
                performance_list.append([_sr, _at])
        
        if tp["use_BCI_readout"]:
            if tp["predict_cursor_location"]:
                np.save(tp["save_path"]+"{}-p{}-prediction_performance_list".format(str(tp["perturbation_type"]), tp["pick_perturbation"]), performance_list)
            else:
                np.save(tp["save_path"]+"{}-p{}-performance_list".format(str(tp["perturbation_type"]), tp["pick_perturbation"]), performance_list)
        else:
            if tp["predict_cursor_location"]:
                np.save(tp["save_path"]+"prediction_performance_list", performance_list)
            else:
                np.save(tp["save_path"]+"performance_list", performance_list)
        
        draw_learning_curve(performance_list, tp["baseline_trial"] if tp["baseline_block"] else 0, tp["washout_trial"] if tp["washout_block"] else 0,\
                            tp["trial_time"], name=str(tp["perturbation_type"]) if tp["use_BCI_readout"] else "train from scratch", save_path=tp["save_path"])
        if tp["use_BCI_readout"] and tp["baseline_block"] and tp["washout_block"]:
            learning_effect = access_learning(performance_list, tp["baseline_trial"], tp["washout_trial"])
            print("amount_of_learning:{}, learning_time:{}, after_effect:{}, relative_after_effect:{}".format(*learning_effect))
            np.save(tp["save_path"]+"learning_effect", learning_effect)
        
        return learning_step, performance_list


def train_on_perturbation(model, tp):
    W_para = np.load(tp["load_path"]+"W_para.npy")
    L = np.load(tp["load_path"]+"PCA_results_10_components.npy", allow_pickle=True)[1]
    tp["L"] = L
    rng = np.random.RandomState(tp["seed"]-1)
    if tp["perturbation_type"] == None:
        W_BCI_out = W_para
    elif tp["perturbation_type"] == "within_manifold":
        W_pert_list = np.load(tp["load_path"]+"W_pert_list.npy", allow_pickle=True)
        choice = tp["pick_perturbation"]
        print("using within manifold perturbed readout No.{} from {} candidates".format(choice+1, len(W_pert_list)))
        W_BCI_out = W_pert_list[choice]
    elif tp["perturbation_type"] == "outside_manifold":
        W_pert_list = np.load(tp["load_path"]+"W_out_pert_list.npy", allow_pickle=True)
        choice = tp["pick_perturbation"]
        print("using outside manifold perturbed readout No.{} from {} candidates".format(choice+1, len(W_pert_list)))
        W_BCI_out = W_pert_list[choice]
    elif tp["perturbation_type"] == "bias":
        W_out_bias = rng.rand(*W_para.shape)
        W_out_bias = W_out_bias/np.linalg.norm(W_out_bias)*np.linalg.norm(W_para)
        W_BCI_out = W_para + W_out_bias*1.0
    print("|W_para|={0:.2f}, |W_BCI_out|={1:.2f}, |W_para-W_BCI_out|={2:.2f}".format(np.linalg.norm(W_para), np.linalg.norm(W_BCI_out), np.linalg.norm(W_para-W_BCI_out)))
    model.network[0].assign_BCI_readout = tf.assign(model.network[0].BCI_readout, tf.constant(W_BCI_out, dtype=tf.float32))
    model.network[0].intuitive_BCI_readout = tf.constant(W_para, dtype=tf.float32)
    model.network[0].intuitive_pred_readout = tf.constant(np.load(tp["load_path"]+"W_para_pred.npy"), dtype=tf.float32)
    learning_step, pl = train(model, tp)
    if tp["use_BCI_readout"] and tp["validate"] and tp["pick_perturbation"] is not None:
        with open("training_speed.txt", 'a') as file:
            file.write(str([choice, tp["seed"], learning_step])+',')
    return pl


def main(argv=[], hp=None, tp=None):
    if hp is None:
        from parameter import model_hyper_parameter as hp
    if tp is None:
        from parameter import training_parameter as tp
    tp = set_params(argv, tp)


    if tp["assisted_prediction_training"]:
        tp["use_BCI_readout"] = True
        tp["perturbation_type"] = None
    if not os.path.exists(tp["save_path"][:-1]):
        os.mkdir(tp["save_path"][:-1])
    hp["delta_t"] = tp["delta_t"]
    hp["batch_size"] = tp["batch_size"]
    hp["number_of_location"] = tp["number_of_location"]
    hp["firing_rate_scale"] = tp["firing_rate_scale"]
    hp["moving_target"] = tp["moving_target"]
    hp["output_bias"] = tp["output_bias"]
    hp["use_BCI_readout"] = tp["use_BCI_readout"]
    if not tp["use_BCI_readout"]:
        tp["baseline_block"], tp["washout_block"] = False, False
        hp["gradient_through_cursor_loc"], hp["readout_trainable"], tp["seperate_training"] = True, True, False
    hp["predict_cursor_location"] = tp["predict_cursor_location"]
    hp["predict_velocity"] = tp["predict_velocity"]
    with open(tp["save_path"]+'params.txt', 'w+') as params_file:
        params_file.write("hp:\n")
        params_file.write(str(hp))
        params_file.write("\ntp:\n")
        params_file.write(str(tp))
    model = Model(hp)
    if not tp["use_BCI_readout"]:
        _ls, pl = train(model, tp)
    else:
        pl = train_on_perturbation(model, tp)
    return pl


if __name__ == "__main__":
    main(sys.argv[1:])