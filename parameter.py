model_hyper_parameter = {"batch_size":1,
                         "number_of_location":8,
                         "delta_t":5.0, # ms
                         "cell_number":500,
                         "input_channel":6,
                         "output_channel":2,
                         "tao":50.0, # ms
                         "activation_function":"relu",
                         "recurrent_noise_std":0.2,
                         "input_noise_std":0.0,
                         "output_noise_std":0.0,
                         "firing_rate_scale":1.0,
                         "input_weight_init_scale":1.0,
                         "output_weight_init_scale":0.1,
                         "recurrent_weight_initial_scale":5,
                         "model":"only_M1",
                         "predict_cursor_location":False,
                         "predict_velocity":False,
                         "seed":1909011200,
                         "Dale's law":False,
                         "input Dale's law":False,
                         "record_state":True,
                         "input_lag":125.0, # ms
                         "output_lag":0.0, # ms
                         "BCI_output_lag":0.0, # ms
                         "moving_target":False,
                         "add_perturbation_to_cursor":False,
                         "update_function":"firing_rate",
                         "output_bias":False,
                         "use_BCI_readout":False,
                         "record_include_lag":True,
                         "gradient_through_cursor_loc":False,
                         "readout_trainable":False,
                         "feedback_prediction":True}

training_parameter = {"error_type":"delayed_location_2",
                      "delta_t":5.0,
                      "target_reaching_time":1000,
                      "target_size":0.1,
                      "max_velocity":1.0,
                      "initial_target_distance":1.0,
                      "batch_size":1,
                      "trial_time":2000.0, # ms
                      "learning_rate":0.0002,
                      "maximum_gradient_norm":1.0,
                      "training_step":500,
                      "lambda_rec":0.0,
                      "lambda_fr":0.0,
                      "lambda_pred":2.0,
                      "lambda_align":0.0,
                      "number_of_location":8,
                      "firing_rate_scale":1.0,
                      "save_path":"save/",
                      "load_path":"save/",
                      "save":True,
                      "seed":1909011300,
                      "validate":False,
                      "assess_performance":True,
                      "learning_rate_downscale_threshold":0.1,
                      "validation_threshold":8.0,
                      "moving_target":False,
                      "output_bias":False,
                      "predict_cursor_location":True,
                      "predict_velocity":True,
                      "assisted_prediction_training":False,
                      "use_BCI_readout":False,
                      "perturbation_type":"within_manifold", # "within_manifold", "outside_manifold", "bias", None
                      "baseline_block":True,    
                      "baseline_trial":200,
                      "washout_block":True,
                      "washout_trial":200,
                      "pick_perturbation":0,
                      "seperate_training":True,
                      "save_record":False}

testing_parameter = {"delta_t":5.0,
                     "target_size":0.2,
                     "initial_target_distance":1.0,
                     "batch_size":40,
                     "trial_time":2000.0, # ms
                     "number_of_location":8,
                     "firing_rate_scale":1.0,
                     "load_path":"save/",
                     "moving_target":False,
                     "moving_velocity":0.5,
                     "add_perturbation_to_cursor":False,
                     "no_bias_test":True,
                     "save_activity":True,
                     "output_bias":False,
                     "use_BCI_readout":False,
                     "test_before_training":False,
                     "test_washout":False,
                     "no_noise":True}
testing_parameter["predict_cursor_location"] = training_parameter["predict_cursor_location"]
testing_parameter["predict_velocity"] = training_parameter["predict_velocity"]
testing_parameter["perturbation_type"] = training_parameter["perturbation_type"]

data_analysis_parameter = {"method":"PCA",
                           "optimal_num_factor":10,
                           "save_figures":True,
                           "load_path":"save/",
                           "activity_filename":"default_test-activity.npy"}
data_analysis_parameter["delta_t"] = testing_parameter["delta_t"]
data_analysis_parameter["time_step"] = int(testing_parameter["trial_time"]/testing_parameter["delta_t"])
data_analysis_parameter["number_of_location"] = testing_parameter["number_of_location"]
data_analysis_parameter["Dale's law"] = model_hyper_parameter["Dale's law"]
data_analysis_parameter["cell_number"] = model_hyper_parameter["cell_number"]