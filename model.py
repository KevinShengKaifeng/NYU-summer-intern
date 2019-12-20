import numpy as np
import tensorflow as tf
from tools import show_activity, show_connection, show_cursor_trail
import time


class RNN_layer(object):
    def __init__(self, hp, init_state, close_noise, layer_name):
        """
        Initializing a RNN layer with information from hp

        Args:
            hp: a dictionary of hyper-parameters
            init_dict: initial value for all variables
        """
        self.hp = hp
        self.init_state = init_state
        self.close_noise = close_noise
        self.layer_name = layer_name
        self._build()


    def _build(self):
        hp = self.hp
        layer_name = self.layer_name
        if hp["input Dale's law"]:
            init_input_weight = tf.random.uniform(shape=(hp["input_channel"], hp["cell_number"]))*hp["input_weight_init_scale"]
        else:
            init_input_weight = tf.random.normal(shape=(hp["input_channel"], hp["cell_number"]), mean=0, stddev=1.0)*hp["input_weight_init_scale"]
        init_output_weight_scale = hp["output_weight_init_scale"]/hp["cell_number"]
        init_output_weight = tf.random.uniform(shape=(hp["cell_number"], hp["output_channel"]))*init_output_weight_scale
        init_BCI_readout_weight = tf.random.uniform(shape=(hp["cell_number"], hp["output_channel"]))*init_output_weight_scale
        if not hp["Dale's law"]:
            init_output_weight = init_output_weight-0.5*init_output_weight_scale
            init_BCI_readout_weight = init_BCI_readout_weight-0.5*init_output_weight_scale
        init_bias = np.zeros((hp["cell_number"]))
        self.input_weight = tf.Variable(init_input_weight, dtype=tf.float32, name=layer_name+'_input_weight')
        self.output_weight = tf.Variable(init_output_weight, dtype=tf.float32, name=layer_name+'_output_weight')
        self.neuron_state = tf.Variable(self.init_state, dtype=tf.float32, trainable=False)
        self.bias = tf.Variable(init_bias, dtype=tf.float32, name=layer_name+'_recurrent_bias')
        self.BCI_readout = tf.Variable(init_BCI_readout_weight, dtype=tf.float32, trainable=True if hp["readout_trainable"] else False, name=layer_name+'_BCI_readout_weight')
        self._activity_record = []
        input_lag_time_step = int(hp["input_lag"]/hp["delta_t"])
        output_lag_time_step = int(hp["output_lag"]/hp["delta_t"])
        BCI_output_lag_time_step = int(hp["BCI_output_lag"]/hp["delta_t"])
        self.input_lag_buffer = [np.zeros((hp["batch_size"], hp["input_channel"])) for i in range(input_lag_time_step + 1)]
        self.output_lag_buffer = [np.zeros((hp["batch_size"], hp["output_channel"])) for i in range(output_lag_time_step + 1)]
        self.BCI_output_lag_buffer = [np.zeros((hp["batch_size"], hp["output_channel"])) for i in range(BCI_output_lag_time_step + 1)]
        if not hp["Dale's law"]:
            init_rec_weight = (tf.random.uniform(shape=(hp["cell_number"], hp["cell_number"]))-0.5)/np.sqrt(hp["cell_number"])*hp["recurrent_weight_initial_scale"]
            #init_rec_weight = tf.random.normal(shape=(hp["cell_number"], hp["cell_number"]), mean=0, stddev=1.0)/np.sqrt(hp["cell_number"])*hp["recurrent_weight_initial_scale"]
            self.recurrent_weight = tf.Variable(init_rec_weight, dtype=tf.float32, name=layer_name+'_recurrent_weight')
        else:
            exc_weight_init = tf.random.gamma(shape=(hp["cell_number"]*4//5, hp["cell_number"]), alpha=2, beta=4)/np.sqrt(hp["cell_number"])*hp["recurrent_weight_initial_scale"]
            inh_weight_init = -tf.random.gamma(shape=(hp["cell_number"]//5, hp["cell_number"]), alpha=2, beta=1)/np.sqrt(hp["cell_number"])*hp["recurrent_weight_initial_scale"]
            rec_weight_init = tf.concat((exc_weight_init, inh_weight_init), 0)
            self.D = tf.constant(np.diag([1.0 if idx < hp["cell_number"]//5*4 else -1.0 for idx in range(hp["cell_number"])]), dtype=tf.float32)
            self.Mask_rec = tf.constant(np.ones((hp["cell_number"], hp["cell_number"]))-np.eye(hp["cell_number"]), dtype=tf.float32)
            self.W_rec_fixed_plus = tf.constant(np.zeros((hp["cell_number"], hp["cell_number"])), dtype=tf.float32)
            self.recurrent_weight = tf.Variable(rec_weight_init, dtype=tf.float32, name=layer_name+'_recurrent_weight')
            self.__assign_rec = tf.assign(self.recurrent_weight, self.D @ (self.Mask_rec * tf.nn.relu(self.D @ self.recurrent_weight) + self.W_rec_fixed_plus))
            self.__assign_out = tf.assign(self.output_weight, tf.nn.relu(self.D @ tf.nn.relu(self.output_weight)))
        self.__assign_in = tf.assign(self.input_weight, tf.nn.relu(self.input_weight))
        self.assign_BCI_readout = None
        self.weight_list = [self.input_weight, self.recurrent_weight, self.output_weight, self.BCI_readout]
        self.save_variable_list = {self.input_weight.name:self.input_weight,
                                   self.output_weight.name:self.output_weight,
                                   self.recurrent_weight.name:self.recurrent_weight,
                                   self.bias.name:self.bias,
                                   self.BCI_readout.name:self.BCI_readout}


    def update_weight(self, sess=None):
        if self.assign_BCI_readout is not None and sess is not None:
            sess.run(self.assign_BCI_readout)
        if self.hp["Dale's law"] and sess is not None:
            sess.run([self.__assign_rec, self.__assign_out])
        if self.hp["input Dale's law"] and sess is not None:
            sess.run(self.__assign_in)


    def run(self, current_input):
        self.input_lag_buffer.append(current_input)
        self.input_lag_buffer = self.input_lag_buffer[1:]
        network_input = self.input_lag_buffer[0]
        
        if self.hp["activation_function"] == "relu":
            activation = tf.nn.relu
        else:
            raise RuntimeError("invalid activation function")
        noise = tf.random.normal(shape=(self.hp["batch_size"], self.hp["cell_number"]), mean=0, stddev=self.hp["recurrent_noise_std"])
        noise = tf.cond(self.close_noise, lambda: tf.zeros((self.hp["batch_size"], self.hp["cell_number"])), lambda: noise)
        alpha = self.hp["delta_t"]/self.hp["tao"]
        if self.hp["update_function"]=="activity":
            self.neuron_state = (1-alpha)*self.neuron_state + alpha*(activation(self.neuron_state) @ self.recurrent_weight
                                 + (network_input @ self.input_weight)*self.hp["firing_rate_scale"] + noise + self.bias)
            output = (activation(self.neuron_state) @ self.output_weight)/self.hp["firing_rate_scale"]
            self.BCI_output = (activation(self.neuron_state) @ self.BCI_readout)/self.hp["firing_rate_scale"]
        elif self.hp["update_function"]=="firing_rate":
            self.neuron_state = (1-alpha)*self.neuron_state + alpha*activation(self.neuron_state @ self.recurrent_weight
                                 + (network_input @ self.input_weight)*self.hp["firing_rate_scale"] + noise + self.bias)
            output = (self.neuron_state @ self.output_weight)/self.hp["firing_rate_scale"]
            self.BCI_output = (self.neuron_state @ self.BCI_readout)/self.hp["firing_rate_scale"]
        
        self.output_lag_buffer.append(output)
        self.output_lag_buffer = self.output_lag_buffer[1:]
        output = self.output_lag_buffer[0]
        self.BCI_output_lag_buffer.append(self.BCI_output)
        self.BCI_output_lag_buffer = self.BCI_output_lag_buffer[1:]
        self.BCI_output = self.BCI_output_lag_buffer[0]
        
        if self.hp["record_state"] and self.hp["update_function"]=="activity":
            self._activity_record.append(activation(self.neuron_state))
        elif self.hp["record_state"] and self.hp["update_function"]=="firing_rate":
            self._activity_record.append(self.neuron_state)
        return output, self.neuron_state
    
    
    def get_activity_record(self):
        if self.hp["record_include_lag"]:
            return tf.stack(self._activity_record)
        elif int(-self.hp["output_lag"]/self.hp["delta_t"]) == 0:
            return tf.stack(self._activity_record[int(self.hp["input_lag"]/self.hp["delta_t"]):])
        else:
            return tf.stack(self._activity_record[int(self.hp["input_lag"]/self.hp["delta_t"]):int(-self.hp["output_lag"]/self.hp["delta_t"])])


class Model(object):
    def __init__(self, hp):
        tf.reset_default_graph()
        tf.set_random_seed(hp['seed'])
        self.hp = hp
        self._build()


    def _build(self):
        hp = self.hp
        self.close_noise = tf.placeholder_with_default(False, shape=[])
        self.cursor_location = tf.Variable(np.zeros((hp["batch_size"], 2)), dtype=tf.float32, trainable=False)
        self.cursor_location_without_gradient = tf.stop_gradient(self.cursor_location)
        self.predicted_velocity = tf.Variable(np.zeros((hp["batch_size"], 2)), dtype=tf.float32, trainable=False)
        if not hp["moving_target"]:
            self.target_location = tf.placeholder(dtype=tf.float32, shape=[hp["batch_size"], 2])
        self.__cursor_trail = []
        self.__predicted_cursor_trail = []
        if hp["predict_cursor_location"]:
            self.predicted_cursor_location = tf.Variable(np.zeros((hp["batch_size"], 2)), dtype=tf.float32, trainable=False)
        if hp["model"]=="only_M1":
            default_init_state = np.zeros((hp["batch_size"], hp["cell_number"]), dtype=np.float32)
            self.init_state = tf.placeholder_with_default(default_init_state, shape=[hp["batch_size"], hp["cell_number"]])
            self.network = [RNN_layer(hp, self.init_state, self.close_noise, 'M1')]
        self.weight_list = [layer.weight_list for layer in self.network]
        self._save_variable_list = {}
        for layer in self.network:
            self._save_variable_list.update(layer.save_variable_list)
        self._saver = tf.train.Saver(self._save_variable_list)
        self.perturb_this_step = False
        #self.output_bias = (tf.random.uniform(shape=(hp["batch_size"], 2))-0.5)*2*hp["delta_t"]/1000
        self.output_bias = tf.constant(np.ones([hp["batch_size"], 2]), dtype=tf.float32)*hp["delta_t"]/1000
        self.alignment_error = tf.constant(0, dtype=tf.float32)
    
    
    def update_weight(self, sess=None):
        for layer in self.network:
            layer.update_weight(sess)
    
    
    def change_saver(self, name, tensor=None):
        if tensor is None:
            self._save_variable_list.pop(name)
        else:
            self._save_variable_list[name] = tensor
        self._saver = tf.train.Saver(self._save_variable_list)
    
    
    def run(self, step=1):
        if not self.hp["moving_target"]:
            self.current_target_location = self.target_location
            for i in range(step):
                if self.hp["add_perturbation_to_cursor"] and i == int(500/self.hp["delta_t"]):
                    self.perturb_this_step = True
                output = self._run_one_step()
        else:
            self.target_location = tf.placeholder(dtype=tf.float32, shape=[step, self.hp["batch_size"], 2])
            unstacked_target_trail = tf.unstack(self.target_location)
            for i in range(step):
                self.current_target_location = unstacked_target_trail[i]
                if self.hp["add_perturbation_to_cursor"] and i == int(500/self.hp["delta_t"]):
                    self.perturb_this_step = True
                output = self._run_one_step()
        self.activity_list = [layer.get_activity_record() for layer in self.network]
        return output
    
    
    def _add_noise(self, noise_std, vector):
        noise = tf.random.normal(vector.shape, 0, noise_std)
        return vector + noise
    
    
    def _transform_location_to_input(self, *argv):
        def __transform(loc):
            _input = tf.nn.relu(loc) @ tf.constant(np.array([[1,0,0,0],[0,1,0,0]]), dtype=tf.float32)\
                    + tf.nn.relu(-loc) @ tf.constant(np.array([[0,0,1,0],[0,0,0,1]]), dtype=tf.float32)
            return _input
        if not self.hp["input Dale's law"]:
            return tf.concat([arg for arg in argv], axis=1)
        else:
            return tf.concat([__transform(arg) for arg in argv], axis=1)
    
    
    def _transform_output_to_velocity(self, output):
        '''
        four outputs represent the desire to move to a specific direction,
        cursor velocity is the sum of each output times the corresponding base vector
        '''
        base_vector = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        cursor_velocity = tf.matmul(output, base_vector)
        return cursor_velocity
    
    
    def _run_one_step(self):
        if self.hp["feedback_prediction"]:
            input_no_noise = self._transform_location_to_input(self.cursor_location_without_gradient, self.current_target_location, self.predicted_velocity)
        else:
            input_no_noise = self._transform_location_to_input(self.cursor_location_without_gradient, self.current_target_location)
        current_input = self._add_noise(self.hp["input_noise_std"], input_no_noise)
        if self.hp["model"]=="only_M1":
            output, current_neuron_state = self.network[0].run(current_input)
            if self.hp["predict_cursor_location"]:
                if not self.hp["predict_velocity"]:
                    self.predicted_cursor_location = output
                else:
                    predicted_velocity = output if not self.hp["Dale's law"] else self._transform_output_to_velocity(output)
                    self.predicted_velocity = predicted_velocity
                output = self.network[0].BCI_output
            elif self.hp["use_BCI_readout"]:
                output = self.network[0].BCI_output
            self.current_neuron_state_list = [current_neuron_state]
        output = self._add_noise(self.hp["output_noise_std"], output)
        cursor_velocity = output if not self.hp["Dale's law"] else self._transform_output_to_velocity(output)
        self.cursor_location = self.cursor_location + self.hp["delta_t"]/1000 * cursor_velocity
        if self.hp["predict_cursor_location"] and self.hp["predict_velocity"]:
            self.predicted_cursor_location = self.predicted_cursor_location + self.hp["delta_t"]/1000 * predicted_velocity
        if self.hp["output_bias"]:
            self.cursor_location = self.cursor_location + self.output_bias
        if self.perturb_this_step:
            perturbation = (tf.random.uniform(shape=(self.hp["batch_size"], 2))-0.5)*2
            self.cursor_location = self.cursor_location + perturbation
            self.perturb_this_step = False
        self.cursor_location_without_gradient = tf.stop_gradient(self.cursor_location)
        alignment_vector = self.current_target_location - self.cursor_location_without_gradient
        vtarget_2 = tf.reduce_sum(predicted_velocity*alignment_vector, axis=1)**2/tf.reduce_sum(alignment_vector**2, axis=1)
        self.alignment_error = self.alignment_error + tf.reduce_mean(tf.reduce_sum(predicted_velocity**2, axis=1) - vtarget_2)
        if not self.hp["predict_cursor_location"]:
            self.__cursor_trail.append(self.cursor_location)
        else:
            if self.hp["gradient_through_cursor_loc"]:
                self.__cursor_trail.append(self.cursor_location)
            else:
                self.__cursor_trail.append(self.cursor_location_without_gradient)
            self.__predicted_cursor_trail.append(self.predicted_cursor_location)
            self.get_predicted_cursor_trail = tf.stack(self.__predicted_cursor_trail)
        self.get_cursor_trail = tf.stack(self.__cursor_trail)
        return output


    def save_model(self, sess, save_path):
        print("start saving")
        self._saver.save(sess, save_path)
        print("model saved at {}".format(save_path))


    def load_model(self, sess, save_path):
        print("start loading")
        self._saver.restore(sess, save_path)
        print("model loaded from {}".format(save_path))


if __name__ == "__main__":
    from parameter import model_hyper_parameter as hp
    start_time = time.time()
    model = Model(hp)
    model.run(int(2500/hp["delta_t"]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        target_location = np.ones((hp["batch_size"], 2), dtype=np.float32)
        activity_list, cursor_trail, weight_list = sess.run([model.activity_list, model.get_cursor_trail, model.weight_list],
                                                feed_dict={model.target_location:target_location})
        show_activity(activity_list[0], 1)
        show_connection(weight_list[0])
        show_cursor_trail(cursor_trail, 1, target_location=target_location)
    print("run time: {0:.2f}s".format(time.time()-start_time))
