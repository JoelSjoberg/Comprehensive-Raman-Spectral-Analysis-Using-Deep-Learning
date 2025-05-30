from Scripts.essentials import *
# The generator
from Scripts.generator import train_generator

# Make a validation set
reset_seed(12345)
for i in train_generator(length = 2000, batch_size = 1000, min_width = 5, max_width = 300, max_num_peaks = 5):
    val_X = i[0]
    val_y = i[1]
    break

def decomposer_model(lr = 0.00001):
    
    reset_seed(SEED = 0)  
    scaler = 10
    dim_red_size = 12**2
    l1_param = 1e-6
    l2_param = 1e-6
    
    inp = Input(shape = (None,1))
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(inp)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    inp_key = Conv1D(filters = dim_red_size, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    inp_key = GlobalMaxPooling1D()(inp_key)
    inp_key = tf.keras.backend.expand_dims(inp_key, -1)
    
    t_dot = Dot(axes=(2, 2))([inp, inp_key])
    t = tf.keras.backend.expand_dims(t_dot, -1)
    
    kernel_size = 32
    t = Conv2D(filters = 1 * scaler,
               kernel_size = (kernel_size, int(dim_red_size/10)),
               strides = (1, int(np.sqrt(dim_red_size))),
               padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 16
    t_res = Conv2D(filters = 1 * scaler, 
              kernel_size = (kernel_size, int(dim_red_size/10)),
              strides = (1, int(np.sqrt(dim_red_size))),
              padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t_res)
    t = LeakyReLU()(t)
    
    t = Reshape((-1, scaler))(t)
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    # Baseline
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    b = Conv1D(filters = 1, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    
    pool_size = 33
    padded = CustomPad(pool_size)(b)
    b = AveragePooling1D(pool_size, 1, padding = "valid")(padded)
    b = tf.keras.activations.relu(b)
    
    # Cosmic rays
    cr_size = 3
    cr_padded = CustomPad(cr_size)(t)
    cr = Conv1D(filters = 1, kernel_size = cr_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(cr_padded)

    cr = tf.keras.activations.relu(cr)

    ### Second part: Extract peaks and noise from the input
    reduced_spectrum = Add()([inp, -b, -cr])
    kernel_size = 128
    padded = CustomPad(kernel_size)(reduced_spectrum)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    inp_key = Conv1D(filters = dim_red_size, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    inp_key = GlobalMaxPooling1D()(inp_key)
    inp_key = tf.keras.backend.expand_dims(inp_key, -1)
    
    t_dot = Dot(axes=(2, 2))([reduced_spectrum, inp_key])
    t = tf.keras.backend.expand_dims(t_dot, -1)
    
    kernel_size = 32
    t = Conv2D(filters = 1 * scaler,
               kernel_size = (kernel_size, int(dim_red_size/10)),
               strides = (1, int(np.sqrt(dim_red_size))),
               padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 16
    t = Conv2D(filters = 1 * scaler, 
              kernel_size = (kernel_size, int(dim_red_size/10)),
              strides = (1, int(np.sqrt(dim_red_size))),
              padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    
    t = Add()([t, t_res])
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    t = Reshape((-1, scaler))(t)
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    
    # Peaks, make the kernel size small to deal with potentially sharp peaks
    kernel_size = 8 
    padded = CustomPad(kernel_size)(t)
    p = Conv1D(filters = 1, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    pool_size = 4
    padded = CustomPad(pool_size)(p)
    p = AveragePooling1D(pool_size, 1, padding = "valid")(padded)
    p = tf.keras.activations.relu(p)

    n = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)

    n = tf.keras.activations.tanh(n)


    # Flatten each part to make them comparable to the labels
    b = Flatten()(b)
    cr = Flatten()(cr)
    n = Flatten()(n)
    p = Flatten()(p)
    
    # Store output in a list which we return
    output = [b, cr, n, p]
    
    model = Model(inp, output)
    
    model.compile(
        optimizer= Adam(learning_rate=lr),
        loss= joined_loss,
        metrics = [
            root_mean_squared_error,
            r_square,
            ])

    return model
gc.collect()

step_scaler = 2
boundaries = list(np.array([100000, 200000, 300000,
              400000, 500000, 600000,
              700000, 800000, 900000,
              1000000, 1100000]) * step_scaler)
values = list(np.array([0.0007, 0.0005, 0.0003, 0.0001,
          0.00007, 0.00005, 0.00003, 0.00001,
          0.000007, 0.000005, 0.000003, 0.000001]) * 1.5)
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

model = decomposer_model(lr = learning_rate_fn)
model.summary()

# Fit the model
model.fit(train_generator(length = 750, 
                          batch_size=100, 
                          min_width = 5,
                          max_width = 256,
                          max_num_peaks = 5), # Define the length of the spectra in wavelengths and batch size
    steps_per_epoch = 100000*step_scaler,     # How many times a batch is created per epoch
    epochs = 12,                     # How many epochs
    validation_data = (val_X, val_y) # Validation data
)
# Save model weights, note that you need to save the architecture (the code that defined the model above)
# in order to recrreate the model using these weights! Therefore it is vital that you copy and paste the 
# code above and document it alongside the saved weights!
model.to_json()
model.save_weights("Models/ensembleModelRes.h5")