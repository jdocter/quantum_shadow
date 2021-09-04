import tensorflow as tf
import numpy as np
from tensorflow import keras

class ShadowQSP(keras.layers.Layer):
    """Quantum shadow interrogation trainable circuit
       
    """

    def __init__(self, poly_deg=0, trainable_map=[]):
        """
        Params
        ------
        poly_deg: The desired degree of the polynomial in the QSP picture
        
        trainable_map: list of bools. trainable[i] indicates if the ith phi should be trainable
        """
        super(ShadowQSP, self).__init__()
        self.poly_deg = poly_deg
        phi_init = tf.random_uniform_initializer(minval=0, maxval=np.pi)
#         phi_init = tf.random_normal_initializer(mean=0.0, stddev=0.005, seed=None)
        
        if len(trainable_map) == 0:
            self.trainable_map = [True]*poly_deg
        else:
            self.trainable_map = trainable_map
           
        self.trainable_phis = tf.Variable(
            initial_value=phi_init(shape=(poly_deg, 1), dtype=tf.float32),
            trainable=True,
        )
        
        self.fixed_phis = tf.Variable(
            initial_value=phi_init(shape=(poly_deg, 1), dtype=tf.float32),
            trainable=False,
        )
        
    def unitary(self, inp, verbose=False):
        """
        Shadow QSP 
        q0 ---
        q1 ---
        Order 
            phi[q1], controlled U[q0,q1], V[q0]
            d times
        """
        th = inp[:,None,0]
        eps = inp[:,None,1]
        batch_dim = tf.gather(tf.shape(th), 0)
        
        
        px = tf.constant([[0.0, 1], [1, 0]], dtype=tf.complex64)
        px = tf.expand_dims(px, axis=0)
        px = tf.repeat(px, [batch_dim], axis=0)
        
        rot_x_arg = tf.complex(real=0.0, imag=th)
        rot_x_arg = tf.expand_dims(rot_x_arg, axis=1)
        rot_x_arg = tf.tile(rot_x_arg, [1, 2, 2])
        

        wx = tf.linalg.LinearOperatorFullMatrix(tf.linalg.expm(tf.multiply(px, rot_x_arg)))
        
        c_op = tf.linalg.LinearOperatorFullMatrix(tf.constant([[0, 0], [0, 1]], dtype=tf.complex64))
        c_op = tf.linalg.LinearOperatorKronecker([c_op, wx])
        i_op = tf.constant([[[1.0,0,0,0], 
                            [0,1,0,0],
                            [0,0,0,0],
                            [0,0,0,0]]], dtype=tf.complex64)
        i_op = tf.repeat(i_op, [batch_dim], axis=0)
        c_op = c_op.to_dense() + i_op
        
        # tiled up V rotations
        px = tf.constant([[0.0, 1], [1, 0]], dtype=tf.complex64)
        px = tf.expand_dims(px, axis=0)
        px = tf.repeat(px, [batch_dim], axis=0)
        
        
        rot_v_arg = tf.complex(real=0.0, imag=eps)
        rot_v_arg = tf.expand_dims(rot_v_arg, axis=1)
        rot_v_arg = tf.tile(rot_v_arg, [1, 2, 2])
        
        rv = tf.linalg.LinearOperatorFullMatrix(tf.linalg.expm(tf.multiply(px, rot_v_arg)))
        
        pi = tf.linalg.LinearOperatorFullMatrix(tf.constant([[1, 0], [0, 1]], dtype=tf.complex64))
        v_op = tf.linalg.LinearOperatorKronecker([rv, pi]).to_dense()
        
        # tiled up Z rotations
        pz = tf.constant([[1.0, 0], [0, -1]], dtype=tf.complex64)
        pz = tf.expand_dims(pz, axis=0)
        pz = tf.repeat(pz, [batch_dim], axis=0)

        z_rotations = []
        for k in range(self.poly_deg):
            phi = self.trainable_phis[k] if self.trainable_map[k] else self.fixed_phis[k]
            rot_z_arg = tf.complex(real=0.0, imag=phi)
            rot_z_arg = tf.expand_dims(rot_z_arg, axis=0)
            rot_z_arg = tf.expand_dims(rot_z_arg, axis=0)
            rot_z_arg = tf.tile(rot_z_arg, [batch_dim, 2, 2])

            rz = tf.linalg.LinearOperatorFullMatrix(tf.linalg.expm(tf.multiply(pz, rot_z_arg)))
            pi = tf.linalg.LinearOperatorFullMatrix(tf.constant([[1, 0], [0, 1]], dtype=tf.complex64))
            rz = tf.linalg.LinearOperatorKronecker([pi, rz]).to_dense()
            z_rotations.append(rz)
            
        extra = []
        # initial state 

        init_state = self.random_init_state(batch_dim)
        
        u = z_rotations[0]
        u = tf.matmul(c_op, u)
        u = tf.matmul(v_op, u)
        for rz in z_rotations[1:]:
            if verbose:
                final_state = tf.matmul(u,init_state)
                extra.append(tf.norm(tf.reshape(final_state, (batch_dim, 2, 2))[:,:,0],axis=1))
            u = tf.matmul(rz, u)
            u = tf.matmul(c_op, u)
            u = tf.matmul(v_op, u)
        
        final_state = tf.matmul(u,init_state)
        
        return u, final_state, extra 
        
        
    
    
    def expectation(self, inp, verbose=False):
        batch_dim = tf.gather(tf.shape(inp[:,None,0]), 0)
        u, final_state, init_state = self.unitary(inp, verbose)
        
        # measure IZ operator
        pi = tf.linalg.LinearOperatorFullMatrix(tf.constant([[1., 0], [0, 1]], dtype=tf.complex64))
        pz = tf.linalg.LinearOperatorFullMatrix(tf.constant([[1., 0], [0, -1]], dtype=tf.complex64))
        measurement_op = tf.linalg.LinearOperatorKronecker([pi, pz]).to_dense()
        measurement_op = tf.expand_dims(measurement_op, axis=0)
        measurement_op = tf.repeat(measurement_op, [batch_dim], axis=0)
        
        expectation = tf.matmul(final_state, tf.linalg.matmul(measurement_op, final_state), adjoint_a=True)
        expectation = tf.squeeze(expectation,2)
        
        if verbose:
            return expectation, extra
        return expectation
        
    def random_init_state(self, batch_dim):
        """
        random initial state (up to a global phase)
        """
        px = tf.tile(tf.constant([[[0.0,1], [1,0]]], dtype=tf.complex64),[batch_dim,1,1])

        init_env_rot_ang = tf.tile(
            tf.complex(
                real=0.0, 
                imag=tf.random.uniform(
                    (batch_dim,1,1), minval=0, maxval=np.pi/2, dtype=tf.dtypes.float32
                )
            ),
            [1,2,2]
        )

        init_env_rot = tf.linalg.LinearOperatorFullMatrix(tf.linalg.expm(tf.multiply(px, init_env_rot_ang)))
        pi = tf.linalg.LinearOperatorFullMatrix(tf.constant([[1,0],[0,1]], dtype=tf.complex64))
        init_env_rot = tf.linalg.LinearOperatorKronecker([init_env_rot,pi]).to_dense()
        init_state = tf.matmul(init_env_rot, [[1],[0],[0],[0]])
        
        return init_state
    
    def call(self, inp):
        return self.expectation(inp)

    
class QSIM(ShadowQSP):
    """ Qrapper class to call specific matrix element"""
    def __init__(self, poly_deg=0, trainable_map=[], i=0,j=0):
        super(QSIM, self).__init__(poly_deg, trainable_map)
        self.i = i
        self.j = j
              
    def call(self, inp):
        return self.unitary(inp)[0][:,self.i,self.j]


def construct_qsp_model(poly_deg,trainable_map=[]):
    """Helper function that compiles a QSP model with mean squared error and adam optimizer.

    Params
    ------
    poly_deg : int
        the desired degree of the polynomial in the QSP sequence.

    Returns
    -------
    Keras model
        a compiled keras model with trainable phis in a poly_deg QSP shadow sequence.
    """
    theta_input = tf.keras.Input(shape=(2,), dtype=tf.float32, name="theta")
    qsp = ShadowQSP(poly_deg,trainable_map)
    u = qsp(theta_input)
    model = tf.keras.Model(inputs=theta_input, outputs=u)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    return qsp, model

def construct_qsim_model(poly_deg,trainable_map=[],i=0,j=0):
    """Helper function that compiles a QSP model with mean squared error and adam optimizer.

    Params
    ------
    poly_deg : int
        the desired degree of the polynomial in the QSP sequence.

    Returns
    -------
    Keras model
        a compiled keras model with trainable phis in a poly_deg QSP sequence.
    """
    theta_input = tf.keras.Input(shape=(2,), dtype=tf.float32, name="theta")
    qsp = QSIM(poly_deg,trainable_map)
    u = qsp(theta_input)
    model = tf.keras.Model(inputs=theta_input, outputs=u)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    return qsp, model
