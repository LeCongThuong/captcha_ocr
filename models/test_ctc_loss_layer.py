from models.ctc_loss_layer import CtcLayer
import numpy as np


def test_ctc_loss_layer():
    ctc_layer = CtcLayer(name='ctc_loss')
    y_true = np.array([[1, 0, 1, 2],
                       [1, 4, 3, 3],
                       [1, 2, 4, 4],
                       [0, 1, 3, 3]], dtype=np.int64)

    y_pred = np.array([[[0.31524086, 0.89162359, 0.71370758, 0.54636533],
                        [0.70043564, 0.33405933, 0.59145792, 0.86413263],
                        [0.95441504, 0.71829488, 0.06930456, 0.93956957],
                        [0.00606225, 0.93815141, 0.34609953, 0.79783431]],

                       [[0.80371128, 0.7601663, 0.74667121, 0.30740272],
                        [0.23876815, 0.89897338, 0.22401414, 0.34941368],
                        [0.30304062, 0.52427138, 0.52566048, 0.30300152],
                        [0.49742669, 0.08680141, 0.66558714, 0.95689286]],

                       [[0.48097162, 0.87980914, 0.28458129, 0.47284835],
                        [0.60424958, 0.60116879, 0.67794718, 0.99748801],
                        [0.25020843, 0.6637288, 0.99357695, 0.17389025],
                        [0.4623111, 0.52276955, 0.04246024, 0.17038571]],

                       [[0.7546928, 0.97589769, 0.87277136, 0.09801435],
                        [0.7990567, 0.22919661, 0.14579512, 0.22081467],
                        [0.14589717, 0.3418751, 0.12832073, 0.61680037],
                        [0.35039222, 0.49164302, 0.00630585, 0.72559287]]])

    ctc_loss = ctc_layer(y_true, y_pred)
    return ctc_loss
